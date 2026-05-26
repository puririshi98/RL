# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os
import random
import re
import time
import uuid
from typing import Any, Dict, List, NotRequired, Optional, Tuple, TypedDict

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn
from nemo_rl.environments.metrics import calculate_pass_rate_per_prompt
from nemo_rl.environments.utils import chunk_list_to_workers


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# Substrings that indicate a transient upstream API failure worth retrying.
_TRANSIENT_HTTP_PHRASES = (
    "502", "503", "504",
    "bad gateway", "service unavailable",
    "timeout", "timed out",
    "429", "rate limit",
)

_JUDGE_SYSTEM_PROMPT = """\
You are evaluating a customer service AI agent. You will be given the domain rules and \
policies the agent must follow, the customer's original request, and the full conversation.

Score the agent from 0.0 to 1.0 based on:
- Whether the customer's request was correctly fulfilled
- Whether all domain policies and rules were followed
- Whether communication was clear and professional

Respond with JSON only: {"score": <float 0-1>, "reasoning": "<one sentence>"}"""


class TauBenchEnvConfig(TypedDict):
    """Configuration for TauBenchEnvironment.

    Attributes:
        num_workers: Number of Ray worker actors for parallel execution.
        env_name: tau-bench domain to run ("retail" or "airline").
        task_split: Dataset split ("train", "test", or "dev").
        user_strategy: How the simulated user behaves ("llm", "react", "verify",
            "reflection", or "mock"). Use "mock" to run without any real LLM calls:
            the user simulator and judge return random outputs after an optional sleep,
            and user_base_url / judge_base_url are ignored.
        user_model: LiteLLM model string for the user simulator (e.g. "openai/gpt-4o-mini").
            Ignored when user_strategy is "mock".
        user_base_url: Optional base URL override for the user model's OpenAI-compatible API.
            Ignored when user_strategy is "mock".
        user_api_key: API key for the user model server. Ignored when user_strategy is "mock".
        max_steps: Maximum tool-call steps per episode before forced termination.
        judge_model: LiteLLM model string for LLM-as-judge scoring (None disables judging).
            Ignored when user_strategy is "mock".
        judge_base_url: Optional base URL override for the judge model's API.
            Ignored when user_strategy is "mock".
        judge_api_key: API key for the judge model server. Ignored when user_strategy is "mock".
        judge_weight: Blend weight: 0.0 = pure tau-bench reward, 1.0 = pure judge score.
        worker_stagger_delay_s: Seconds to delay each successive worker's first API call.
            Spreading requests avoids a thundering-herd burst that can overwhelm the
            inference endpoint. Worker i sleeps i * worker_stagger_delay_s before its
            first API call. Default: 1.0.
        mock_user_latency_s: Seconds to sleep per mock user simulator call.
        mock_judge_latency_s: Seconds to sleep per mock judge call.
        mock_stop_prob: Probability that the mock user simulator ends the conversation on
            any given turn by emitting ###STOP###.
    """

    num_workers: int
    env_name: str
    task_split: str
    user_strategy: str
    user_model: str
    user_base_url: NotRequired[str | None]
    user_api_key: NotRequired[str | None]
    max_steps: int
    judge_model: NotRequired[str | None]
    judge_base_url: NotRequired[str | None]
    judge_api_key: NotRequired[str | None]
    judge_weight: float
    worker_stagger_delay_s: NotRequired[float]
    mock_user_latency_s: NotRequired[float | None]
    mock_judge_latency_s: NotRequired[float | None]
    mock_stop_prob: NotRequired[float | None]


class TauBenchEnvMetadata(TypedDict):
    """Per-episode state carried through nemo-rl's metadata channel.

    Attributes:
        task_index: Which tau-bench task index this episode is running.
        episode_id: Unique ID assigned on the first step; None before that.
        step_count: Number of environment steps taken in this episode so far.
        tau_reward: tau-bench task-completion reward, set when the episode terminates.
        judge_score: LLM-judge score, set when the episode terminates (None if judging is off).
    """

    task_index: int
    episode_id: str | None
    step_count: int
    tau_reward: NotRequired[float | None]
    judge_score: NotRequired[float | None]


@ray.remote  # pragma: no cover
class TauBenchWorker:
    """Executes tau-bench environment steps for a chunk of rollout samples.

    Each active episode gets its own ``tau_bench.Env`` instance, stored by episode_id,
    so that database state persists correctly across multi-turn rollouts.
    """

    def __init__(
        self,
        env_name: str,
        task_split: str,
        user_strategy: str,
        user_model: str,
        user_base_url: Optional[str],
        user_api_key: str,
        max_steps: int,
        judge_model: Optional[str],
        judge_base_url: Optional[str],
        judge_api_key: str,
        mock_user_latency_s: Optional[float] = None,
        mock_judge_latency_s: Optional[float] = None,
        mock_stop_prob: Optional[float] = None,
    ) -> None:
        self._env_name = env_name
        self._task_split = task_split
        self._user_strategy = user_strategy
        self._user_model = user_model
        self._max_steps = max_steps
        self._judge_model = judge_model
        self._judge_base_url = judge_base_url
        self._judge_api_key = judge_api_key
        self._mock_user = user_strategy == "mock"
        self._mock_judge = judge_model == "mock"
        self._active_envs: Dict[str, Any] = {}

        if self._mock_user:
            assert mock_user_latency_s is not None, "mock_user_latency_s must be set when user_strategy='mock'"
            assert mock_stop_prob is not None, "mock_stop_prob must be set when user_strategy='mock'"
        if self._mock_judge:
            assert mock_judge_latency_s is not None, "mock_judge_latency_s must be set when judge_model='mock'"
        self._mock_judge_latency_s = mock_judge_latency_s

        if self._mock_user:
            # Build a mock litellm.completion that returns random customer-like text.
            # Applied as a scoped patch only around tau-bench's reset/step calls so
            # other litellm users in the same worker process are not affected.
            _latency = mock_user_latency_s
            _stop_prob = mock_stop_prob
            _responses = [
                "I need help with a recent order.",
                "Can you look into this for me?",
                "I'd like to return an item.",
                "What is the status of my request?",
                "That works for me, thank you.",
            ]

            class _Msg:
                def __init__(self, content: str) -> None:
                    self.content = content
                    self.role = "assistant"

                def model_dump(self) -> dict:
                    return {"role": self.role, "content": self.content}

            class _Choice:
                def __init__(self, content: str) -> None:
                    self.message = _Msg(content)

            class _Resp:
                def __init__(self, content: str) -> None:
                    self.choices = [_Choice(content)]
                    self._hidden_params = {"response_cost": 0.0}

            def _mock_completion(*args: Any, **kwargs: Any) -> "_Resp":
                time.sleep(_latency)
                content = "###STOP###" if random.random() < _stop_prob else random.choice(_responses)
                return _Resp(content)

            self._mock_completion = _mock_completion
        else:
            # Configure LiteLLM (used by tau-bench's user simulator) once at startup.
            if user_base_url:
                os.environ["OPENAI_BASE_URL"] = user_base_url
            if user_api_key:
                os.environ["OPENAI_API_KEY"] = user_api_key
            elif "OPENAI_API_KEY" not in os.environ:
                os.environ["OPENAI_API_KEY"] = "dummy"
            # Set a global request timeout so truly hung API calls fail within a
            # bounded time. The timeout must be long enough to accommodate
            # legitimately slow inference responses (the NVIDIA inference endpoint
            # can take 60-90s under load); a timeout that is too short converts
            # normal slow responses into spurious failures and retry storms.
            # 120s gives ample headroom for slow responses while still bounding
            # genuinely stuck connections (vs the OS TCP timeout of ~10 min).
            import litellm as _litellm
            _litellm.request_timeout = 120

    def _call_with_retry(self, fn: Any, max_retries: int = 5, base_delay: float = 2.0) -> Any:
        """Call fn, retrying on transient API errors with exponential back-off.

        Uses full-jitter backoff (uniform sample in [0, delay]) so that many
        workers retrying simultaneously don't all hit the API at the same instant.

        Re-raises as RuntimeError on exhaustion so Ray can serialize the
        exception across actor boundaries (litellm exception classes are not
        Ray-serializable and produce UnserializableException in the driver).
        """
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"[TauBench] API call failed after {max_retries + 1} attempts: {e}"
                    ) from None
                msg = str(e).lower()
                if any(phrase in msg for phrase in _TRANSIENT_HTTP_PHRASES):
                    delay = random.uniform(0, base_delay * (2 ** attempt))
                    print(
                        f"[TauBench] transient API error on attempt {attempt + 1}/{max_retries + 1},"
                        f" retrying in {delay:.1f}s: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"[TauBench] non-retryable API error: {type(e).__name__}: {e}") from None

    def _make_env(self, task_index: int) -> tuple:
        """Create and reset a tau-bench Env for a specific task index.

        Returns:
            (env, initial_obs) where initial_obs is the customer's opening
            message generated by the user simulator during env.reset().
        """
        from tau_bench.envs import get_env

        # tau_bench expects provider and model separately; split the LiteLLM
        # "provider/model" format used in the config (e.g. "openai/gpt-4o-mini").
        user_model = self._user_model or "mock"
        user_provider = None
        if "/" in user_model:
            user_provider, user_model = user_model.split("/", 1)

        # tau-bench does not know about the "mock" strategy; translate it to "llm"
        # so it instantiates an LLMUserSimulationEnv whose litellm.completion calls
        # are intercepted by the mock installed in __init__.
        tau_user_strategy = "llm" if self._mock_user else self._user_strategy
        # llm strategy requires a provider; default to "openai" in mock mode.
        if self._mock_user:
            user_provider = user_provider or "openai"

        if self._mock_user:
            from unittest.mock import patch
            env = get_env(
                env_name=self._env_name,
                user_strategy=tau_user_strategy,
                user_model=user_model,
                task_split=self._task_split,
                user_provider=user_provider,
                task_index=task_index,
            )
            with patch("litellm.completion", self._mock_completion):
                reset_response = env.reset(task_index=task_index)
        else:
            # Both get_env() and env.reset() make LLM API calls:
            # LLMUserSimulationEnv.__init__ calls self.reset() during get_env(),
            # and env.reset() generates the customer's opening message.
            # Wrap both together so a transient failure retries from scratch.
            def _create_and_reset() -> tuple:
                env = get_env(
                    env_name=self._env_name,
                    user_strategy=tau_user_strategy,
                    user_model=user_model,
                    task_split=self._task_split,
                    user_provider=user_provider,
                    task_index=task_index,
                )
                return env, env.reset(task_index=task_index)

            env, reset_response = self._call_with_retry(_create_and_reset)
        initial_obs = str(reset_response.observation)
        return env, initial_obs

    def _parse_action(self, agent_text: str) -> Any:
        """Convert agent-generated text into a tau_bench Action.

        Text wrapped in ``<tool_call>...</tool_call>`` is parsed as a JSON tool
        invocation.  Anything else is treated as a ``respond`` action (the agent's
        final message to the user).
        """
        from tau_bench.types import Action

        # The stop string "</tool_call>" causes the model to halt before emitting
        # the closing tag.  Restore it so the regex can match.
        if "<tool_call>" in agent_text and "</tool_call>" not in agent_text:
            agent_text = agent_text + "</tool_call>"

        match = _TOOL_CALL_RE.search(agent_text)
        if match:
            try:
                payload = json.loads(match.group(1).strip())
                return Action(
                    name=payload["name"],
                    kwargs=payload.get("arguments", payload.get("kwargs", {})),
                )
            except (json.JSONDecodeError, KeyError):
                pass
        # Guard against empty content: some model APIs (including NVIDIA NIM) reject
        # messages with empty string content.
        content = agent_text.strip()
        return Action(name="respond", kwargs={"content": content or "[no response]"})

    def _call_judge(
        self,
        conversation: List[Dict[str, str]],
        domain_rules: str,
        task_instruction: str,
    ) -> float:
        """Score a completed episode using an LLM judge.

        Returns a float in [0, 1].  Returns 0.0 on any error so that a judge
        misconfiguration degrades gracefully rather than crashing training.
        """
        if self._mock_judge:
            time.sleep(self._mock_judge_latency_s)
            return random.random()

        import litellm

        convo_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in conversation
        )
        user_prompt = (
            f"DOMAIN RULES:\n{domain_rules}\n\n"
            f"CUSTOMER REQUEST:\n{task_instruction}\n\n"
            f"CONVERSATION:\n{convo_text}"
        )
        try:
            resp = litellm.completion(
                model=self._judge_model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                timeout=60,
                api_base=self._judge_base_url,
                api_key=self._judge_api_key,
            )
            content = resp.choices[0].message.content
            return float(json.loads(content).get("score", 0.0))
        except Exception as e:
            print(f"[TauBench] judge call failed: {e}")
            return 0.0

    def execute(
        self,
        message_batch: List[str],
        metadata_batch: List[TauBenchEnvMetadata],
        judge_weight: float,
        initial_delay_s: float = 0.0,
    ) -> Tuple[
        List[Dict[str, str]],
        List[bool],
        List[float],
        List[TauBenchEnvMetadata],
    ]:
        """Run one environment step for each sample in this worker's chunk.

        Args:
            message_batch: Latest assistant response texts, one per sample.
            metadata_batch: Per-episode state dicts, one per sample.
            judge_weight: Blending weight between tau-bench reward and judge score.
            initial_delay_s: Seconds to sleep before issuing any API request.
                Used to stagger workers so they don't all hit the endpoint at once.

        Returns:
            Tuple of (observations, terminateds, rewards, updated_metadata).
        """
        if initial_delay_s > 0.0:
            time.sleep(initial_delay_s)
        observations: List[Dict[str, str]] = []
        terminateds: List[bool] = []
        rewards: List[float] = []
        new_metadata: List[TauBenchEnvMetadata] = []

        for agent_text, meta in zip(message_batch, metadata_batch):
            task_index: int = meta["task_index"]
            step_count: int = meta.get("step_count", 0)
            episode_id: str = meta.get("episode_id") or str(uuid.uuid4())

            if episode_id not in self._active_envs:
                tau_env, initial_obs = self._make_env(task_index)
                self._active_envs[episode_id] = {
                    "env": tau_env,
                    "initial_obs": initial_obs,
                    "conversation": [],  # list of (label, text) for episode logging
                }

            env_state = self._active_envs[episode_id]
            tau_env = env_state["env"]
            conversation: List[Tuple[str, str]] = env_state["conversation"]

            if env_state.get("initial_obs") is not None:
                # Pre-step: return the user simulator's actual opening message
                # as the first observation.  The agent's generation at this point
                # was based on a placeholder user message in the dataset and is
                # discarded here; the agent will generate its real first response
                # on the next turn after seeing the customer's actual message.
                initial_obs = env_state.pop("initial_obs")
                # Record the wasted agent response and the real customer opening.
                conversation.append(("AGENT", agent_text.strip()))
                conversation.append(("CUSTOMER", initial_obs.strip()))
                #print(
                #    f"[TauBench] episode={episode_id}  task={task_index}"
                #    f"  PRE-STEP: customer: {initial_obs[:200]}",
                #    flush=True,
                #)
                observations.append({"role": "user", "content": initial_obs})
                terminateds.append(False)
                rewards.append(0.0)
                new_metadata.append({
                    "task_index": task_index,
                    "episode_id": episode_id,
                    "step_count": step_count,  # not incremented: no real action taken
                    "tau_reward": None,
                    "judge_score": None,
                })
                continue

            action = self._parse_action(agent_text)
            if self._mock_user:
                from unittest.mock import patch
                with patch("litellm.completion", self._mock_completion):
                    env_response = tau_env.step(action)
            else:
                env_response = self._call_with_retry(lambda: tau_env.step(action))
            step_count += 1
            #print(
            #    f"[TauBench] episode={episode_id}  task={task_index}"
            #    f"  step={step_count}  action={action.name}",
            #    flush=True,
            #)

            # Record this agent turn and the resulting observation.
            obs_role = "user" if action.name == "respond" else "tool"
            obs_label = "CUSTOMER" if obs_role == "user" else "TOOL"
            conversation.append(("AGENT", agent_text.strip()))
            conversation.append((obs_label, str(env_response.observation).strip()))

            done = bool(env_response.done) or step_count >= self._max_steps
            reward = 0.0
            tau_reward: Optional[float] = None
            judge_score: Optional[float] = None

            if done:
                if env_response.done:
                    # tau_env.step() already called calculate_reward() internally when
                    # the episode ended naturally; reuse that result.  Calling it again
                    # would compare the gold-replayed DB state against itself, always
                    # returning 1.0.
                    tau_reward = float(env_response.reward)
                else:
                    # Force-terminated by max_steps — task incomplete.
                    tau_reward = 0.0
                if self._judge_model and judge_weight > 0.0:
                    rules = "\n".join(getattr(tau_env, "rules", []))
                    tasks = getattr(tau_env, "tasks", [])
                    instruction = tasks[tau_env.task_index].instruction if tasks else ""
                    user_sim = getattr(tau_env, "user", None)
                    user_msgs = getattr(user_sim, "messages", []) if user_sim else []
                    # From the user sim's POV, role="user" is the agent speaking and
                    # role="assistant" is the customer speaking — flip for judge.
                    convo = [
                        {"role": "assistant" if m["role"] == "user" else "user", "content": m["content"]}
                        for m in user_msgs
                        if m.get("role") in ("user", "assistant")
                    ]
                    judge_score = self._call_judge(convo, rules, instruction)
                    reward = (1.0 - judge_weight) * tau_reward + judge_weight * judge_score
                else:
                    reward = tau_reward

                # Log episode summary with clean text (no template markup).
                blended = reward
                tasks_list = getattr(tau_env, "tasks", [])
                task_instruction = (
                    tasks_list[task_index].instruction[:150]
                    if tasks_list and task_index < len(tasks_list)
                    else ""
                )
                score_str = f"tau_reward={tau_reward:.3f}"
                if judge_score is not None:
                    score_str += f"  judge_score={judge_score:.3f}"
                score_str += f"  blended={blended:.3f}"
                #print(f"\n{'='*70}")
                #print(
                #    f"[TauBench] episode_id={episode_id}  task_index={task_index}  steps={step_count}\n"
                #    f"  {score_str}"
                #)
                #if task_instruction:
                #    print(f"  task: {task_instruction}")
                #print("-" * 70)
                #print("  CONVERSATION:")
                for turn_num, (label, text) in enumerate(conversation, 1):
                    if len(text) > 300:
                        text = text[:300] + " ..."
                    #print(f"  [{turn_num}] {label}: {text}")
                #print("-" * 70)
                tool_calls: List[Tuple[str, Any]] = []
                for _label, _text in conversation:
                    if _label == "AGENT":
                        for _m in _TOOL_CALL_RE.finditer(_text):
                            try:
                                _p = json.loads(_m.group(1).strip())
                                tool_calls.append((_p.get("name", "?"), _p.get("arguments", {})))
                            except (json.JSONDecodeError, KeyError):
                                pass
                """if tool_calls:
                    print("  AGENT ACTIONS (tool calls):")
                    for i, (name, kwargs) in enumerate(tool_calls):
                        print(f"  [{i}] {name}: {kwargs}")
                else:
                    print("  AGENT ACTIONS (tool calls): none")
                print("=" * 70, flush=True)"""

                del self._active_envs[episode_id]

            observations.append(
                {"role": obs_role, "content": str(env_response.observation)}
            )
            terminateds.append(done)
            rewards.append(reward)
            new_metadata.append(
                {
                    "task_index": task_index,
                    "episode_id": episode_id,
                    "step_count": step_count,
                    "tau_reward": tau_reward,
                    "judge_score": judge_score,
                }
            )

        return observations, terminateds, rewards, new_metadata


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class TauBenchEnvironment(EnvironmentInterface[TauBenchEnvMetadata]):
    """Multi-turn customer-service environment backed by tau-bench.

    The agent communicates with a simulated user across multiple turns, calling
    domain tools (e.g. order management, flight rescheduling) to complete tasks.
    Rewards blend tau-bench's deterministic task-completion check with an optional
    LLM-as-judge score for policy compliance and communication quality.

    **Tool call format**: the policy must wrap tool calls in ``<tool_call>`` tags::

        <tool_call>{"name": "cancel_order", "arguments": {"order_id": "O123"}}</tool_call>

    Any generation that does not contain a ``<tool_call>`` block is treated as a
    final ``respond`` action (the agent sending a message to the user), which may
    trigger a user-simulator turn or end the episode.

    **Stop strings**: ``["</tool_call>"]``.

    **Data format**: each dataset sample's ``extra_env_info`` must contain at minimum
    ``{"task_index": <int>, "episode_id": null, "step_count": 0}``.  The initial
    ``message_log`` should be constructed by calling ``tau_bench.envs.get_env`` and
    ``env.reset(task_index=N)`` during data preparation to ensure prompt/env
    consistency.
    """

    def __init__(self, cfg: TauBenchEnvConfig) -> None:
        self.cfg = cfg
        self._num_workers = cfg["num_workers"]
        self._judge_weight = float(cfg["judge_weight"])
        self._stagger_delay_s = float(cfg.get("worker_stagger_delay_s", 1.0))

        # TauBenchWorkers call external LLM APIs that may only be reachable from
        # specific nodes (e.g. the head node has outbound internet, compute nodes
        # do not).  Pin workers to the same node that hosts TauBenchEnvironment
        # itself so they share the same network access.
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        _this_node = ray.get_runtime_context().get_node_id()
        _schedule_here = NodeAffinitySchedulingStrategy(
            node_id=_this_node, soft=False
        )

        self._workers = [
            TauBenchWorker.options(
                runtime_env={"py_executable": PY_EXECUTABLES.TAU_BENCH},
                scheduling_strategy=_schedule_here,
            ).remote(
                env_name=cfg["env_name"],
                task_split=cfg["task_split"],
                user_strategy=cfg["user_strategy"],
                user_model=cfg["user_model"],
                user_base_url=cfg.get("user_base_url"),
                user_api_key=cfg.get("user_api_key") or "dummy",
                max_steps=cfg["max_steps"],
                judge_model=cfg.get("judge_model"),
                judge_base_url=cfg.get("judge_base_url"),
                judge_api_key=cfg.get("judge_api_key") or "dummy",
                mock_user_latency_s=cfg.get("mock_user_latency_s"),
                mock_judge_latency_s=cfg.get("mock_judge_latency_s"),
                mock_stop_prob=cfg.get("mock_stop_prob"),
            )
            for _ in range(self._num_workers)
        ]

    def obs_use_chat_template(self) -> bool:
        return True

    def step(
        self,
        message_log_batch: List[LLMMessageLogType],
        metadata_batch: List[TauBenchEnvMetadata],
    ) -> EnvironmentReturn[TauBenchEnvMetadata]:
        """Run one environment step for each sample in the batch.

        Only the latest assistant message is forwarded to tau-bench; earlier turns
        are already captured in the tau-bench env's internal message history.

        Args:
            message_log_batch: Full conversation histories; only the last message
                               (the most recent assistant turn) is used.
            metadata_batch: Per-episode state, including task_index and episode_id.

        Returns:
            EnvironmentReturn with tool results or user responses as observations.
            Rewards are non-zero only on terminal steps.
        """
        message_batch = [ml[-1]["content"] for ml in message_log_batch]

        chunked_messages = chunk_list_to_workers(message_batch, self._num_workers)
        chunked_metadata = chunk_list_to_workers(metadata_batch, self._num_workers)

        futures = [
            self._workers[i].execute.remote(
                chunked_messages[i],
                chunked_metadata[i],
                self._judge_weight,
                i * self._stagger_delay_s,
            )
            for i in range(self._num_workers)
        ]
        results = ray.get(futures)

        observations: List[Dict[str, str]] = []
        terminateds_list: List[bool] = []
        rewards_list: List[float] = []
        new_metadata: List[TauBenchEnvMetadata] = []

        for obs, term, rew, meta in results:
            observations.extend(obs)
            terminateds_list.extend(term)
            rewards_list.extend(rew)
            new_metadata.extend(meta)

        next_stop_strings = [["</tool_call>"]] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=new_metadata,
            next_stop_strings=next_stop_strings,
            rewards=torch.tensor(rewards_list, dtype=torch.float32),
            terminateds=torch.tensor(terminateds_list, dtype=torch.bool),
            answers=None,
        )

    def shutdown(self) -> None:
        for worker in self._workers:
            ray.kill(worker)

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> Tuple[BatchedDataDict, Dict[str, Any]]:
        """Compute aggregate metrics over a completed rollout batch."""
        rewards = batch["rewards"]
        if rewards.ndim > 1:
            rewards = rewards[:, 0]
        rewards = rewards * batch["is_end"]

        tau_rewards: List[float] = []
        judge_scores: List[float] = []
        for meta in batch.get("extra_env_info", []):
            if meta and meta.get("tau_reward") is not None:
                tau_rewards.append(float(meta["tau_reward"]))
            if meta and meta.get("judge_score") is not None:
                judge_scores.append(float(meta["judge_score"]))

        metrics: Dict[str, Any] = {
            "tau_bench/task_completion_rate": rewards.mean().item(),
            "tau_bench/pass_at_k": calculate_pass_rate_per_prompt(
                batch["text"], rewards
            ),
            "tau_bench/fraction_properly_ended": batch["is_end"].float().mean().item(),
        }
        if tau_rewards:
            metrics["tau_bench/mean_tau_reward"] = sum(tau_rewards) / len(tau_rewards)
        if judge_scores:
            metrics["tau_bench/mean_judge_score"] = (
                sum(judge_scores) / len(judge_scores)
            )

        return batch, metrics
