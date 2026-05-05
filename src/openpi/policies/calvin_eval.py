from __future__ import annotations

import collections
import concurrent.futures
import multiprocessing
import random
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi.policies import calvin_actions as _calvin_actions


def extract_goal_text(goal) -> str:
    if isinstance(goal, str):
        return goal
    if isinstance(goal, bytes):
        return goal.decode("utf-8")
    if isinstance(goal, Mapping):
        for key in ("lang_text", "language", "goal", "text"):
            value = goal.get(key)
            if value is not None:
                return extract_goal_text(value)
    if isinstance(goal, (list, tuple)) and goal:
        return extract_goal_text(goal[0])
    return str(goal)


def build_calvin_policy_input(obs: Mapping[str, object], goal) -> Dict[str, object]:
    rgb_obs = obs["rgb_obs"]
    if not isinstance(rgb_obs, Mapping):
        raise TypeError(f"Expected obs['rgb_obs'] to be a mapping, got {type(rgb_obs)}.")

    base_image = rgb_obs["rgb_static"]
    wrist_image = rgb_obs["rgb_gripper"]
    state = obs["robot_obs"]

    return {
        "observation/image": np.asarray(base_image),
        "observation/wrist_image": np.asarray(wrist_image),
        "observation/state": np.asarray(state, dtype=np.float32),
        "prompt": extract_goal_text(goal),
    }


def compute_chain_success(results: List[int]) -> Dict[int, float]:
    if not results:
        return {idx: 0.0 for idx in range(1, 6)}
    total = float(len(results))
    return {idx: sum(result >= idx for result in results) / total for idx in range(1, 6)}


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        from pytorch_lightning import seed_everything as _pl_seed_everything

        _pl_seed_everything(seed, workers=True)
    except ImportError:
        pass


def _sample_valid_sequences_for_state(
    *,
    state: Mapping[str, Any],
    num_sequences: int,
    seed: int,
    task_names: list[str],
    check_sequence,
) -> list[tuple[str, ...]]:
    if num_sequences <= 0:
        return []

    rng = np.random.RandomState(seed)
    seq_len = 5
    results: list[tuple[str, ...]] = []
    while len(results) < num_sequences:
        seq = tuple(rng.choice(task_names, size=seq_len, replace=False).tolist())
        if check_sequence(state, seq):
            results.append(seq)
    return results


def _sample_valid_sequences_for_state_parallel(args) -> list[tuple[str, ...]]:
    state, num_sequences, seed = args
    from calvin_agent.evaluation import multistep_sequences as _multistep_sequences

    return _sample_valid_sequences_for_state(
        state=state,
        num_sequences=num_sequences,
        seed=seed,
        task_names=list(_multistep_sequences.tasks.keys()),
        check_sequence=_multistep_sequences.check_sequence,
    )


def make_seeded_get_sequences(seed: int, *, sequence_module=None):
    if sequence_module is None:
        from calvin_agent.evaluation import multistep_sequences as sequence_module

    def _get_sequences(num_sequences=1000, num_workers=None):
        possible_conditions = {
            "led": [0, 1],
            "lightbulb": [0, 1],
            "slider": ["right", "left"],
            "drawer": ["closed", "open"],
            "red_block": ["table", "slider_right", "slider_left"],
            "blue_block": ["table", "slider_right", "slider_left"],
            "pink_block": ["table", "slider_right", "slider_left"],
            "grasped": [0],
        }

        def _is_valid_initial_condition(values):
            return values.count("table") in [1, 2] and values.count("slider_right") < 2 and values.count("slider_left") < 2

        value_combinations = filter(_is_valid_initial_condition, sequence_module.product(*possible_conditions.values()))
        initial_states = [dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations]
        num_sequences_per_state = list(map(len, np.array_split(range(num_sequences), len(initial_states))))
        sequence_module.logger.info("Start generating evaluation sequences.")

        worker_args = [
            (state, count, int(seed) + idx) for idx, (state, count) in enumerate(zip(initial_states, num_sequences_per_state))
        ]
        task_names = list(sequence_module.tasks.keys())
        if num_workers in (0, 1):
            sequences_per_state = [
                _sample_valid_sequences_for_state(
                    state=state,
                    num_sequences=count,
                    seed=worker_seed,
                    task_names=task_names,
                    check_sequence=sequence_module.check_sequence,
                )
                for state, count, worker_seed in worker_args
            ]
        else:
            max_workers = multiprocessing.cpu_count() if num_workers is None else num_workers
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                sequences_per_state = list(executor.map(_sample_valid_sequences_for_state_parallel, worker_args))

        results = [
            (state, sequence)
            for state, state_sequences in zip(initial_states, sequences_per_state)
            for sequence in state_sequences
        ]
        shuffle_rng = np.random.RandomState(int(seed))
        shuffle_rng.shuffle(results)
        sequence_module.logger.info("Done generating evaluation sequences.")
        return results

    return _get_sequences


class CalvinOpenPIModel:
    """Adapter from CALVIN's reset/step interface to the OpenPI websocket server."""

    def __init__(self, *, host: str, port: int, replan_steps: int = 5):
        self._client = _websocket_client_policy.WebsocketClientPolicy(host, port)
        self._replan_steps = int(replan_steps)
        if self._replan_steps <= 0:
            raise ValueError(f"`replan_steps` must be >= 1, got {replan_steps}.")
        self._action_plan: collections.deque[np.ndarray] = collections.deque()

    def reset(self):
        self._action_plan.clear()

    def step(self, obs, goal):
        if not self._action_plan:
            result = self._client.infer(build_calvin_policy_input(obs, goal))
            action_chunk = _calvin_actions.to_calvin_action_chunk(result["actions"])
            if action_chunk.ndim != 2:
                raise ValueError(
                    f"Expected policy to return an action chunk with shape [T, D], got {action_chunk.shape}."
                )
            for action in action_chunk[: self._replan_steps]:
                self._action_plan.append(np.asarray(action, dtype=np.float32))
        return self._action_plan.popleft()
