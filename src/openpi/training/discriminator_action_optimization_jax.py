from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class ActionOptimizationResult:
    optimized_parameter: jax.Array
    evaluated_actions: jax.Array
    logits: jax.Array
    scores: jax.Array
    best_iteration: int
    best_optimized_parameter: jax.Array
    best_evaluated_actions: jax.Array
    best_logits: jax.Array
    best_scores: jax.Array
    initial_vector_l2_norms: jax.Array
    history: list[dict[str, float]]


def _ensure_action_batch_array(action_chunk: jax.Array | np.ndarray) -> jax.Array:
    action = jnp.asarray(action_chunk, dtype=jnp.float32)
    if action.ndim == 2:
        return action[jnp.newaxis, ...]
    if action.ndim == 3:
        return action
    raise ValueError(
        "Expected action tensor with shape (T, D) or (N, T, D), "
        f"got shape={tuple(action.shape)}"
    )


def _as_batched_non_gripper_vectors(action_parameter: jax.Array) -> jax.Array:
    action = _ensure_action_batch_array(action_parameter)
    non_gripper = action if action.shape[-1] <= 1 else action[..., :-1]
    return non_gripper


def compute_action_parameter_vector_l2_norms(action_parameter: jax.Array | np.ndarray) -> jax.Array:
    vectors = _as_batched_non_gripper_vectors(jnp.asarray(action_parameter, dtype=jnp.float32))
    norms = jnp.linalg.norm(vectors, ord=2, axis=-1)
    if jnp.asarray(action_parameter).ndim == 2:
        return norms[0]
    return norms


def compute_action_update_rescale_vector(action_stats: dict[str, Any] | None) -> np.ndarray | None:
    if action_stats is None or "q01" not in action_stats or "q99" not in action_stats:
        return None

    low = np.asarray(action_stats["q01"], dtype=np.float32)
    high = np.asarray(action_stats["q99"], dtype=np.float32)
    span = np.maximum(high - low, 1e-6)
    scale = np.ones_like(span, dtype=np.float32)

    non_zero_mask = np.asarray(high > low, dtype=bool)
    if not np.any(non_zero_mask):
        return scale

    reference_span = float(np.median(span[non_zero_mask]))
    if not np.isfinite(reference_span) or reference_span <= 0.0:
        reference_span = 1.0

    scale[non_zero_mask] = reference_span / span[non_zero_mask]
    scale[~non_zero_mask] = 0.0
    return scale.astype(np.float32, copy=False)


def project_action_parameter(
    action_parameter: jax.Array | np.ndarray,
    *,
    normalized_space: bool,
) -> jax.Array:
    action = jnp.asarray(action_parameter, dtype=jnp.float32)
    if not normalized_space:
        return action
    return jnp.clip(action, -1.0, 1.0)


def rescale_action_update(
    action_parameter: jax.Array,
    action_before_step: jax.Array,
    update_rescale_vector: jax.Array | None,
) -> jax.Array:
    if update_rescale_vector is None:
        return action_parameter
    return action_before_step + (action_parameter - action_before_step) * update_rescale_vector


def _smooth_single_action_sequence(action_sequence: jax.Array, radius: int) -> jax.Array:
    if radius <= 0:
        return action_sequence
    padded = jnp.concatenate(
        [
            jnp.repeat(action_sequence[:1], radius, axis=0),
            action_sequence,
            jnp.repeat(action_sequence[-1:], radius, axis=0),
        ],
        axis=0,
    )

    def _window_mean(start_idx: jax.Array) -> jax.Array:
        window = jax.lax.dynamic_slice_in_dim(padded, start_idx, 2 * radius + 1, axis=0)
        return jnp.mean(window, axis=0)

    return jax.vmap(_window_mean)(jnp.arange(action_sequence.shape[0], dtype=jnp.int32))


def apply_action_moving_average(
    action_parameter: jax.Array | np.ndarray,
    radius: int,
) -> jax.Array:
    action = jnp.asarray(action_parameter, dtype=jnp.float32)
    if radius <= 0:
        return action
    if action.ndim == 2:
        return _smooth_single_action_sequence(action, radius)
    if action.ndim == 3:
        return jax.vmap(lambda x: _smooth_single_action_sequence(x, radius))(action)
    raise ValueError(
        "Expected action tensor with shape (T, D) or (N, T, D), "
        f"got shape={tuple(action.shape)}"
    )


def match_action_parameter_vector_l2_norms(
    action_parameter: jax.Array | np.ndarray,
    target_l2_norms: jax.Array | np.ndarray,
    *,
    clip_mode: bool = False,
) -> jax.Array:
    action = jnp.asarray(action_parameter, dtype=jnp.float32)
    targets = jnp.asarray(target_l2_norms, dtype=jnp.float32)
    single = action.ndim == 2
    batched_action = _ensure_action_batch_array(action)
    if targets.ndim == 1:
        targets = targets[jnp.newaxis, :]
    elif targets.ndim != 2:
        raise ValueError(f"Expected `target_l2_norms` with ndim 1 or 2, got shape={tuple(targets.shape)}")

    if batched_action.shape[0] != targets.shape[0]:
        if targets.shape[0] == 1:
            targets = jnp.repeat(targets, batched_action.shape[0], axis=0)
        else:
            raise ValueError(
                "Batched action-parameter vector L2 norm batch mismatch: "
                f"action_shape={tuple(batched_action.shape)}, target_shape={tuple(targets.shape)}"
            )
    if batched_action.shape[1] != targets.shape[1]:
        raise ValueError(
            "Batched action-parameter vector L2 norm shape mismatch: "
            f"action_shape={tuple(batched_action.shape)}, target_shape={tuple(targets.shape)}"
        )

    non_gripper = batched_action if batched_action.shape[-1] <= 1 else batched_action[..., :-1]
    # Stabilize gradients near zero norm to avoid NaNs when this transform is part
    # of the optimization graph.
    sq_norms = jnp.sum(jnp.square(non_gripper), axis=-1)
    raw_l2_norms = jnp.sqrt(sq_norms + 1e-12)
    current_l2_norms = jnp.where(sq_norms > 0.0, raw_l2_norms, 0.0)
    safe_den = jnp.where(current_l2_norms > 0.0, current_l2_norms, 1.0)
    target_is_zero = targets == 0.0
    scales = jnp.where(target_is_zero, 0.0, targets / safe_den)
    scalable_mask = (targets != 0.0) & (current_l2_norms > 0.0)
    if clip_mode:
        scalable_mask = scalable_mask & (current_l2_norms > targets)
    scales = jnp.where(scalable_mask, scales, 1.0)
    scaled_non_gripper = non_gripper * scales[..., jnp.newaxis]
    batched_action = batched_action.at[..., :-1].set(scaled_non_gripper)
    if single:
        return batched_action[0]
    return batched_action


def prepare_action_parameter_for_evaluation(
    action_parameter: jax.Array | np.ndarray,
    target_l2_norms: jax.Array | np.ndarray,
    *,
    normalized_space: bool,
    match_action_l2_norm_every_step: bool,
    match_action_l2_norm_clip_mode: bool,
) -> jax.Array:
    evaluated = jnp.asarray(action_parameter, dtype=jnp.float32)
    if not match_action_l2_norm_every_step:
        evaluated = match_action_parameter_vector_l2_norms(
            evaluated,
            target_l2_norms,
            clip_mode=bool(match_action_l2_norm_clip_mode),
        )
    return project_action_parameter(
        evaluated,
        normalized_space=normalized_space,
    )


def zero_frozen_action_grad(
    action_grad: jax.Array,
    *,
    update_gripper_action: bool,
) -> jax.Array:
    if update_gripper_action:
        return action_grad
    return action_grad.at[..., -1].set(0.0)


def restore_frozen_action_dims(
    action_parameter: jax.Array,
    frozen_reference: jax.Array,
    *,
    update_gripper_action: bool,
) -> jax.Array:
    if update_gripper_action:
        return action_parameter
    return action_parameter.at[..., -1].set(frozen_reference[..., -1])


def compute_pairwise_cosine_similarity_matrix(action_parameter: jax.Array | np.ndarray) -> jax.Array:
    action = jnp.asarray(action_parameter, dtype=jnp.float32)
    if action.ndim == 2:
        flattened = _as_batched_non_gripper_vectors(action).reshape(1, -1)
    elif action.ndim == 3:
        flattened = _as_batched_non_gripper_vectors(action).reshape(action.shape[0], -1)
    else:
        raise ValueError(
            "Expected action tensor with shape (T, D) or (N, T, D), "
            f"got shape={tuple(action.shape)}"
        )
    if flattened.shape[-1] == 0:
        return jnp.zeros((flattened.shape[0], flattened.shape[0]), dtype=jnp.float32)
    normalized = flattened / jnp.maximum(jnp.linalg.norm(flattened, ord=2, axis=-1, keepdims=True), 1e-8)
    return normalized @ normalized.T


def compute_mean_pairwise_cosine_similarity(action_parameter: jax.Array | np.ndarray) -> jax.Array:
    similarity_matrix = compute_pairwise_cosine_similarity_matrix(action_parameter)
    sample_count = int(similarity_matrix.shape[-1])
    if sample_count <= 1:
        return jnp.zeros((), dtype=jnp.float32)
    off_diagonal_mask = ~jnp.eye(sample_count, dtype=bool)
    return jnp.mean(similarity_matrix[off_diagonal_mask])


def _compute_action_l1_l2_norms(action_parameter: jax.Array) -> tuple[float, float]:
    arr = np.asarray(action_parameter, dtype=np.float32)
    return float(np.sum(np.abs(arr))), float(np.sqrt(np.sum(np.square(arr))))


def _compute_thresholded_mean_square(
    values: jax.Array | np.ndarray,
    *,
    clip_value: float,
) -> jax.Array:
    flattened = jnp.asarray(values, dtype=jnp.float32).reshape(-1)
    if float(clip_value) > 0.0:
        flattened = jnp.maximum(flattened - float(clip_value), 0.0)
    return jnp.mean(flattened**2)


def compute_action_l2_norm_penalty(
    action_parameter: jax.Array | np.ndarray,
    *,
    initial_vector_l2_norms: jax.Array | np.ndarray,
    action_l2_norm_penalty_clip_value: float = 0.0,
) -> jax.Array:
    current_vector_l2_norms = compute_action_parameter_vector_l2_norms(action_parameter).reshape(-1)
    initial_vector_l2_norms = jnp.asarray(initial_vector_l2_norms, dtype=jnp.float32).reshape(-1)
    return _compute_thresholded_mean_square(
        jnp.abs(current_vector_l2_norms - initial_vector_l2_norms),
        clip_value=float(action_l2_norm_penalty_clip_value),
    )


def compute_optimization_objective(
    logits: jax.Array,
    *,
    action_parameter: jax.Array,
    initial_vector_l2_norms: jax.Array,
    diversity_similarity_weight: float,
    action_l2_norm_penalty_weight: float,
    action_l2_norm_penalty_clip_value: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    logits = jnp.asarray(logits, dtype=jnp.float32)
    mean_pairwise_cosine_similarity = (
        compute_mean_pairwise_cosine_similarity(action_parameter)
        if float(diversity_similarity_weight) > 0.0
        else jnp.zeros((), dtype=jnp.float32)
    )
    action_l2_norm_penalty = (
        compute_action_l2_norm_penalty(
            action_parameter,
            initial_vector_l2_norms=initial_vector_l2_norms,
            action_l2_norm_penalty_clip_value=float(action_l2_norm_penalty_clip_value),
        )
        if float(action_l2_norm_penalty_weight) > 0.0
        else jnp.zeros((), dtype=jnp.float32)
    )
    optimization_objective = jnp.mean(jax.nn.log_sigmoid(logits))
    optimization_objective -= float(diversity_similarity_weight) * mean_pairwise_cosine_similarity
    optimization_objective -= float(action_l2_norm_penalty_weight) * action_l2_norm_penalty
    return optimization_objective, mean_pairwise_cosine_similarity, action_l2_norm_penalty


def _summarize_history_row(
    *,
    iteration: int,
    action_parameter: jax.Array,
    evaluated_actions: jax.Array,
    logits: jax.Array,
    initial_vector_l2_norms: jax.Array,
    diversity_similarity_weight: float,
    action_l2_norm_penalty_weight: float,
    action_l2_norm_penalty_clip_value: float,
) -> dict[str, float]:
    logits_np = np.asarray(logits, dtype=np.float32).reshape(-1)
    evaluated_scores = jax.nn.sigmoid(logits).astype(jnp.float32)
    optimization_objective, mean_pairwise_cosine_similarity, action_l2_norm_penalty = compute_optimization_objective(
        logits,
        action_parameter=action_parameter,
        initial_vector_l2_norms=initial_vector_l2_norms,
        diversity_similarity_weight=float(diversity_similarity_weight),
        action_l2_norm_penalty_weight=float(action_l2_norm_penalty_weight),
        action_l2_norm_penalty_clip_value=float(action_l2_norm_penalty_clip_value),
    )
    action_l1_norm, action_l2_norm = _compute_action_l1_l2_norms(action_parameter)

    return {
        "iteration": float(iteration),
        "mean_logit": float(np.mean(logits_np)),
        "mean_score": float(np.mean(np.asarray(evaluated_scores, dtype=np.float32))),
        "optimization_objective": float(optimization_objective),
        "action_l2_norm_sq_error": float(action_l2_norm_penalty),
        "action_l2_norm_penalty": float(action_l2_norm_penalty),
        "mean_pairwise_cosine_similarity": float(mean_pairwise_cosine_similarity),
        "action_l1_norm": float(action_l1_norm),
        "action_l2_norm": float(action_l2_norm),
        "evaluated_action_l2_norm": float(_compute_action_l1_l2_norms(evaluated_actions)[1]),
    }


def optimize_action_parameters(
    initial_actions: jax.Array | np.ndarray,
    *,
    score_fn: Callable[[jax.Array], jax.Array],
    num_opt_steps: int,
    lr: float,
    action_stats: dict[str, Any] | None = None,
    action_l2_norm_penalty_weight: float = 0.0,
    action_l2_norm_penalty_clip_value: float = 0.0,
    action_moving_average_radius: int = 0,
    match_action_l2_norm_every_step: bool = False,
    match_action_l2_norm_clip_mode: bool = False,
    update_gripper_action: bool = False,
    diversity_similarity_weight: float = 0.0,
    record_history: bool = True,
    step_callback: Callable[[int, jax.Array, jax.Array, jax.Array, jax.Array], None] | None = None,
) -> ActionOptimizationResult:
    if num_opt_steps < 0:
        raise ValueError(f"`num_opt_steps` must be non-negative, got {num_opt_steps}.")
    if lr <= 0.0:
        raise ValueError(f"`lr` must be positive, got {lr}.")
    if action_stats is None:
        raise ValueError("`optimize_action_parameters` requires `action_stats`.")

    normalized_space = True
    action_parameter = project_action_parameter(
        jnp.asarray(initial_actions, dtype=jnp.float32),
        normalized_space=normalized_space,
    )
    frozen_action_reference = action_parameter
    initial_vector_l2_norms = compute_action_parameter_vector_l2_norms(action_parameter)
    update_rescale_vector_np = compute_action_update_rescale_vector(action_stats)
    if update_rescale_vector_np is None:
        update_rescale_vector = None
    else:
        expand_shape = (1,) * (action_parameter.ndim - 1) + (action_parameter.shape[-1],)
        update_rescale_vector = jnp.asarray(update_rescale_vector_np, dtype=jnp.float32).reshape(expand_shape)

    optimizer = optax.adam(float(lr))
    opt_state = optimizer.init(action_parameter)

    def _prepare_evaluated_actions(current_action_parameter: jax.Array) -> jax.Array:
        return prepare_action_parameter_for_evaluation(
            current_action_parameter,
            initial_vector_l2_norms,
            normalized_space=normalized_space,
            match_action_l2_norm_every_step=bool(match_action_l2_norm_every_step),
            match_action_l2_norm_clip_mode=bool(match_action_l2_norm_clip_mode),
        )

    def _loss_fn(current_action_parameter: jax.Array) -> jax.Array:
        # Keep optimization and evaluation spaces aligned. When L2-norm matching is
        # applied only at evaluation time, optimizing the raw parameter directly can
        # maximize a different objective than the one we report/select on.
        evaluated_actions = _prepare_evaluated_actions(current_action_parameter)
        logits = score_fn(evaluated_actions).astype(jnp.float32)
        discriminator_objective = jnp.mean(jax.nn.log_sigmoid(logits))
        if float(diversity_similarity_weight) > 0.0:
            pairwise_cosine_similarity = compute_mean_pairwise_cosine_similarity(current_action_parameter)
        else:
            pairwise_cosine_similarity = jnp.zeros((), dtype=jnp.float32)
        action_l2_norm_penalty = (
            compute_action_l2_norm_penalty(
                current_action_parameter,
                initial_vector_l2_norms=initial_vector_l2_norms,
                action_l2_norm_penalty_clip_value=float(action_l2_norm_penalty_clip_value),
            )
            if float(action_l2_norm_penalty_weight) > 0.0
            else jnp.zeros((), dtype=jnp.float32)
        )
        optimization_objective = discriminator_objective - (
            float(diversity_similarity_weight) * pairwise_cosine_similarity
        )
        return -optimization_objective + (
            float(action_l2_norm_penalty_weight) * action_l2_norm_penalty
        )

    grad_fn = jax.grad(_loss_fn)

    def _step_fn(
        current_action_parameter: jax.Array,
        current_opt_state: optax.OptState,
    ) -> tuple[jax.Array, optax.OptState]:
        # Keep the outer optimization step in Python. In real pi0/pi05 runs `score_fn`
        # often closes over a large frozen NNX model state, and wrapping the full step in
        # `jax.jit` can force XLA to embed that state as a giant executable constant.
        grad = grad_fn(current_action_parameter)
        grad = zero_frozen_action_grad(
            grad,
            update_gripper_action=bool(update_gripper_action),
        )
        updates, next_opt_state = optimizer.update(grad, current_opt_state, current_action_parameter)
        updated_action_parameter = optax.apply_updates(current_action_parameter, updates)
        updated_action_parameter = rescale_action_update(
            updated_action_parameter,
            current_action_parameter,
            update_rescale_vector,
        )
        updated_action_parameter = apply_action_moving_average(
            updated_action_parameter,
            int(action_moving_average_radius),
        )
        updated_action_parameter = project_action_parameter(
            updated_action_parameter,
            normalized_space=normalized_space,
        )
        updated_action_parameter = restore_frozen_action_dims(
            updated_action_parameter,
            frozen_action_reference,
            update_gripper_action=bool(update_gripper_action),
        )
        if bool(match_action_l2_norm_every_step):
            updated_action_parameter = match_action_parameter_vector_l2_norms(
                updated_action_parameter,
                initial_vector_l2_norms,
                clip_mode=bool(match_action_l2_norm_clip_mode),
            )
            updated_action_parameter = project_action_parameter(
                updated_action_parameter,
                normalized_space=normalized_space,
            )
        return updated_action_parameter, next_opt_state

    history: list[dict[str, float]] = []
    best_iteration = 0
    best_mean_score = -np.inf
    best_optimized_parameter = action_parameter
    best_evaluated_actions = action_parameter
    best_logits = score_fn(action_parameter).astype(jnp.float32)
    best_scores = jax.nn.sigmoid(best_logits).astype(jnp.float32)
    if record_history:
        initial_evaluated_actions = _prepare_evaluated_actions(action_parameter)
        initial_logits = score_fn(initial_evaluated_actions).astype(jnp.float32)
        initial_scores = jax.nn.sigmoid(initial_logits).astype(jnp.float32)
        best_mean_score = float(jnp.mean(initial_scores))
        best_optimized_parameter = action_parameter
        best_evaluated_actions = initial_evaluated_actions
        best_logits = initial_logits
        best_scores = initial_scores
        history.append(
            _summarize_history_row(
                iteration=0,
                action_parameter=action_parameter,
                evaluated_actions=initial_evaluated_actions,
                logits=initial_logits,
                initial_vector_l2_norms=initial_vector_l2_norms,
                diversity_similarity_weight=float(diversity_similarity_weight),
                action_l2_norm_penalty_weight=float(action_l2_norm_penalty_weight),
                action_l2_norm_penalty_clip_value=float(action_l2_norm_penalty_clip_value),
                )
            )
    else:
        initial_evaluated_actions = _prepare_evaluated_actions(action_parameter)
        initial_logits = score_fn(initial_evaluated_actions).astype(jnp.float32)
        initial_scores = jax.nn.sigmoid(initial_logits).astype(jnp.float32)
        best_mean_score = float(jnp.mean(initial_scores))
        best_evaluated_actions = initial_evaluated_actions
        best_logits = initial_logits
        best_scores = initial_scores

    for iteration in range(int(num_opt_steps)):
        action_parameter, opt_state = _step_fn(action_parameter, opt_state)
        evaluated_actions = _prepare_evaluated_actions(action_parameter)
        logits = score_fn(evaluated_actions).astype(jnp.float32)
        scores = jax.nn.sigmoid(logits).astype(jnp.float32)
        current_mean_score = float(jnp.mean(scores))
        if current_mean_score > best_mean_score:
            best_iteration = iteration + 1
            best_mean_score = current_mean_score
            best_optimized_parameter = action_parameter
            best_evaluated_actions = evaluated_actions
            best_logits = logits
            best_scores = scores
        if record_history:
            history.append(
                _summarize_history_row(
                    iteration=iteration + 1,
                    action_parameter=action_parameter,
                    evaluated_actions=evaluated_actions,
                    logits=logits,
                    initial_vector_l2_norms=initial_vector_l2_norms,
                    diversity_similarity_weight=float(diversity_similarity_weight),
                    action_l2_norm_penalty_weight=float(action_l2_norm_penalty_weight),
                    action_l2_norm_penalty_clip_value=float(action_l2_norm_penalty_clip_value),
                )
            )
        if step_callback is not None:
            step_callback(iteration + 1, action_parameter, evaluated_actions, logits, scores)

    evaluated_actions = _prepare_evaluated_actions(action_parameter)
    logits = score_fn(evaluated_actions).astype(jnp.float32)
    scores = jax.nn.sigmoid(logits).astype(jnp.float32)
    return ActionOptimizationResult(
        optimized_parameter=action_parameter,
        evaluated_actions=evaluated_actions,
        logits=logits,
        scores=scores,
        best_iteration=int(best_iteration),
        best_optimized_parameter=best_optimized_parameter,
        best_evaluated_actions=best_evaluated_actions,
        best_logits=best_logits,
        best_scores=best_scores,
        initial_vector_l2_norms=initial_vector_l2_norms,
        history=history,
    )


def _ensure_vector_batch_array(vector_parameter: jax.Array | np.ndarray) -> jax.Array:
    vector = jnp.asarray(vector_parameter, dtype=jnp.float32)
    if vector.ndim == 1:
        return vector[jnp.newaxis, ...]
    if vector.ndim == 2:
        return vector
    raise ValueError(
        "Expected vector tensor with shape (D,) or (N, D), "
        f"got shape={tuple(vector.shape)}"
    )


def compute_vector_l2_norms(vector_parameter: jax.Array | np.ndarray) -> jax.Array:
    vectors = _ensure_vector_batch_array(vector_parameter)
    norms = jnp.linalg.norm(vectors, ord=2, axis=-1)
    if jnp.asarray(vector_parameter).ndim == 1:
        return norms[0]
    return norms


def compute_vector_reconstruction_penalty(
    vector_parameter: jax.Array | np.ndarray,
    *,
    reconstruct_vector_fn: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    vectors = _ensure_vector_batch_array(vector_parameter)
    reconstructed = _ensure_vector_batch_array(reconstruct_vector_fn(vectors))
    if reconstructed.shape != vectors.shape:
        raise ValueError(
            "Reconstructed vectors must match the optimized latent shape. "
            f"vector_shape={tuple(vectors.shape)}, reconstructed_shape={tuple(reconstructed.shape)}"
    )
    return jnp.mean(jnp.square(vectors - reconstructed))


def compute_vector_knn_distance_penalty(
    vector_parameter: jax.Array | np.ndarray,
    *,
    reference_vectors: jax.Array | np.ndarray,
    k: int,
) -> jax.Array:
    vectors = _ensure_vector_batch_array(vector_parameter)
    references = _ensure_vector_batch_array(reference_vectors)
    if references.shape[0] == 0:
        raise ValueError("`reference_vectors` must be non-empty.")
    if vectors.shape[-1] != references.shape[-1]:
        raise ValueError(
            "vector/reference latent dim mismatch. "
            f"vector_dim={vectors.shape[-1]}, reference_dim={references.shape[-1]}"
        )
    if int(k) <= 0:
        raise ValueError(f"`k` must be >= 1, got {k}.")

    k_eff = min(int(k), int(references.shape[0]))
    vector_sq_norms = jnp.sum(jnp.square(vectors), axis=-1, keepdims=True)
    reference_sq_norms = jnp.sum(jnp.square(references), axis=-1, keepdims=True).T
    sq_distances = vector_sq_norms + reference_sq_norms - 2.0 * (vectors @ references.T)
    sq_distances = jnp.maximum(sq_distances, 0.0)
    if k_eff == int(references.shape[0]):
        nearest_sq_distances = sq_distances
    else:
        nearest_sq_distances = -jax.lax.top_k(-sq_distances, k_eff)[0]
    nearest_distances = jnp.sqrt(jnp.maximum(nearest_sq_distances, 1e-12))
    return jnp.mean(nearest_distances)


def match_vector_l2_norms(
    vector_parameter: jax.Array | np.ndarray,
    target_l2_norms: jax.Array | np.ndarray,
    *,
    clip_mode: bool = False,
) -> jax.Array:
    vector = jnp.asarray(vector_parameter, dtype=jnp.float32)
    targets = jnp.asarray(target_l2_norms, dtype=jnp.float32)
    single = vector.ndim == 1
    batched_vector = _ensure_vector_batch_array(vector)
    if targets.ndim == 0:
        targets = targets[jnp.newaxis]
    elif targets.ndim != 1:
        raise ValueError(f"Expected `target_l2_norms` with ndim 0 or 1, got shape={tuple(targets.shape)}")

    if batched_vector.shape[0] != targets.shape[0]:
        if targets.shape[0] == 1:
            targets = jnp.repeat(targets, batched_vector.shape[0], axis=0)
        else:
            raise ValueError(
                "Batched vector L2 norm batch mismatch: "
                f"vector_shape={tuple(batched_vector.shape)}, target_shape={tuple(targets.shape)}"
            )

    # Stabilize gradients near zero norm to avoid NaNs when this transform is part
    # of the optimization graph.
    sq_norms = jnp.sum(jnp.square(batched_vector), axis=-1)
    raw_l2_norms = jnp.sqrt(sq_norms + 1e-12)
    current_l2_norms = jnp.where(sq_norms > 0.0, raw_l2_norms, 0.0)
    safe_den = jnp.where(current_l2_norms > 0.0, current_l2_norms, 1.0)
    target_is_zero = targets == 0.0
    scales = jnp.where(target_is_zero, 0.0, targets / safe_den)
    scalable_mask = (targets != 0.0) & (current_l2_norms > 0.0)
    if clip_mode:
        scalable_mask = scalable_mask & (current_l2_norms > targets)
    scales = jnp.where(scalable_mask, scales, 1.0)
    scaled_vectors = batched_vector * scales[..., jnp.newaxis]
    if single:
        return scaled_vectors[0]
    return scaled_vectors


def prepare_vector_parameter_for_evaluation(
    vector_parameter: jax.Array | np.ndarray,
    target_l2_norms: jax.Array | np.ndarray,
    *,
    match_action_l2_norm_every_step: bool,
    match_action_l2_norm_clip_mode: bool,
) -> jax.Array:
    evaluated = jnp.asarray(vector_parameter, dtype=jnp.float32)
    if match_action_l2_norm_every_step:
        evaluated = match_vector_l2_norms(
            evaluated,
            target_l2_norms,
            clip_mode=bool(match_action_l2_norm_clip_mode),
        )
    return evaluated


def compute_pairwise_cosine_similarity_matrix_for_vectors(vector_parameter: jax.Array | np.ndarray) -> jax.Array:
    vectors = _ensure_vector_batch_array(vector_parameter)
    if vectors.shape[-1] == 0:
        return jnp.zeros((vectors.shape[0], vectors.shape[0]), dtype=jnp.float32)
    normalized = vectors / jnp.maximum(jnp.linalg.norm(vectors, ord=2, axis=-1, keepdims=True), 1e-8)
    return normalized @ normalized.T


def compute_mean_pairwise_cosine_similarity_for_vectors(vector_parameter: jax.Array | np.ndarray) -> jax.Array:
    similarity_matrix = compute_pairwise_cosine_similarity_matrix_for_vectors(vector_parameter)
    sample_count = int(similarity_matrix.shape[-1])
    if sample_count <= 1:
        return jnp.zeros((), dtype=jnp.float32)
    off_diagonal_mask = ~jnp.eye(sample_count, dtype=bool)
    return jnp.mean(similarity_matrix[off_diagonal_mask])


def compute_vector_optimization_objective(
    logits: jax.Array,
    *,
    vector_parameter: jax.Array,
    initial_vector_l2_norms: jax.Array,
    diversity_similarity_weight: float,
    vector_l2_norm_penalty_weight: float,
    vector_l2_norm_penalty_clip_value: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    logits = jnp.asarray(logits, dtype=jnp.float32)
    del initial_vector_l2_norms
    mean_pairwise_cosine_similarity = (
        compute_mean_pairwise_cosine_similarity_for_vectors(vector_parameter)
        if float(diversity_similarity_weight) > 0.0
        else jnp.zeros((), dtype=jnp.float32)
    )
    vector_l2_norm_penalty = (
        _compute_thresholded_mean_square(
            compute_vector_l2_norms(vector_parameter).reshape(-1),
            clip_value=float(vector_l2_norm_penalty_clip_value),
        )
        if float(vector_l2_norm_penalty_weight) > 0.0
        else jnp.zeros((), dtype=jnp.float32)
    )
    optimization_objective = jnp.mean(jax.nn.log_sigmoid(logits))
    optimization_objective -= float(diversity_similarity_weight) * mean_pairwise_cosine_similarity
    optimization_objective -= float(vector_l2_norm_penalty_weight) * vector_l2_norm_penalty
    return optimization_objective, mean_pairwise_cosine_similarity, vector_l2_norm_penalty


def _summarize_vector_history_row(
    *,
    iteration: int,
    vector_parameter: jax.Array,
    evaluated_vectors: jax.Array,
    logits: jax.Array,
    initial_vector_l2_norms: jax.Array,
    diversity_similarity_weight: float,
    vector_l2_norm_penalty_weight: float,
    vector_l2_norm_penalty_clip_value: float,
    vector_reconstruction_penalty_weight: float,
    vector_reconstruction_consistency_fn: Callable[[jax.Array], jax.Array] | None,
    vector_knn_distance_penalty_weight: float,
    vector_knn_reference_vectors: jax.Array | None,
    vector_knn_k: int,
) -> dict[str, float]:
    logits_np = np.asarray(logits, dtype=np.float32).reshape(-1)
    evaluated_scores = jax.nn.sigmoid(logits).astype(jnp.float32)
    optimization_objective, mean_pairwise_cosine_similarity, vector_l2_norm_penalty = compute_vector_optimization_objective(
        logits,
        vector_parameter=vector_parameter,
        initial_vector_l2_norms=initial_vector_l2_norms,
        diversity_similarity_weight=float(diversity_similarity_weight),
        vector_l2_norm_penalty_weight=float(vector_l2_norm_penalty_weight),
        vector_l2_norm_penalty_clip_value=float(vector_l2_norm_penalty_clip_value),
    )
    vector_reconstruction_penalty = (
        compute_vector_reconstruction_penalty(
            vector_parameter,
            reconstruct_vector_fn=vector_reconstruction_consistency_fn,
        )
        if float(vector_reconstruction_penalty_weight) > 0.0 and vector_reconstruction_consistency_fn is not None
        else jnp.zeros((), dtype=jnp.float32)
    )
    vector_knn_distance_penalty = (
        compute_vector_knn_distance_penalty(
            vector_parameter,
            reference_vectors=vector_knn_reference_vectors,
            k=int(vector_knn_k),
        )
        if float(vector_knn_distance_penalty_weight) > 0.0
        and vector_knn_reference_vectors is not None
        and int(vector_knn_k) > 0
        else jnp.zeros((), dtype=jnp.float32)
    )
    optimization_objective -= float(vector_reconstruction_penalty_weight) * vector_reconstruction_penalty
    optimization_objective -= float(vector_knn_distance_penalty_weight) * vector_knn_distance_penalty
    vector_l1_norm, vector_l2_norm = _compute_action_l1_l2_norms(vector_parameter)

    return {
        "iteration": float(iteration),
        "mean_logit": float(np.mean(logits_np)),
        "mean_score": float(np.mean(np.asarray(evaluated_scores, dtype=np.float32))),
        "optimization_objective": float(optimization_objective),
        "action_l2_norm_sq_error": float(vector_l2_norm_penalty),
        "action_l2_norm_penalty": float(vector_l2_norm_penalty),
        "vector_reconstruction_penalty": float(vector_reconstruction_penalty),
        "vector_knn_distance_penalty": float(vector_knn_distance_penalty),
        "mean_pairwise_cosine_similarity": float(mean_pairwise_cosine_similarity),
        "action_l1_norm": float(vector_l1_norm),
        "action_l2_norm": float(vector_l2_norm),
        "evaluated_action_l2_norm": float(_compute_action_l1_l2_norms(evaluated_vectors)[1]),
    }


def optimize_vector_parameters(
    initial_vectors: jax.Array | np.ndarray,
    *,
    score_fn: Callable[[jax.Array], jax.Array],
    num_opt_steps: int,
    lr: float,
    vector_l2_norm_penalty_weight: float = 0.0,
    vector_l2_norm_penalty_clip_value: float = 0.0,
    vector_reconstruction_penalty_weight: float = 0.0,
    vector_reconstruction_consistency_fn: Callable[[jax.Array], jax.Array] | None = None,
    vector_knn_distance_penalty_weight: float = 0.0,
    vector_knn_reference_vectors: jax.Array | np.ndarray | None = None,
    vector_knn_k: int = 0,
    match_action_l2_norm_every_step: bool = False,
    match_action_l2_norm_clip_mode: bool = False,
    diversity_similarity_weight: float = 0.0,
    record_history: bool = True,
    step_callback: Callable[[int, jax.Array, jax.Array, jax.Array, jax.Array], None] | None = None,
) -> ActionOptimizationResult:
    if num_opt_steps < 0:
        raise ValueError(f"`num_opt_steps` must be non-negative, got {num_opt_steps}.")
    if lr <= 0.0:
        raise ValueError(f"`lr` must be positive, got {lr}.")
    if vector_reconstruction_penalty_weight < 0.0:
        raise ValueError(
            "`vector_reconstruction_penalty_weight` must be non-negative, "
            f"got {vector_reconstruction_penalty_weight}."
        )
    if vector_reconstruction_penalty_weight > 0.0 and vector_reconstruction_consistency_fn is None:
        raise ValueError(
            "`vector_reconstruction_consistency_fn` is required when "
            "`vector_reconstruction_penalty_weight > 0`."
        )
    if vector_knn_distance_penalty_weight < 0.0:
        raise ValueError(
            "`vector_knn_distance_penalty_weight` must be non-negative, "
            f"got {vector_knn_distance_penalty_weight}."
        )
    if int(vector_knn_k) < 0:
        raise ValueError(f"`vector_knn_k` must be >= 0, got {vector_knn_k}.")
    if vector_knn_distance_penalty_weight > 0.0:
        if vector_knn_reference_vectors is None:
            raise ValueError(
                "`vector_knn_reference_vectors` is required when "
                "`vector_knn_distance_penalty_weight > 0`."
            )
        if int(vector_knn_k) <= 0:
            raise ValueError(
                "`vector_knn_k` must be >= 1 when `vector_knn_distance_penalty_weight > 0`."
            )

    vector_parameter = jnp.asarray(initial_vectors, dtype=jnp.float32)
    vector_knn_reference_vectors_jax = (
        None
        if vector_knn_reference_vectors is None
        else _ensure_vector_batch_array(jnp.asarray(vector_knn_reference_vectors, dtype=jnp.float32))
    )
    if (
        vector_knn_reference_vectors_jax is not None
        and vector_knn_reference_vectors_jax.shape[-1] != vector_parameter.shape[-1]
    ):
        raise ValueError(
            "Initial vector dim does not match vector_knn_reference_vectors dim. "
            f"initial_dim={vector_parameter.shape[-1]}, reference_dim={vector_knn_reference_vectors_jax.shape[-1]}"
        )
    initial_vector_l2_norms = compute_vector_l2_norms(vector_parameter)

    optimizer = optax.adam(float(lr))
    opt_state = optimizer.init(vector_parameter)

    def _prepare_evaluated_vectors(current_vector_parameter: jax.Array) -> jax.Array:
        return prepare_vector_parameter_for_evaluation(
            current_vector_parameter,
            initial_vector_l2_norms,
            match_action_l2_norm_every_step=bool(match_action_l2_norm_every_step),
            match_action_l2_norm_clip_mode=bool(match_action_l2_norm_clip_mode),
        )

    def _loss_fn(current_vector_parameter: jax.Array) -> jax.Array:
        evaluated_vectors = _prepare_evaluated_vectors(current_vector_parameter)
        logits = score_fn(evaluated_vectors).astype(jnp.float32)
        discriminator_objective = jnp.mean(jax.nn.log_sigmoid(logits))
        if float(diversity_similarity_weight) > 0.0:
            pairwise_cosine_similarity = compute_mean_pairwise_cosine_similarity_for_vectors(current_vector_parameter)
        else:
            pairwise_cosine_similarity = jnp.zeros((), dtype=jnp.float32)
        vector_l2_norm_penalty = (
            _compute_thresholded_mean_square(
                compute_vector_l2_norms(current_vector_parameter).reshape(-1),
                clip_value=float(vector_l2_norm_penalty_clip_value),
            )
            if float(vector_l2_norm_penalty_weight) > 0.0
            else jnp.zeros((), dtype=jnp.float32)
        )
        vector_reconstruction_penalty = (
            compute_vector_reconstruction_penalty(
                current_vector_parameter,
                reconstruct_vector_fn=vector_reconstruction_consistency_fn,
            )
            if float(vector_reconstruction_penalty_weight) > 0.0 and vector_reconstruction_consistency_fn is not None
            else jnp.zeros((), dtype=jnp.float32)
        )
        vector_knn_distance_penalty = (
            compute_vector_knn_distance_penalty(
                current_vector_parameter,
                reference_vectors=vector_knn_reference_vectors_jax,
                k=int(vector_knn_k),
            )
            if float(vector_knn_distance_penalty_weight) > 0.0
            and vector_knn_reference_vectors_jax is not None
            and int(vector_knn_k) > 0
            else jnp.zeros((), dtype=jnp.float32)
        )
        optimization_objective = discriminator_objective - (
            float(diversity_similarity_weight) * pairwise_cosine_similarity
        )
        return (
            -optimization_objective
            + float(vector_l2_norm_penalty_weight) * vector_l2_norm_penalty
            + float(vector_reconstruction_penalty_weight) * vector_reconstruction_penalty
            + float(vector_knn_distance_penalty_weight) * vector_knn_distance_penalty
        )

    grad_fn = jax.grad(_loss_fn)

    def _step_fn(
        current_vector_parameter: jax.Array,
        current_opt_state: optax.OptState,
    ) -> tuple[jax.Array, optax.OptState]:
        # Keep the outer optimization step in Python. In latent optimization the score
        # function and reconstruction penalty can close over full discriminator/VAE
        # states; jitting the whole step makes XLA materialize them as large constants.
        grad = grad_fn(current_vector_parameter)
        updates, next_opt_state = optimizer.update(grad, current_opt_state, current_vector_parameter)
        updated_vector_parameter = optax.apply_updates(current_vector_parameter, updates)
        if bool(match_action_l2_norm_every_step):
            updated_vector_parameter = match_vector_l2_norms(
                updated_vector_parameter,
                initial_vector_l2_norms,
                clip_mode=bool(match_action_l2_norm_clip_mode),
            )
        return updated_vector_parameter, next_opt_state

    history: list[dict[str, float]] = []
    best_iteration = 0
    best_mean_score = -np.inf
    best_optimized_parameter = vector_parameter
    initial_evaluated_vectors = _prepare_evaluated_vectors(vector_parameter)
    initial_logits = score_fn(initial_evaluated_vectors).astype(jnp.float32)
    initial_scores = jax.nn.sigmoid(initial_logits).astype(jnp.float32)
    best_evaluated_vectors = initial_evaluated_vectors
    best_logits = initial_logits
    best_scores = initial_scores
    best_mean_score = float(jnp.mean(initial_scores))
    if record_history:
        history.append(
            _summarize_vector_history_row(
                iteration=0,
                vector_parameter=vector_parameter,
                evaluated_vectors=initial_evaluated_vectors,
                logits=initial_logits,
                initial_vector_l2_norms=initial_vector_l2_norms,
                diversity_similarity_weight=float(diversity_similarity_weight),
                vector_l2_norm_penalty_weight=float(vector_l2_norm_penalty_weight),
                vector_l2_norm_penalty_clip_value=float(vector_l2_norm_penalty_clip_value),
                vector_reconstruction_penalty_weight=float(vector_reconstruction_penalty_weight),
                vector_reconstruction_consistency_fn=vector_reconstruction_consistency_fn,
                vector_knn_distance_penalty_weight=float(vector_knn_distance_penalty_weight),
                vector_knn_reference_vectors=vector_knn_reference_vectors_jax,
                vector_knn_k=int(vector_knn_k),
            )
        )

    for iteration in range(int(num_opt_steps)):
        vector_parameter, opt_state = _step_fn(vector_parameter, opt_state)
        evaluated_vectors = _prepare_evaluated_vectors(vector_parameter)
        logits = score_fn(evaluated_vectors).astype(jnp.float32)
        scores = jax.nn.sigmoid(logits).astype(jnp.float32)
        current_mean_score = float(jnp.mean(scores))
        if current_mean_score > best_mean_score:
            best_iteration = iteration + 1
            best_mean_score = current_mean_score
            best_optimized_parameter = vector_parameter
            best_evaluated_vectors = evaluated_vectors
            best_logits = logits
            best_scores = scores
        if record_history:
            history.append(
                _summarize_vector_history_row(
                    iteration=iteration + 1,
                    vector_parameter=vector_parameter,
                    evaluated_vectors=evaluated_vectors,
                    logits=logits,
                    initial_vector_l2_norms=initial_vector_l2_norms,
                    diversity_similarity_weight=float(diversity_similarity_weight),
                    vector_l2_norm_penalty_weight=float(vector_l2_norm_penalty_weight),
                    vector_l2_norm_penalty_clip_value=float(vector_l2_norm_penalty_clip_value),
                    vector_reconstruction_penalty_weight=float(vector_reconstruction_penalty_weight),
                    vector_reconstruction_consistency_fn=vector_reconstruction_consistency_fn,
                    vector_knn_distance_penalty_weight=float(vector_knn_distance_penalty_weight),
                    vector_knn_reference_vectors=vector_knn_reference_vectors_jax,
                    vector_knn_k=int(vector_knn_k),
                )
            )
        if step_callback is not None:
            step_callback(iteration + 1, vector_parameter, evaluated_vectors, logits, scores)

    evaluated_vectors = _prepare_evaluated_vectors(vector_parameter)
    logits = score_fn(evaluated_vectors).astype(jnp.float32)
    scores = jax.nn.sigmoid(logits).astype(jnp.float32)
    return ActionOptimizationResult(
        optimized_parameter=vector_parameter,
        evaluated_actions=evaluated_vectors,
        logits=logits,
        scores=scores,
        best_iteration=int(best_iteration),
        best_optimized_parameter=best_optimized_parameter,
        best_evaluated_actions=best_evaluated_vectors,
        best_logits=best_logits,
        best_scores=best_scores,
        initial_vector_l2_norms=initial_vector_l2_norms,
        history=history,
    )
