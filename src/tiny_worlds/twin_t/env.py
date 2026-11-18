"""The TwinT environment: two colored T targets, one T block, one round agent."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_CONFIG, TwinTConfig
from .physics import PhysicsWorld, coverage, keypoints
from .render import Scene, TwinTRenderer
from .sampling import parse_state, parse_target_poses, sample_layout, sample_targets_avoiding
from .types import Command, FloatArray, StepInfo, TwinTObservation, TwinTState


class TwinTEnv:
    """A PushT-style world with a commanded pink or green target."""

    def __init__(self, config: TwinTConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.np_random = np.random.default_rng()
        self._renderer = TwinTRenderer(config)
        self._world = PhysicsWorld(config)
        self._target_poses = np.zeros((2, 3), dtype=np.float32)
        self._command = Command(None)
        self._step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        command: int | None = None,
        state: NDArray[np.floating] | None = None,
        target_poses: NDArray[np.floating] | None = None,
    ) -> tuple[TwinTObservation, StepInfo]:
        """Reset the episode; explicit ``state`` and ``target_poses`` are evaluation-only."""
        self.np_random = np.random.default_rng(seed)
        self._command = Command(command)
        agent_pos, block_pose, targets = self._initial_layout(state, target_poses)
        self._target_poses = targets
        self._world = PhysicsWorld(self.config)
        self._world.set_bodies(agent_pos, block_pose)
        self._step = 0
        return self._observation(), self._info()

    def _initial_layout(
        self,
        state: NDArray[np.floating] | None,
        target_poses: NDArray[np.floating] | None,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        if state is None:
            layout = sample_layout(self.np_random, self.config, target_poses)
            return layout.agent_pos, layout.block_pose, layout.target_poses
        agent_pos, block_pose = parse_state(state, self.config)
        if target_poses is None:
            targets = sample_targets_avoiding(self.np_random, self.config, agent_pos, block_pose)
        else:
            targets = parse_target_poses(target_poses, self.config)
        return agent_pos, block_pose, targets

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[TwinTObservation, float, bool, bool, StepInfo]:
        """Run one control step with the agent tracking ``action``."""
        target = _clip_action(action, float(self.config.workspace))
        self._world.step_control(target, self.config.substeps, self.config.max_agent_speed)
        self._step += 1
        info = self._info()
        terminated = self._terminated(info)
        truncated = self._step >= self.config.episode_limit and not terminated
        return self._observation(), self._reward(info.coverage), terminated, truncated, info

    def set_command(self, command: int | None) -> None:
        """Change the commanded target without resetting the episode."""
        self._command = Command(command)

    def pause(self) -> None:
        """Zero the agent velocity; safe to call when idle."""
        self._world.pause()

    def state(self) -> TwinTState:
        """Return the current simulator state."""
        return TwinTState(
            agent_pos=self._world.agent_pos,
            block_pose=self._world.block_pose,
            target_poses=self.target_poses,
            command=self._command,
            step=self._step,
        )

    @property
    def target_poses(self) -> FloatArray:
        """The ``(2, 3)`` target body poses, ``[pink, green]``."""
        return self._target_poses.copy()

    def render(self, size: int | None = None) -> NDArray[np.uint8]:
        """Return an RGB frame, ``(size, size, 3)`` uint8."""
        frame_size = self.config.render_size if size is None else int(size)
        return self._renderer.render(frame_size, self._scene())

    def close(self) -> None:
        """Release rendering resources; idempotent."""
        self._renderer.close()

    def _scene(self) -> Scene:
        return Scene(
            agent_pos=self._world.agent_pos,
            block_pose=self._world.block_pose,
            target_poses=self._target_poses,
        )

    def _observation(self) -> TwinTObservation:
        agent_pos = self._world.agent_pos
        scene = Scene(agent_pos=agent_pos, block_pose=self._world.block_pose, target_poses=self._target_poses)
        pixels = self._renderer.render(self.config.observation_size, scene)
        return TwinTObservation(pixels=pixels, agent_pos=agent_pos)

    def _info(self) -> StepInfo:
        block_pose = self._world.block_pose
        cover = coverage(block_pose, self._target_poses, self.config.block_scale)
        task_success = bool(np.any(cover >= self.config.success_threshold))
        target = self._command.target
        command_success = target is not None and bool(cover[target] >= self.config.success_threshold)
        return StepInfo(
            block_pose=block_pose,
            block_keypoints=keypoints(block_pose, self.config.block_scale),
            target_poses=self.target_poses,
            coverage=cover,
            command_success=command_success,
            task_success=task_success,
            achieved_target=int(np.argmax(cover)) if task_success else None,
            n_contacts=self._world.n_contacts(self.config.substeps),
            step=self._step,
        )

    def _reward(self, cover: FloatArray) -> float:
        target = self._command.target
        selected = float(np.max(cover)) if target is None else float(cover[target])
        return float(np.clip(selected / self.config.success_threshold, 0.0, 1.0))

    def _terminated(self, info: StepInfo) -> bool:
        if self.config.terminate_on == "command" and not self._command.is_null:
            return info.command_success
        return info.task_success


def _clip_action(action: NDArray[np.floating], limit: float) -> FloatArray:
    try:
        array = np.asarray(action, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("action must be numeric") from error
    if array.shape != (2,):
        raise ValueError(f"action must have shape (2,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("action must be finite")
    return np.clip(array, 0.0, limit).astype(np.float32)
