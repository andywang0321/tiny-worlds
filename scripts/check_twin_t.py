"""A headless smoke check for the TwinT environment.

Run with ``uv run --extra twin-t python scripts/check_twin_t.py``.
"""

from __future__ import annotations

import numpy as np

from tiny_worlds.twin_t import PINK, TARGET_COLORS, TwinTEnv
from tiny_worlds.twin_t.physics import center_from_pose, tee_polygon


def main() -> int:
    env = TwinTEnv()
    obs, info = env.reset(seed=0, command=PINK)
    assert obs.pixels.shape == (96, 96, 3), obs.pixels.shape
    assert obs.pixels.dtype == np.uint8
    assert obs.agent_pos.shape == (2,)
    assert info.target_poses.shape == (2, 3)
    assert info.coverage.shape == (2,)
    assert env.render().shape == (512, 512, 3)

    centers = np.stack([center_from_pose(pose, env.config.block_scale) for pose in info.target_poses])
    separation = float(np.linalg.norm(centers[0] - centers[1]))
    assert separation >= env.config.target_min_separation, separation
    assert all(tee_polygon(pose, env.config.block_scale).is_valid for pose in info.target_poses)

    for action in (obs.agent_pos, [256.0, 256.0], [350.0, 300.0]):
        obs, reward, _terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
        assert np.isfinite(reward)
        assert np.isfinite(info.coverage).all()
        assert not truncated or info.step >= env.config.episode_limit

    frame = env.render()
    pink_pixels = int(np.all(frame == np.array(TARGET_COLORS[0], dtype=np.uint8), axis=-1).sum())
    green_pixels = int(np.all(frame == np.array(TARGET_COLORS[1], dtype=np.uint8), axis=-1).sum())
    assert pink_pixels > 0 and green_pixels > 0
    env.close()
    print(
        f"ok: targets={info.target_poses[:, :2].round(1).tolist()} "
        f"separation={separation:.1f} coverage={info.coverage.round(3).tolist()} "
        f"pink_px={pink_pixels} green_px={green_pixels}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
