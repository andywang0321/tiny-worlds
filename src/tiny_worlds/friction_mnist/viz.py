"""Interactive visualization for friction Moving MNIST."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .dataset import FrictionMNIST

_COLORS = ("tab:red", "tab:cyan")


def _footer(dataset: FrictionMNIST, index: int, frame: int, digit: int) -> str:
    x, y = dataset.position[index, frame, digit]
    vx, vy = dataset.velocity[index, frame, digit]
    return (
        f"d{digit} label={int(dataset.digit_label[index, digit])} "
        f"x={x:5.1f} y={y:5.1f} vx={vx:+5.2f} vy={vy:+5.2f} "
        f"friction={dataset.friction[index, digit]:.3f} "
        f"restitution={dataset.restitution[index, digit]:.3f}"
    )


def visualize_sample(
    dataset: FrictionMNIST,
    index: int,
    *,
    fps: int = 15,
    velocity_scale: float = 5.0,
    digit_size: int = 28,
) -> None:
    """Play one sequence up to its stop frame with physical metadata."""
    frames = dataset.frames[index]
    stop = int(dataset.stop_time[index])
    positions = dataset.position[index]
    velocities = dataset.velocity[index]
    centers = positions + digit_size / 2.0

    with plt.ion():
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(bottom=0.18)
        image = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255)
        arrows = ax.quiver(
            centers[0, :, 0],
            centers[0, :, 1],
            velocities[0, :, 0] * velocity_scale,
            velocities[0, :, 1] * velocity_scale,
            color=_COLORS,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            pivot="tail",
            width=0.008,
        )
        footers = [
            fig.text(
                0.01,
                0.10 - 0.08 * digit,
                "",
                color=_COLORS[digit],
                family="monospace",
                fontsize=8,
                va="bottom",
            )
            for digit in range(2)
        ]
        ax.set_xlim(-0.5, frames.shape[1] - 0.5)
        ax.set_ylim(frames.shape[1] - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"friction Moving MNIST: stop={stop}  arrows = velocity x{velocity_scale:g}",
            fontsize=9,
        )

        for frame in range(stop + 1):
            image.set_data(frames[frame])
            arrows.set_offsets(centers[frame])
            arrows.set_UVC(
                velocities[frame, :, 0] * velocity_scale,
                velocities[frame, :, 1] * velocity_scale,
            )
            for digit, footer in enumerate(footers):
                footer.set_text(_footer(dataset, index, frame, digit))
            plt.draw()
            plt.pause(1.0 / fps)
    plt.show()
