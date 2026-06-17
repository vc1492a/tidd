"""Pluggable image writers for saving encoded arrays to disk."""

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Writer(Protocol):
    """Interface for saving a 2-D image array to disk."""

    def save(self, array: np.ndarray, path: Path) -> None:
        """
        Persist *array* as an image file at *path*.

        :param array: 2-D image array.
        :param path: Destination file path (parent dirs must exist).
        """
        ...


class PILWriter:
    """
    Save images via Pillow -- fast, no matplotlib overhead.

    The array is normalised to 0-255, colormapped, and saved directly.
    """

    def __init__(self, colormap: str = "viridis", size: tuple[int, int] = (224, 224)):
        self.colormap = colormap
        self.size = size

    def save(self, array: np.ndarray, path: Path) -> None:
        import matplotlib
        from PIL import Image

        norm = (array - array.min()) / (array.max() - array.min() + 1e-12)
        cmap = matplotlib.colormaps[self.colormap]
        rgba = cmap(norm)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

        img = Image.fromarray(rgb, mode="RGB")
        if img.size != self.size:
            img = img.resize(self.size, Image.LANCZOS)
        img.save(path)


class MatplotlibWriter:
    """Save images via matplotlib (compatible with the legacy pipeline)."""

    def __init__(self, colormap: str = "viridis", figsize: tuple[int, int] = (5, 5)):
        self.colormap = colormap
        self.figsize = figsize

    def save(self, array: np.ndarray, path: Path) -> None:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=self.figsize, frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(array, cmap=self.colormap, origin="lower")
        fig.savefig(path)
        plt.close(fig)
