"""Pluggable time-series-to-image encoders."""

from typing import Protocol, runtime_checkable

import numpy as np
from pyts.image import GramianAngularField


@runtime_checkable
class Encoder(Protocol):
    """Interface for time-series-to-image encoders."""

    def encode(self, series: np.ndarray) -> np.ndarray:
        """
        Convert a 1-D time-series window into a 2-D image array.

        :param series: 1-D array of length *window_size*.
        :returns: 2-D array of shape ``(window_size, window_size)``.
        """
        ...


class GADFEncoder:
    """Gramian Angular Difference Field encoder."""

    def encode(self, series: np.ndarray) -> np.ndarray:
        transformer = GramianAngularField()
        image = transformer.fit_transform(series.reshape(1, -1))
        return image[0]
