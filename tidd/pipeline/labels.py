"""Ground-truth label management for TID events."""

import json
from pathlib import Path
from typing import Union


class Labels:
    """
    Load and query ground-truth anomaly labels from a
    ``tid_start_finish_times.json`` file.

    The JSON structure is::

        {
          "<day_of_year>": {
            "<satellite>": {"start": <sod>, "finish": <sod>},
            ...
          },
          ...
        }
    """

    def __init__(self, label_data: dict):
        self._data = label_data

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Labels":
        with open(path, "r") as f:
            return cls(json.load(f))

    @classmethod
    def from_raw_path(cls, raw_path: Union[str, Path]) -> "Labels":
        """Load labels from the ``tid_start_finish_times.json`` in *raw_path*."""
        return cls.from_file(Path(raw_path) / "tid_start_finish_times.json")

    def classify_window(
        self,
        day_of_year: int,
        satellite: str,
        sod_end: float,
    ) -> str:
        """
        Determine whether a window is ``"anomalous"`` or ``"normal"``.

        Matches the original logic: a window is anomalous when its
        trailing edge (``sod_end``) falls within the labeled anomaly range.

        :param day_of_year: Day-of-year for the window.
        :param satellite: Satellite identifier (e.g. ``"G20"``).
        :param sod_end: Second-of-day at the trailing edge of the window.
        :returns: ``"anomalous"`` or ``"normal"``.
        """

        doy_key = str(day_of_year)
        try:
            start = self._data[doy_key][satellite]["start"]
            finish = self._data[doy_key][satellite]["finish"]
        except KeyError:
            return "normal"

        if start <= int(sod_end) < finish:
            return "anomalous"
        return "normal"
