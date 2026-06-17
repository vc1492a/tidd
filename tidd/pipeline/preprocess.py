"""Read, resample, split, and window raw sTEC data."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tidd.pipeline.discovery import FileInfo
from tidd.utils import Data, Transform


@dataclass
class Window:
    """A single windowed time-series segment ready for encoding."""

    array: np.ndarray
    sod_start: float
    sod_end: float
    event_idx: int
    window_idx: int
    station: str
    satellite: str
    day_of_year: int
    pass_id: str


def read_and_resample(
    file_info: FileInfo,
    resample_freq: str = "1min",
) -> pd.DataFrame:
    """
    Read a raw sTEC file and resample to a uniform frequency.

    :param file_info: Metadata for the file to read.
    :param resample_freq: Pandas frequency string for resampling.
    :returns: Resampled DataFrame with DateTimeIndex.
    """

    df = Data.read_data_from_file(file_info.path)
    df = Transform.sod_to_timestamp(df, year=file_info.year, day_of_year=file_info.day_of_year)
    df = df.resample(resample_freq).mean()
    return df


def split_events(
    df: pd.DataFrame,
    min_length: int = 100,
) -> list[pd.DataFrame]:
    """
    Split a DataFrame into contiguous non-NaN event segments.

    :param df: Resampled DataFrame with potential NaN gaps.
    :param min_length: Minimum rows for a segment to be kept.
    :returns: List of contiguous DataFrames.
    """

    return Transform.split_by_nan(df, min_sequence_length=min_length)


def window_events(
    events: list[pd.DataFrame],
    window_size: int = 60,
) -> list[Window]:
    """
    Slide a fixed-size window across each event and produce :class:`Window` objects.

    :param events: List of contiguous event DataFrames (with ``sod`` column).
    :param window_size: Number of rows per window.
    :returns: List of :class:`Window` instances.
    """

    windows: list[Window] = []

    for event_idx, event in enumerate(events):
        pass_id = Transform._get_station_satellite_combinations(event)[0]
        station = pass_id.split("__")[0]
        satellite = pass_id.split("__")[1]
        doy = event.index[0].dayofyear

        for win_idx in range(len(event) - window_size + 1):
            subset = event.iloc[win_idx : win_idx + window_size]
            windows.append(Window(
                array=subset[pass_id].values,
                sod_start=float(subset["sod"].values[0]),
                sod_end=float(subset["sod"].values[-1]),
                event_idx=event_idx,
                window_idx=win_idx,
                station=station,
                satellite=satellite,
                day_of_year=doy,
                pass_id=pass_id,
            ))

    return windows
