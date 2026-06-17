"""Discover raw sTEC data files on disk."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Union


@dataclass(frozen=True)
class FileInfo:
    """Metadata about a single raw sTEC data file."""

    path: Path
    location: str
    year: int
    day_of_year: int
    station: str
    satellite: str


def discover_files(raw_path: Union[str, Path]) -> list[FileInfo]:
    """
    Walk a raw data directory tree and return metadata for every data file.

    Expected layout::

        raw_path/
          {year}/
            {day_of_year}/
              {station}{doy}0.{yy}o_{satellite}.txt

    :param raw_path: Root of a location directory (e.g. ``data/hawaii``).
    :returns: Sorted list of :class:`FileInfo` instances.
    """

    raw_path = Path(raw_path)
    location = raw_path.name
    results: list[FileInfo] = []

    if not raw_path.is_dir():
        return results

    for year_dir in sorted(raw_path.iterdir()):
        if not year_dir.is_dir() or not year_dir.name.isdigit():
            continue
        year = int(year_dir.name)

        for doy_dir in sorted(year_dir.iterdir()):
            if not doy_dir.is_dir() or not doy_dir.name.isdigit():
                continue
            doy = int(doy_dir.name)

            for file_path in sorted(doy_dir.iterdir()):
                if file_path.name.startswith(".") or not file_path.is_file():
                    continue
                if file_path.name == "tid_start_finish_times.json":
                    continue

                station = file_path.name.split(".")[0][:4]
                satellite = file_path.name.split("_")[-1].split(".")[0]

                results.append(FileInfo(
                    path=file_path,
                    location=location,
                    year=year,
                    day_of_year=doy,
                    station=station,
                    satellite=satellite,
                ))

    return results
