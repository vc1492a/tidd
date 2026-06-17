"""
A set of unit tests which tests the capabilities provided within tidd/utils.py.
"""

import json
import pandas as pd
import pytest
import random
import shutil
from tidd.utils import Data, Transform
from tidd.pipeline import Pipeline, GADFEncoder, PILWriter
from tidd.pipeline.labels import Labels
from tidd.pipeline.runner import process_file
from tidd.pipeline.discovery import FileInfo
import os


@pytest.fixture
def test_fixed_data() -> pd.DataFrame:
    """
    fixture that returns a Pandas DataFrame containing
    some data which can be used for testing.
    :return: a Pandas DataFrame
    """

    # read data
    df = Data.read_data_from_file(
        "./tests/data/ahup3020.12o_G20.txt"
    )

    return df


def test_transform_sod_to_timestamp(test_fixed_data) -> None:
    """
    Tests whether the sod_to_timestamp returns the expected values.
    :return: None
    """

    # add timestamps
    test_fixed_data = Transform.sod_to_timestamp(
        test_fixed_data,
        year=2012,
        day_of_year=302
    )

    # check that the index is a datetime index
    assert type(test_fixed_data.index) == pd.DatetimeIndex

    # check that sod is in the column values
    assert "sod" in list(test_fixed_data.columns.values)


def test_transform_split_by_nan(test_fixed_data) -> None:
    """
    Tests whether the split_by_nan returns the expected values.
    :return: None
    """

    # get the combinations of ground stations and satellites
    combinations = Transform()._get_station_satellite_combinations(
        dataframe=test_fixed_data
    )

    # we only have some combinations for testing
    combinations = [x for x in combinations if "G20" in x]

    # select the first set of data as an example
    station_sat = combinations[0]

    # add timestamps
    test_fixed_data = Transform.sod_to_timestamp(
        test_fixed_data,
        year=2012,
        day_of_year=302
    )

    df_model = test_fixed_data.filter(regex=station_sat, axis=1).resample("1min").mean()  # resample by mean

    # transform values by first getting the individual events
    min_sequence_length = 100
    events = Transform().split_by_nan(
        dataframe=df_model,
        min_sequence_length=min_sequence_length
    )

    # check that the length is appropriate
    assert len(events) > 0
    assert len(events[0]) >= min_sequence_length


def test_transform_get_station_satellite_combinations(test_fixed_data) -> None:
    """
    Tests whether various combinations of satellites can be retrieved
    from the Pandas DataFrame.
    :return: None
    """

    # get the combinations of ground stations and satellites
    combinations = Transform()._get_station_satellite_combinations(
        dataframe=test_fixed_data
    )

    # assert more than one
    assert len(combinations) > 0

    # formatting
    assert type(combinations[0]) == str


@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_pipeline_generate_images(tmp_path) -> None:
    """
    Tests whether the pipeline image generation produces labeled images.
    """

    labels_dict = {
        "302": {
            "G04": {"start": 31400, "finish": 33200},
            "G07": {"start": 31160, "finish": 32960},
            "G08": {"start": 31900, "finish": 33700},
            "G10": {"start": 29900, "finish": 31700},
            "G20": {"start": 31150, "finish": 32950},
        }
    }

    fi = FileInfo(
        path="./tests/data/ahup3020.12o_G20.txt",
        location="test",
        year=2012,
        day_of_year=302,
        station="ahup",
        satellite="G20",
    )
    labels = Labels(labels_dict)

    count = process_file(
        file_info=fi,
        labels=labels,
        encoder=GADFEncoder(),
        writer=PILWriter(),
        output_dir=tmp_path,
        split="train",
        window_size=60,
    )

    assert count > 0

    images = list(tmp_path.rglob("*.jpg"))
    assert len(images) > 0

    classes = sorted({p.parent.name for p in images})
    assert "anomalous" in classes or "normal" in classes
