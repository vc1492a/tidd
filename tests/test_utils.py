"""
A set of unit tests which tests the capabilities provided within tidd/data.py.
"""

import json
import pandas as pd
import pytest
import random
import shutil
from tidd.utils import Data, Transform
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
        "./data/hawaii/2012/302/ahup3020.12o_G20.txt"
    )

    return df


@pytest.fixture
def test_random_data() -> 'str':
    """
    fixture that creates the random data for testing
    with read_data
    :yield: the file paths of the random data
    """
    path = './data/test'
    year_path = path + '/2012'
    day_path = year_path + '/300'
    os.mkdir(path)
    os.mkdir(year_path)
    os.mkdir(day_path)
    file_path = day_path + '/test3000.12o_G01.txt'
    with open(file_path, 'w') as test_file:
        index_row = (
            'sod\t\t'
            'dsTEC/dt\t\t'
            'lon\t\tlat\t\t'
            'h_ipp\t\telev\t\tazi\n'
        )
        test_data_row = (
            '8250.0\t\t'
            '-0.01919536491673681\t\t'
            '-165.3638966481876\t\t'
            '24.961242295471187\t\t'
            '349982.9213181995\t\t'
            '10.1815617588652\t\t'
            '302.5328897665781\n'
        )
        test_file.write(index_row)
        test_file.write(test_data_row)
    yield file_path

    # deleting the test files
    os.remove(file_path)
    file_path = file_path.split('/')
    # deleting the directories
    os.rmdir('/'.join(file_path[:-1]))
    os.rmdir('/'.join(file_path[:-2]))
    os.rmdir('/'.join(file_path[:-3]))


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
def test_transform_generate_images(test_fixed_data) -> None:
    """
    Tests whether the image generation code runs and generates image files.
    :return: None
    """

    # get the combinations of ground stations and satellites
    combinations = Transform()._get_station_satellite_combinations(
        dataframe=test_fixed_data
    )

    # we only have some combinations for testing
    combinations = [x for x in combinations if "G20" in x]

    # add timestamps
    df_model = Transform.sod_to_timestamp(
        test_fixed_data,
        year=2012,
        day_of_year=302
    )

    # select the first set of data as an example
    df_model = df_model.resample("1min").mean()  # resample by mean

    # transform values by first getting the individual events
    min_sequence_length = 100
    events = Transform().split_by_nan(
        dataframe=df_model,
        min_sequence_length=min_sequence_length
    )

    # continue transforming by converting the float data into images
    labels = {
        "302": {
            "G04": {
                "start": 31400,
                "finish": 33200
            },
            "G07": {
                "start": 31160,
                "finish": 32960
            },
            "G08": {
                "start": 31900,
                "finish": 33700
            },
            "G10": {
                "start": 29900,
                "finish": 31700
            },
            "G20": {
                "start": 31150,
                "finish": 32950
            }
        }
    }

    pth = "./tests"

    Transform().generate_images(
        events=events,
        labels=labels,
        output_dir=pth,
        window_size=60,
        verbose=True
    )

    # get all the images
    images = list()
    for root, dirs, files in os.walk(pth):
        for file in files:
            if file.endswith(".jpg"):
                images.append(os.path.join(root, file))

    # check that some images were made
    assert len(images) > 0

    # check that both classes were generated
    classes = sorted(list(set([x.split("/")[-2] for x in images])))
    assert classes == ["anomalous", "normal"]

    # find all image data directories
    directories = [x[0] for x in os.walk(pth) if len(x[0].split("/")) == 3]

    # delete each directory
    [shutil.rmtree(x) for x in directories]
