"""
A set of unit tests which tests the capabilities provided within tidd/data.py.
"""

import json
import pandas as pd
import pytest
import random
import shutil
from tidd.utils import Data, Transforms
import os


@pytest.fixture
def test_fixed_data() -> pd.DataFrame:
    """
    fixture that returns a Pandas DataFrame containing
    some data which can be used for testing.
    :return: a Pandas DataFrame
    """

    # read data
    df = Data().read_day(
        location="hawaii",
        year=2012,
        day_of_year=302,
        verbose=True
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


def test_read_data_day(test_random_data) -> None:
    """
    Tests whether the read_day function reads the appropriate year and day
    of year of data, and that it returns the data in the appropriate format.
    :return: None
    """

    file_path = test_random_data.split('/')

    s = tuple((int(file_path[3]), int(file_path[4]), file_path[2]))

    # try to read data from a specific location
    df = Data().read_day(
        location=s[2],
        year=s[0],
        day_of_year=s[1],
        verbose=True
    )

    # check to ensure that the returned data is a dataframe
    assert isinstance(df, pd.DataFrame)

    # check to ensure that the index is a Pandas DateTimeIndex
    assert type(df.index).__name__ == "DatetimeIndex"

    # check that the day corresponds to the correct day of year
    random_date = random.choice(pd.to_datetime(df.index.values))
    assert random_date.dayofyear == s[1]
    assert random_date.year == s[0]

    # check that the column count is the same across data types
    cols = ["_lat", "_lon", "_h_ipp", "_ele", "_azi"]
    counts = list()
    columns = list(df.columns.values)
    for c in cols:
        type_cols = [col for col in columns if c in col]
        counts.append(len(type_cols))

    assert all(x == counts[0] for x in counts)


def test_read_data_days(test_random_data) -> None:
    """
    Tests whether the read_days function reads the appropriate year and days
    of year of data, and that it returns the data in the appropriate format.
    :return: None
    """

    days = [300, 301]

    # first read in the data
    df = Data().read_days(
        location="hawaii",
        year=2012,
        days=days,
        verbose=True
    )

    # check to ensure that the returned data is a dataframe
    assert isinstance(df, pd.DataFrame)

    # check to ensure that the index is a Pandas DateTimeIndex
    assert type(df.index).__name__ == "DatetimeIndex"

    # check that the day corresponds to the correct day of year
    random_date = random.choice(pd.to_datetime(df.index.values))
    assert random_date.dayofyear in days
    assert random_date.year == 2012

    # check that the column count is the same across data types
    cols = ["_lat", "_lon", "_h_ipp", "_ele", "_azi"]
    counts = list()
    columns = list(df.columns.values)
    for c in cols:
        type_cols = [col for col in columns if c in col]
        counts.append(len(type_cols))

    assert all(x == counts[0] for x in counts)


def test_transform_split_by_nan(test_fixed_data) -> None:
    """
    Tests whether the split_by_nan returns the expected values.
    :return: None
    """

    # get the combinations of ground stations and satellites
    combinations = Transforms().get_station_satellite_combinations(
        dataframe=test_fixed_data
    )

    # we only have some combinations for testing
    combinations = [x for x in combinations if "G20" in x]

    # select the first set of data as an example
    station_sat = combinations[0]
    df_model = test_fixed_data.filter(regex=station_sat, axis=1).resample("1min").mean()  # resample by mean

    # transform values by first getting the individual events
    min_sequence_length = 100
    events = Transforms().split_by_nan(
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
    combinations = Transforms().get_station_satellite_combinations(
        dataframe=test_fixed_data
    )

    # assert more than one
    assert len(combinations) > 0

    # formatting
    assert type(combinations[0]) == str


# # TODO: fix below test. Works when ran as function but not as test not sure why.
# def test_transform_generate_images(test_fixed_data) -> None:
#     """
#     Tests whether the image generation code runs and generates image files.
#     :return: None
#     """
#
#     # get the combinations of ground stations and satellites
#     combinations = Transforms().get_station_satellite_combinations(
#         dataframe=test_fixed_data
#     )
#
#     # we only have some combinations for testing
#     combinations = [x for x in combinations if "G20" in x]
#
#     # select the first set of data as an example
#     station_sat = combinations[0]
#     df_model = test_fixed_data.filter(regex=station_sat, axis=1).resample("1min").mean()  # resample by mean
#
#     # transform values by first getting the individual events
#     min_sequence_length = 100
#     events = Transforms().split_by_nan(
#         dataframe=df_model,
#         min_sequence_length=min_sequence_length
#     )
#
#     # continue transforming by converting the float data into images
#     with open("../data/experiments/proof_of_concept/hawaii/tid_start_times.json", "rb") as f_in:
#         labels = json.load(f_in)
#
#     Transforms().generate_images(
#         events=events,
#         labels=labels,
#         output_dir=".",
#         window_size=60,
#         event_size=30,
#         verbose=True
#     )
#
#     pth = os.path.dirname(os.path.realpath(__file__))
#
#     # get all the images
#     images = list()
#     for root, dirs, files in os.walk(pth):
#         for file in files:
#             if file.endswith(".jpg"):
#                 images.append(os.path.join(root, file))
#
#     assert len(images) > 0
#
#     # check that both classes were generated
#     classes = sorted(list(set([x.split("/")[-2] for x in images])))
#     assert classes == ["anomalous", "normal"]
#
#     # find all directories
#     directories = [x for x in os.listdir(pth) if os.path.isdir(x)]
#
#     # delete each directory
#     [shutil.rmtree(x) for x in directories]
