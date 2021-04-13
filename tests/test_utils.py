"""
A set of unit tests which tests the capabilities provided within tidd/data.py.
"""

import pandas as pd
import pytest
import random
from tidd.utils import Data
import os


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
    

