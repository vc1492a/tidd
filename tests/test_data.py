"""
A set of unit tests which tests the capabilities provided within src/utils.py.
"""

import pandas as pd
import random
from src.data import read_day


def test_read_data_day() -> None:
    """
    Tests whether the read_day function reads the appropriate year and day
    of year of data, and that it returns the data in the appropriate format.
    :return: None
    """

    # define a year and day of year
    year = 2012
    doy = 303

    # try to read data from a specific location
    df = read_day(
        location="hawaii",
        year=year,
        day_of_year=doy
    )

    # check to ensure that the returned data is a dataframe
    assert isinstance(df, pd.DataFrame)

    # check to ensure that the index is a Pandas DateTimeIndex
    assert type(df.index).__name__ == "DatetimeIndex"

    # check that the day corresponds to the correct day of year
    random_date = random.choice(pd.to_datetime(df.index.values))
    assert random_date.dayofyear == doy
    assert random_date.year == year

