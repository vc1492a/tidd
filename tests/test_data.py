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

    settings = [
        tuple((2012, 303, "hawaii")),
        tuple((2015, 259, "chile"))
    ]

    for s in settings:

        # try to read data from a specific location
        df = read_day(
            location=s[2],
            year=s[0],
            day_of_year=s[1]
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
