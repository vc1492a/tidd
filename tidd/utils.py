"""
Core data utilities for reading raw sTEC files and transforming time-series data.
"""

import datetime
import io
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import sys
from typing import List, Union


# set logging verbosity
logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # set to logging.DEBUG in development
logger = logging.getLogger()


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)


class Transform:

    @staticmethod
    def split_by_nan(dataframe: pd.DataFrame, min_sequence_length: int = 100) -> list:
        """
        Splits a Pandas DataFrame into a list of Pandas DataFrames based on periods of
        consecutive NaN values. Also only retains dataframes of a certain number of periods.
        :param dataframe: A Pandas Dataframe to split by consecutive NaNs.
        :param min_sequence_length: The minimum length of values for the returned dataframes.
        :return: a list of Pandas Dataframes with at least min_sequence_length observations.
        """

        nan_mask = dataframe.isna().any(axis=1)
        group_id = nan_mask.cumsum()
        groups = [group_df.dropna() for _, group_df in dataframe.groupby(group_id)]
        events = [g for g in groups if not g.empty and g.shape[0] > min_sequence_length]

        return events

    @staticmethod
    def sod_to_timestamp(dataframe: pd.DataFrame, year: int, day_of_year: int) -> pd.DataFrame:

        """
        Converts seconds of day information to timestamp that can be used to do time-series
        resampling and other functions.
        :param dataframe: a Pandas dataframe that contains the second of day as the index.
        :param year: the year for time represented by the seconds of day.
        :param day_of_year: the day of year (DOY) for the time represented by the seconds of day.
        :return: a Pandas dataframe that moves the second of day from the index to a column and contains timestamps
        as a Pandas DateTimeIndex.
        """

        # now convert second of day (sod) to timestamps
        sod = dataframe.index
        timestamps = list()
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)

        for s in sod:
            # hours, minutes, seconds
            hours = int(s // 3600)
            minutes = int((s % 3600) // 60)
            seconds = int((s % 60))

            # create a datetime object and append to the list
            date_time = datetime.datetime(date.year, date.month, date.day, hours, minutes, seconds)
            timestamps.append(date_time)

        # set the timestamps as a Pandas DateTimeIndex
        df = dataframe.reset_index()
        df["timestamp"] = timestamps
        df = df.set_index("timestamp")

        return df

    @staticmethod
    def group_consecutive_values(values: list, step: int = 1) -> list:
        """
        Return list of consecutive lists of numbers from values (number list).
        https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
        :param values: A series of values in a list.
        :param step: The step size.
        :return: a list of consecutive lists of numbers.
        """
        run = []
        result = [run]
        expect = None
        for v in values:
            if (v == expect) or (expect is None):
                run.append(v)
            else:
                run = [v]
                result.append(run)
            expect = v + step
        return result

    @staticmethod
    def _get_station_satellite_combinations(dataframe: pd.DataFrame) -> list:
        """
        For a given Pandas DataFrame, gets all the possible combinations of
        ground station and satellite.
        :param dataframe: A Pandas DataFrame containing the modeling data.
        :return: A list of ground station and satellite combination.
        """

        combinations = list(set(["_".join(x.split("_")[0:3]) for x in dataframe.columns.values if "sod" not in x]))

        return combinations

class Data:

    @staticmethod
    def read_data_from_file(file_name: Union[str, Path]) -> pd.DataFrame:

        """
        For a given satellite, reads in the satellite and returns a Pandas DataFrame.
        :param file_name: the filename of the data file.
        :return: a Pandas DataFrame containing the data for this particular satellite / set of data.
        """

        sat_name = str(file_name).split("/")[-1].split(".")[0][:4]
        ground_station_name = str(file_name).split("_")[-1].split(".")[0]
        pass_id = sat_name + "__" + ground_station_name

        f = open(file_name, 'r')
        line1 = f.readline()
        line1 = line1.replace('#', '').replace("dsTEC/dt [TECU/s]", "dsTEC/dt").replace("elev", "ele")
        rename_cols = line1.split()
        rename_cols.remove("sod")
        new_cols = list()

        # rename the columns
        for rn_col in rename_cols:
            new_col = pass_id + "_" + rn_col
            if rn_col == "dsTEC/dt":
                new_col = pass_id
            new_cols.append(new_col)
        new_cols = ["sod"] + new_cols

        df = pd.read_table(
            file_name,
            index_col='sod',
            sep="\t\t| ",
            names=new_cols,
            engine="python",
            skiprows=1
        )

        new_cols.remove('sod')

        df = df[new_cols]

        return df

