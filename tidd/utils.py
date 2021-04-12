"""
A collection of functions covering basic capabilities like:

- Reading data from files and delivering Pandas DataFrames
- Outputting the tqdm progress bar in sys.stdout.

"""

# imports #
import datetime
import io
import logging
import operator
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

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


class Data:

    """
    A class which provides functionality for reading data from local disk.
    """

    def __init__(self):

        # gather the data for each satellite from this day and location
        self.stec_dfs = list()

    def _process_file(self, satellite_name: str) -> pd.DataFrame:

        """
        For a given satellite, reads in the satellite and returns a Pandas DataFrame.
        :param satellite_name: the name of the satellite.
        :return: a Pandas DataFrame containing the data for this particular satellite / set of data.
        """

        sat_name = str(satellite_name).split("/")[-1].split(".")[0][:4]
        ground_station_name = str(satellite_name).split("_")[-1].split(".")[0]
        pass_id = sat_name + "__" + ground_station_name

        f = open(satellite_name, 'r')
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
            satellite_name,
            index_col='sod',
            sep="\t\t| ",
            names=new_cols,
            engine="python",
            skiprows=1
        )

        new_cols.remove('sod')

        self.stec_dfs.append(df[new_cols])

        return df[new_cols]

    def read_day(self, location: str = "hawaii", year: int = 2000, day_of_year: int = 300,
                 verbose: bool = True) -> pd.DataFrame:
        """
        Reads the data for a particular location and day of year.
        :param location: Specifies the location in which we want to load data (default: hawaii).
        :param year: Specifies the year in which to load data, specified as an integer (default: 2000).
        :param day_of_year: Specifies the day of year in which to load data, specified as an
        integer (default: 300).
        :param verbose: Dictates whether messages and progress bars are pushed to stdout.
        :return: A Pandas dataframe that includes the data for the specified location and day, with
        a Pandas datetime index and columns which represent combinations of satellites and ground
        stations.
        """

        # specify the root path to the data
        data_path = Path(__file__).parents[1] / "data"
        year = year
        day = str(day_of_year)
        location_year_doy_path = data_path / location / str(year) / day

        # collect the paths for each satellite
        satellite_paths = [location_year_doy_path / Path(p) for p in os.listdir(location_year_doy_path) if
                           p != ".DS_Store"]

        if verbose:
            logging.info("Reading files for: " + location + " " + str(year) + " " + str(day_of_year))

        # pool = mp.Pool(os.cpu_count() - 1)  # to keep the system alive yo

        # with pool as pp:

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        with tqdm(file=tqdm_out, total=len(satellite_paths), mininterval=3, disable=operator.not_(verbose)) as pbar:

            # for sat, _ in enumerate(pp.imap_unordered(self._process_file, satellite_paths)):
            for sat in satellite_paths:

                self._process_file(satellite_name=sat)
                pbar.update()

        if verbose:
            logging.info("Merging files for: " + location + " " + str(year) + " " + str(day_of_year))
        # merge all of the satellite specific dataframes together

        stec_values = pd.concat(self.stec_dfs, axis=1)

        # convert second of day (sod) to timestamps
        sod = stec_values.index
        timestamps = list()
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)

        for s in tqdm(sod, file=tqdm_out, total=len(sod), mininterval=3, disable=operator.not_(verbose)):

            # hours, minutes, seconds
            hours = int(s // 3600)
            minutes = int((s % 3600) // 60)
            seconds = int((s % 60))

            # create a datetime object and append to the list
            date_time = datetime.datetime(date.year, date.month, date.day, hours, minutes, seconds)
            timestamps.append(date_time)

        # set the timestamps as a Pandas DateTimeIndex
        df = stec_values.reset_index().drop(columns="sod")
        df["timestamp"] = timestamps
        df = df.set_index("timestamp")

        return df



