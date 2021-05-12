"""
A collection of functions covering basic capabilities like:

- Reading data from files and delivering Pandas DataFrames
- Outputting the tqdm progress bar in sys.stdout.
- Rescaling values to a specific range. This is a common technique used in many machine and
deep learning applications.

"""

# imports #
import datetime
import io
import logging
import matplotlib.pyplot as plt
import numpy as np
import operator
import os
import pandas as pd
from pathlib import Path
from pyts.image import GramianAngularField
import sys
from tqdm import tqdm
from typing import Union

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

    @staticmethod
    def _process_file(satellite_name: str) -> pd.DataFrame:

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

        return df[new_cols]

    def read_day(self, location: str = "hawaii", year: int = 2012, day_of_year: int = 300,
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
            logger.info("Reading files for: " + location + " " + str(year) + " " + str(day_of_year))

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        stec_dfs = list()

        for sat in tqdm(satellite_paths, file=tqdm_out, total=len(satellite_paths), mininterval=3, disable=operator.not_(verbose)):
            stec_dfs.append(self._process_file(satellite_name=sat))

        if verbose:
            logger.info("Merging files for: " + location + " " + str(year) + " " + str(day_of_year))

        # merge all of the satellite specific dataframes together
        stec_values = pd.concat(stec_dfs, axis=1)

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

    @staticmethod
    def read_days(days: list, location: str = "hawaii", year: int = 2012, verbose: bool = True) -> pd.DataFrame:

        """
        Reads multiple days worth of data for a particular location and year and returns the data as a Pandas
        DataFrame.
        :param days: a list of integers representing days of the year, e.g. [1, 2, 3]
        :param location: Specifies the location in which we want to load data (default: hawaii).
        :param year: Specifies the year in which to load data, specified as an integer (default: 2000).
        :param verbose: Dictates whether messages and progress bars are pushed to stdout.
        :return: A Pandas dataframe that includes the data for the specified location and year for multiple days, with
        a Pandas datetime index and columns which represent combinations of satellites and ground stations.
        """

        # establish a logger
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        # TODO: add multi-core processing         # num_cores: int = os.cpu_count() - 1
        dfs = list()
        for doy in tqdm(days, file=tqdm_out, total=len(days), mininterval=3, disable=operator.not_(verbose)):

            dfs.append(
                Data().read_day(
                    location=location,
                    year=year,
                    day_of_year=doy,
                    verbose=False
                )
            )

        # concatenate the dataframes loaded previously into one large dataframe
        df_all_days = pd.concat(dfs)

        return df_all_days


class Transforms:

    """
    Set of functions that applies transforms to the source data for use in modeling and
    analysis.
    """

    @staticmethod
    def get_station_satellite_combinations(dataframe: pd.DataFrame) -> list:
        """
        For a given Pandas DataFrame, gets all the possible combinations of
        ground station and satellite.
        :param dataframe: A Pandas DataFrame containing the modeling data.
        :return: A list of ground station and satellite combination.
        """

        combinations = list(set(["_".join(x.split("_")[0:3]) for x in dataframe.columns.values]))

        return combinations

    @staticmethod
    def split_by_nan(dataframe: pd.DataFrame, min_sequence_length: int = 100) -> list:
        """
        Splits a Pandas DataFrame into a list of Pandas DataFrames based on periods of
        consecutive NaN values. Also only retains dataframes of a certain number of periods.
        :param dataframe: A Pandas Dataframe to split by consecutive NaNs.
        :param min_sequence_length: The minimum length of values for the returned dataframes.
        :return: a list of Pandas Dataframes with at least min_sequence_length observations.
        """

        # split by NaN
        events = np.split(dataframe, np.where(np.isnan(dataframe))[0])

        # keep non-NaN entries
        events = [ev[~np.isnan(ev)] for ev in events if not isinstance(ev, np.ndarray)]

        # filter by min_sequence_length
        events = [ev.dropna() for ev in events if not ev.empty and ev.shape[0] > min_sequence_length]

        return events

    @staticmethod
    def group_consecutives(vals: list, step: int = 1) -> list:
        """
        Return list of consecutive lists of numbers from vals (number list).
        https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-in-a-numpy-array
        :param vals: A series of values.
        :param step: The step size.
        :return: a list of consecutive lists of numbers.
        """
        run = []
        result = [run]
        expect = None
        for v in vals:
            if (v == expect) or (expect is None):
                run.append(v)
            else:
                run = [v]
                result.append(run)
            expect = v + step
        return result

    @staticmethod
    def generate_images(events: list, labels: dict, output_dir: Union[str, Path], window_size: int = 60,
                        event_size: int = 30, verbose: bool = True) -> None:

        """
        Generates images from windowed time-series data, specifically Gramnian Angular Difference Fields (GADFs).
        :param events: A list of events (streams of time-series) to process into images.
        :param labels: A dictionary of subject matter expert labels used to distinguish which time periods
        are representative of anomalies (e.g. 302 - 6400 (second of day)).
        :param output_dir: The path in which to export the generated images.
        :param window_size: The window size (in minutes) to use for image generation. Default 60.
        :param event_size: The event size (in minutes) to use in approximating the ending time. Default 30.
        :param verbose: If true, show progress and other information.
        :return: None
        """

        # establish a logger
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        for period in tqdm(events, file=tqdm_out, total=len(events), mininterval=3, disable=operator.not_(verbose)):

            # get the doy
            doy = period.index[0].dayofyear

            # get the combo
            combo = Transforms().get_station_satellite_combinations(period)[0]
            # TODO: raise a warning if not length 1 above?

            # generate the path if it doesn't exist
            Path(output_dir + "/" + combo + "/" + str(doy) + "/GAF/anomalous").mkdir(parents=True, exist_ok=True)
            Path(output_dir + "/" + combo + "/" + str(doy) + "/GAF/normal").mkdir(parents=True, exist_ok=True)

            # convert to seconds of the day for later annotation
            period["sod"] = (period.index.hour * 60 + period.index.minute) * 60 + period.index.second

            # get the satellite
            sat = combo.split("__")[1]

            # get the start time of the sat and the end time
            try:
                anom_range = [labels[str(doy)][sat], labels[str(doy)][sat] + (event_size * 60)]
            except KeyError:
                anom_range = [0, 1] # NOTE: assumes window size is never less than 3, may want userwarning

            # process all the windows
            for idx in list(range(period.shape[0])):

                # get subsetted window
                subset = period.iloc[idx:idx + window_size, :]

                # if the data is smaller than the window size, do not process
                if subset.shape[0] < window_size:
                    pass

                else:

                    # now generate the field
                    transformer = GramianAngularField()
                    X_new = transformer.fit_transform(np.array([subset[combo]]))

                    figure = plt.figure(figsize=(5, 5), frameon=False)

                    ax = plt.Axes(figure, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    figure.add_axes(ax)

                    figure = plt.imshow(X_new[0], cmap='viridis', origin='lower')

                    x_axis = figure.axes.get_xaxis()
                    x_axis.set_visible(False)

                    y_axis = figure.axes.get_yaxis()
                    y_axis.set_visible(False)

                    # save to a particular path based on if we are within the anomalous range
                    if (period.iloc[idx]["sod"] + window_size) in list(range(anom_range[0], anom_range[1])):
                        plt.savefig(output_dir + "/" + combo + "/" + str(doy) + "/GAF/anomalous/" + str(doy) + "_" + str(
                            idx) + "_" + str(idx + window_size) + "_GAF.jpg")
                    else:
                        plt.savefig(
                            output_dir + "/" + combo + "/" + str(doy) + "/GAF/normal/" + str(doy) + "_" + str(
                                idx) + "_" + str(idx + window_size) + "_GAF.jpg")

                    plt.close()

