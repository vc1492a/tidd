# imports
import datetime
import io
import json
import logging
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import operator
import os
import pandas as pd
from pathlib import Path
from pyts.image import GramianAngularField
import sys
from tidd.plotting import gramian_angular_field
from tqdm import tqdm
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

        # split by NaN
        events = np.split(dataframe, np.where(np.isnan(dataframe))[0])

        # keep non-NaN entries
        events = [ev[~np.isnan(ev)] for ev in events if not isinstance(ev, np.ndarray)]

        # filter by min_sequence_length
        events = [ev.dropna() for ev in events if not ev.empty and ev.shape[0] > min_sequence_length]

        return events

    @staticmethod
    def sod_to_timestamp(df: pd.DataFrame, year: int, day_of_year: int) -> pd.DataFrame:

        """
        # TODO:
        """

        # now convert second of day (sod) to timestamps
        sod = df.index
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
        df = df.reset_index().drop(columns="sod")
        df["timestamp"] = timestamps
        df = df.set_index("timestamp")

        return df

    @staticmethod
    def _get_station_satellite_combinations(dataframe: pd.DataFrame) -> list:
        """
        For a given Pandas DataFrame, gets all the possible combinations of
        ground station and satellite.
        :param dataframe: A Pandas DataFrame containing the modeling data.
        :return: A list of ground station and satellite combination.
        """

        combinations = list(set(["_".join(x.split("_")[0:3]) for x in dataframe.columns.values]))

        return combinations

    @staticmethod
    def generate_images(events: list, labels: dict, output_dir: Union[str, Path], window_size: int = 60,
                        verbose: bool = True) -> None:

        """
        Generates images from windowed time-series data, specifically Gramnian Angular Difference Fields (GADFs).
        :param events: A list of events (streams of time-series) to process into images.
        :param labels: A dictionary of subject matter expert labels used to distinguish which time periods
        are representative of anomalies (e.g. 302 - 6400 (second of day)).
        :param output_dir: The path in which to export the generated images.
        :param window_size: The window size (in minutes) to use for image generation. Default 60.
        :param verbose: If true, show progress and other information.
        :return: None
        """

        # establish a logger
        tqdm_out = TqdmToLogger(logger, level=logging.INFO)

        for period in tqdm(events, file=tqdm_out, total=len(events), mininterval=3, disable=operator.not_(verbose)):

            # get the doy
            doy = period.index[0].dayofyear

            # get the combo
            combo = Transform()._get_station_satellite_combinations(period)[0]
            # TODO (low priority): raise a warning if not length 1 above?

            # generate the path if it doesn't exist
            Path(output_dir + "/" + combo + "/labeled/anomalous").mkdir(parents=True, exist_ok=True)
            Path(output_dir + "/" + combo + "/labeled/normal").mkdir(parents=True, exist_ok=True)
            Path(output_dir + "/" + combo + "/unlabeled").mkdir(parents=True, exist_ok=True)

            # convert to seconds of the day for later annotation
            period["sod"] = (period.index.hour * 60 + period.index.minute) * 60 + period.index.second

            # get the satellite
            sat = combo.split("__")[1]

            # get the start time of the sat and the end time
            try:
                anom_range = [labels[str(doy)][sat]["start"], labels[str(doy)][sat]["finish"]]
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
                    X_new = Transform().data_to_image(
                        df=subset[combo]
                    )

                    # plot and save
                    ax = gramian_angular_field(
                        X_new
                    )

                    # save to a particular path based on if we are within the anomalous range
                    if (period.iloc[idx]["sod"] + window_size) in list(range(anom_range[0], anom_range[1])):
                        plt.savefig(output_dir + "/" + combo + "/labeled/anomalous/" + str(doy) + "_" + str(
                            idx) + "_" + str(idx + window_size) + "_GAF.jpg")
                    else:
                        plt.savefig(
                            output_dir + "/" + combo + "/labeled/normal/" + str(doy) + "_" + str(
                                idx) + "_" + str(idx + window_size) + "_GAF.jpg")

                    if "validation" in output_dir + "/" + combo:
                        plt.savefig(
                            output_dir + "/" + combo + "/unlabeled/" + str(doy) + "_" + str(
                                idx) + "_" + str(idx + window_size) + "_GAF.jpg")

                    plt.close()

    @staticmethod
    def data_to_image(df: Union[pd.Series, np.array]) -> pd.Series:

        # TODO

        # now generate the field
        transformer = GramianAngularField()
        X_new = transformer.fit_transform(np.array([df]))

        return X_new


class Data:

    """
    # TODO:
    """

    # def read_image_from_file(self):

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

    @staticmethod
    def _get_image_directories(path):
        directory_list = []

        # return nothing if path is a file
        if os.path.isfile(path):
            return []

        # add dir to directorylist if it contains .txt files
        if len([f for f in os.listdir(path) if f.endswith('.jpg')]) > 0:
            directory_list.append(path)

        for d in os.listdir(path):
            new_path = os.path.join(path, d)
            if os.path.isdir(new_path):
                directory_list += Data._get_image_directories(new_path)

        return directory_list

    def _pipe_prepare_training_validation_data(self, path_objects: tuple) -> None:

        """
        # TODO
        """

        try:

            # load in the labels
            sat_name = str(path_objects[1]).split("_")[-1].split(".")[0]
            location = str(path_objects[1]).split("/")[-4]
            year = int(str(path_objects[1]).split("/")[-3])
            day_of_year = int(str(path_objects[1]).split("/")[-2])
            labels = path_objects[2]

            # if we have a label for the satellite but not doy, then normal
            # if on doy with label, then need to distinguish between anom and normal with labels
            satellites_with_labels = list()
            for k, v in labels.items():
                for k1, v1 in v.items():
                    satellites_with_labels.append(k1)

            if sat_name in satellites_with_labels:

                # read data from stand-alone file
                df = self.read_data_from_file(path_objects[1])

                # convert sod to timestamp
                df = Transform.sod_to_timestamp(
                    df,
                    year=year,
                    day_of_year=day_of_year
                )

                # resample to 1 min # TODO (low priority): expose as parameter through Experiment
                df = df.resample("1min").mean()

                # transform values by first getting the individual events
                events = Transform().split_by_nan(
                    dataframe=df,
                    min_sequence_length=100
                )

                # generate the images based on the ground truth labels
                path_prefix = "/".join(str(path_objects[1]).split("/")[:-4])

                Transform().generate_images(
                    events=events,
                    labels=labels,
                    output_dir=path_prefix + "/experiments/" + path_objects[0] + "/" + location + "/" + path_objects[4],
                    window_size=path_objects[3],
                    verbose=False
                )

        except Exception as ex:
            logging.warning(RuntimeWarning, str(ex))

    @staticmethod
    def prepare_training_validation_data(experiment_name: str = "experiment-1",
                                         training_data_paths: List[Union[str, Path]] = ["./"],
                                         validation_data_paths: List[Union[str, Path]] = None,
                                         window_size: int = 60
                                         ) -> list:

        """
        # TODO:
        """

        # assign types for later differentiation
        paths = dict()
        paths_to_process = training_data_paths
        for p in training_data_paths:
            paths[p] = {"type": "train", "file_paths": list(),
                        "labels": json.load(open(p + "/tid_start_finish_times.json", "rb"))}
        if validation_data_paths is not None:
            [paths_to_process.append(v) for v in validation_data_paths]
            for p in validation_data_paths:
                paths[p] = {"type": "validation", "file_paths": list(),
                            "labels": json.load(open(p + "/tid_start_finish_times.json", "rb"))}

        # gather all file paths for processing
        #  first gather all the file paths
        for p in list(paths.keys()):

            # get all available years and for each year, the available day of years (DoYs)
            years = [name for name in os.listdir(p) if os.path.isdir(os.path.join(p, name))]
            for y in years:
                doys = [name for name in os.listdir(p + "/" + y) if os.path.isdir(os.path.join(p + "/" + y, name))]
                # for each doy, get the file paths
                for d in doys:
                    location_year_doy_path = Path(p) / y / d
                    satellite_paths = [location_year_doy_path / Path(p) for p in os.listdir(location_year_doy_path) if
                                       p != ".DS_Store"]
                    paths[p]["file_paths"] += satellite_paths

        # gather all file paths and process rapidly in parallel
        all_paths = [paths[p]["file_paths"] for p in paths]
        all_paths_flat = [item for sublist in all_paths for item in sublist]

        # for each of the file paths, create a tuple containing the name
        # of the experiment, the file path, and the labeled data
        # TODO: check that validation data is unlabeled
        path_objects = [tuple((experiment_name, p, paths["/".join(str(p).split("/")[:-3])]["labels"], window_size,
                               paths["/".join(str(p).split("/")[:-3])]["type"])) for p in all_paths_flat]

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        pool = mp.Pool(os.cpu_count() - 1)  # to keep the system alive yo

        with pool as pp:

            # labels are only generated for those which contain labeled data
            with tqdm(file=tqdm_out, total=len(path_objects), mininterval=10) as pbar:
                for i, _ in enumerate(pp.imap_unordered(Data()._pipe_prepare_training_validation_data, path_objects)):
                    pbar.update()

        training_data_path = "/".join(training_data_paths[-1].split("/")[:-1]) + "/experiments/" + experiment_name + "/train"
        validation_data_path = None
        if validation_data_paths is not None:
            validation_data_path = "/".join(validation_data_paths[-1].split("/")[:-1]) + "/experiments/" + experiment_name  + "/validation"

        return [training_data_path, validation_data_path]

