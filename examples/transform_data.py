"""
In this file, we show how one can read data from a particular available dataset(s) and convert it
for use in modeling by using the Experiment object. The sTEC data is read from disk and
transformed using Gramian Angular Difference Fields (GADFs) for use in modeling and inference (predictions).
"""

# imports #
import datetime
import json
import logging
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import os
import sys
from tqdm import tqdm

from tidd.utils import Data, Transforms
from tidd.utils import TqdmToLogger


# set logging verbosity
logging.basicConfig(stream=sys.stdout, level=logging.INFO)  # set to logging.DEBUG in development
logger = logging.getLogger()
tqdm_out = TqdmToLogger(logger, level=logging.INFO)


#from fastai.vision.all import Adam, resnet34

# from tidd.model import Model, Experiment


# # need to define a model before you can define an experiment
# M = Model(
#     architecture=resnet34,
#     batch_size=256,
#     learning_rate=0.0001,
#     optimization_function=Adam
# )

# we want to u use the 2012 Hawaii data for training, 2015 Chile data for validation


training_data_paths = [
    "../data/hawaii"
]

validation_data_paths = [
    "../data/chile"
]



# TODO: create a pipeline that can be used by any path to generate the files in a parallel way

# assign types for later differentiation
paths = dict()
for p in training_data_paths:
    paths[p] = {"type": "train", "file_paths": list(), "labels": json.load(open(p + "/tid_start_finish_times.json", "rb"))}
# for p in validation_data_paths:
#     paths[p] = {"type": "validation", "file_paths": list(), "labels": json.load(open(p + "/tid_start_finish_times.json", "rb"))}

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

    #

# gather all file paths and process rapidly in parallel
all_paths = [paths[p]["file_paths"] for p in paths]
all_paths_flat = [item for sublist in all_paths for item in sublist]

# print(all_paths_flat)


# this is what is multiprocessed, the main pipeline
def rename_later(path) -> None:

    try:

        # load in the labels
        sat_name = str(path).split("_")[-1].split(".")[0]
        location = str(path).split("/")[-4]
        year = int(str(path).split("/")[-3])
        day_of_year = int(str(path).split("/")[-2])

        labels = paths["/".join(str(path).split("/")[:-3])]["labels"]

        if str(day_of_year) in labels.keys():

            if sat_name in labels[str(day_of_year)]:


                # go
                df = Data._process_file(path)

                # convert second of day (sod) to timestamps
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

                # durrr
                df = df.resample("1min").mean()

                # transform values by first getting the individual events
                events = Transforms().split_by_nan(
                    dataframe=df,
                    min_sequence_length=100
                )

                # generate the images based on the ground truth labels
                Transforms().generate_images(
                    events=events,
                    labels=labels,
                    # TODO: note exp name
                    output_dir="../data/experiments/proof_of_concept_revA/" + location + "/" + paths["/".join(str(path).split("/")[:-3])]["type"],
                    window_size=60,
                    verbose=False
                )

    except Exception as ex:
        logging.warning(RuntimeWarning, str(ex))


if __name__ == '__main__':

    pool = mp.Pool(os.cpu_count() - 1) # to keep the system alive yo

    with pool as pp:

        with tqdm(file=tqdm_out, total=len(all_paths_flat), mininterval=10) as pbar:
            for i, _ in enumerate(pp.imap_unordered(rename_later, all_paths_flat)):
                pbar.update()

    pbar.close()



# E = Experiment(
#     model=M,
#     name="proof_of_concept_revA",
#     cuda_device=3,
#     training_data_paths="../data/experiments/proof_of_concept/hawaii/train",
#     validation_data_paths="../data/experiments/proof_of_concept/hawaii/validation",
#     test_percent=0.2,
#     parallel_gpus=False,
#     max_epochs=50
# )







# # imports #
# import json
# from tidd.utils import Data, Transforms
#
# if __name__ == '__main__':
#
#     location = "hawaii" # can be hawaii or chile
#     year = 2012 # 2015 for chile, 2012 for hawaii
#     day_of_year = 302 # 290-304 for hawaii, 259 for chile
#
#     # read multiple days of data
#     days = list(range(290, 305))
#
#     df_all_days = Data().read_days(
#         location=location,
#         year=year,
#         days=days,
#         verbose=True
#     )
#
#     print(df_all_days.sample(frac=1).head())
#
#     print(df_all_days.shape)
#
#     # get the combinations of ground stations and satellites
#     combinations = Transforms().get_station_satellite_combinations(
#         dataframe=df_all_days
#     )
#
#     # we only have some combinations for testing
#     combinations = [x for x in combinations if "G20" in x]
#
#     # select the first set of data as an example
#     station_sat = combinations[0]
#     df_model = df_all_days.filter(regex=station_sat, axis=1).resample("1min").mean() # resample by mean
#
#     # transform values by first getting the individual events
#     events = Transforms().split_by_nan(
#         dataframe=df_model,
#         min_sequence_length=100
#     )
#
#     # continue transforming by converting the float data into images
#     with open("../data/experiments/proof_of_concept/hawaii/tid_start_finish_times.json", "rb") as f_in:
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
