"""
In this file, we show how one can read data from a particular time period for use in modeling.
"""

# imports #
import json
from tidd.utils import Data, Transforms

if __name__ == '__main__':

    location = "hawaii" # can be hawaii or chile
    year = 2012 # 2015 for chile, 2012 for hawaii
    day_of_year = 302 # 290-304 for hawaii, 259 for chile

    # # try to read data from a specific location
    # df = Data().read_day(
    #     location=location,
    #     year=year,
    #     day_of_year=day_of_year,
    #     verbose=True
    # )
    #
    # print(df.sample(frac=1).head())
    #
    # print(df.shape)

    # read multiple days of data
    days = list(range(290, 305))

    df_all_days = Data().read_days(
        location=location,
        year=year,
        days=days,
        verbose=True
    )

    print(df_all_days.sample(frac=1).head())

    print(df_all_days.shape)

    # TODO: move the below to new example file
    # get the combinations of ground stations and satelli r4tes
    combinations = Transforms().get_station_satellite_combinations(
        dataframe=df_all_days
    )

    # we only have some combinations for testing
    combinations = [x for x in combinations if "G20" in x]

    # select the first set of data as an example
    station_sat = combinations[0]
    df_model = df_all_days.filter(regex=station_sat, axis=1).resample("1min").mean() # resample by mean

    # transform values by first getting the individual events
    events = Transforms().split_by_nan(
        dataframe=df_model,
        min_sequence_length=100
    )

    print(len(events))

    # continue transforming by converting the float data into images
    with open("../data/experiments/proof_of_concept/hawaii/tid_start_times.json", "rb") as f_in:
        labels = json.load(f_in)

    print(labels)

    Transforms().generate_images(
        events=events,
        labels=labels,
        output_dir=".",
        window_size=60,
        event_size=30,
        verbose=True
    )

