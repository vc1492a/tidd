import datetime
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def read_day(location: str = "hawaii", year: int = 2000, day_of_year: int = 300) -> pd.DataFrame:
    """
    Reads the data for a particular location and day of year.
    :param location: Specifies the location in which we want to load data (default: hawaii).
    :param year: Specifies the year in which to load data, specified as an integer (default: 2000).
    :param day_of_year: Specifies the day of year in which to load data, specified as an
    integer (default: 300).
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
    satellite_paths = [location_year_doy_path / Path(p) for p in os.listdir(location_year_doy_path) if p != ".DS_Store"]

    # gather the data for each satellite from this day and location
    stec_dfs = list()
    for sat in tqdm(satellite_paths):
        df = pd.read_table(sat, index_col="sod", sep="\t\t", engine="python")
        # rename the columns
        sat_name = str(sat).split("/")[-1].split(".")[0]
        ground_station_name = str(sat).split("_")[-1].split(".")[0]
        pass_id = sat_name + "__" + ground_station_name
        df = df.rename(columns={"dsTEC/dt": pass_id})
        stec_dfs.append(df[[pass_id]])

    # merge all of the satellite specific dataframes together
    stec_values = pd.concat(stec_dfs, axis=1)

    # TODO: change sod to timestamp index datetime index
    # convert second of day (sod) to timestamps
    sod = stec_values.index
    timestamps = list()
    for s in sod:

        # day of year and year to month, day
        date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year - 1)

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


