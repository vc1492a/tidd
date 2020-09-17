import datetime
import os
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import minmax_scale
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
    stec_values = None
    first = True
    
    print("Reading dataframes...")
    
    for sat in tqdm(satellite_paths):
        
        sat_name = str(sat).split("/")[-1].split(".")[0][:4]
        ground_station_name = str(sat).split("_")[-1].split(".")[0]
        pass_id = sat_name + "__" + ground_station_name
        
        f = open(sat, 'r')
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
            sat,
            index_col='sod',
            sep="\t\t| ",
            names=new_cols,
            engine="python",
            skiprows=1
        )
        
        new_cols.remove('sod')
    
        
        stec_dfs.append(df[new_cols])
    
   
    print("Concatenating dataframes...")
    # merge all of the satellite specific dataframes together
    
    stec_values = pd.concat(stec_dfs, axis=1)
    

    # convert second of day (sod) to timestamps
    sod = stec_values.index
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
    df = stec_values.reset_index().drop(columns="sod")
    df["timestamp"] = timestamps
    df = df.set_index("timestamp")

    return df


def normalize(dataframe: 'pd.DataFrame', minimum: int = 0, maximum: int = 1) -> 'pd.DataFrame':
    """
    Rescales the dStec/dt values on a scale from 0 to 1 or other scales common
    to many analysis and modeling problems.
    :param dataframe: a dataframe created by the read_day function that
    contains dStec/dt values.
    :param minimum: the minimum scale value, default 0.
    :param maximum: the maximum scale value, default 1.
    :return: a dataframe identical to the input dataframe, but with
    normalized values.
    """

    for col in dataframe.columns.values:
        if len(col.split("__")[1]) == 3:
            dataframe[col] = minmax_scale(X=dataframe[col], feature_range=(minimum, maximum))

    return dataframe
