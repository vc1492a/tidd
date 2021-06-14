"""
In this file, we show how one can read data from a particular time period for use in modeling.
"""

# imports #
import matplotlib.pyplot as plt
import seaborn as sns
from tidd.utils import Data, Transform

if __name__ == '__main__':

    # try to read data from a specific location
    df = Data.read_data_from_file(
        file_name="../data/chile/2015/259/aeda2590_G24.txt"
    )

    # print(df.head())
    # print(df.tail())

    # convert sod to timestamp
    df = Transform.sod_to_timestamp(
        df,
        year=2015,
        day_of_year=259
    )

    # resample to 1 min # TODO (low priority): expose as parameter through Experiment
    df = df.resample("1min").mean()

    # transform values by first getting the individual events
    events = Transform().split_by_nan(
        dataframe=df,
        min_sequence_length=100
    )

    print(len(events))

    print(events[1].tail())

    sns.lineplot(data=events[1], x="sod", y="aeda__G24")
    plt.show()



