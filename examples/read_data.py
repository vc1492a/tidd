"""
In this file, we show how one can read data from a particular time period for use in modeling.
"""

# imports #
from tidd.utils import Data

location = "hawaii" # can be hawaii or chile
year = 2012 # 2015 for chile, 2012 for hawaii
day_of_year = 302 # 290-304 for hawaii, 259 for chile

# try to read data from a specific location
df = Data().read_day(
    location=location,
    year=year,
    day_of_year=day_of_year,
    verbose=True
)

print(df.sample(frac=1).head())

print(df.shape)
