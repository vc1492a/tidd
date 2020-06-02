# Changelog
All notable changes to the software will be documented in this Changelog.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) 
and adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html). 

## 0.0.1
### Added

- A `src/data.py`, which contains functions for reading and manipulating 
data that is available for analysis and experimentation. 
- A unit testing suite that uses [pytest](https://docs.pytest.org/en/latest/) 
and [pytest-cov](https://pypi.org/project/pytest-cov/) to 
ensure the software is working as intended, along with a `test_data.py` 
file that tests the functionality of `src/data.py`. 
- Added an [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

### Changed 
- The project `readme.md` to include information about additional datasets 
available for the [2015 Chilean Earthquake](https://earthquake.usgs.gov/earthquakes/eventpage/us20003k7a/executive). 
This data is new data that is in addition to previous available data 
which covered a 2012 Hawaiian event. 