## Toolkit for Ionospheric Disturbance Detection (TIDD)

A toolkit for the detection of Traveling Ionospheric Disturbances (TIDs) in the Earth's atmosphere as a result of 
tsunami waves, earthquakes, large explosions and other phenomena. A collaboration between individuals and teams the 
NASA Jet Propulsion Laboratory (JPL), Sapienza University of Rome and the University of California - Los Angeles (UCLA). 

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/vc1492a/sTEC-d-dt-Anomaly-Detection/archive/0.1.0.tar.gz)
[![Language](https://img.shields.io/badge/python-3.8-blue)](#)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## About 

**Top-Down View of TIDs Detected by Satellite over Hawaii**
![TIDs_hawaii](https://github.com/vc1492a/sTEC-d-dt-Anomaly-Detection/blob/main/images/geoplot_gopm_light.png)

Tsunamis can trigger internal gravity waves (IGWs) that are able to propagate to the ionosphere, causing a perturbation in the natural Total Electron Content (TEC). 
These perturbations are often referred as Traveling Ionospheric Disturbances (TIDs) and are easily detectable through 
the Global Navigation Satellite System (GNSS) signal. The perturbation in the below image is generated 
by a TID. 

**GOPM Ground Station G07 Satellite slant Total Electron Content (sTEC) data**
![gopm G07 doy](https://github.com/vc1492a/sTEC-d-dt-Anomaly-Detection/blob/main/images/gopm_G07_doy.png)

The large quantity of GNSS data currently available allows us to explore the possibility of using deep learning methods 
for TID detection. This toolkit demonstrates the effectiveness of training a convolutional neural network (CNN) to 
detect signs of IGWs. The approach utilized in this toolkit employs Gramian Angular Difference Fields (GADFs) to 
encode the time-series as images for model training. 

**Animation of GADFs over TIDs**
![TIDs GADF](https://github.com/vc1492a/sTEC-d-dt-Anomaly-Detection/blob/main/images/anom.gif)

In a real-time system, slant Total Electron Content (sTEC) may be calculated, windowed, converted 
to an image using GADFs and fed into a model for predicting whether a TID is present and occurring. 
Currently, simple experiments show conversion from float data to a prediction takes 
approximately 1.1. seconds. 

## Getting Started 

First, setup a Python virtual environment in which to install project 
dependencies and then install the library. Then, pull the appropriate 
branch and then install: 

```
pip install . -e
```

Make sure to check out the `notebooks` and `data` directories 
which contain Jupyter notebooks with latest work and the source data 
used in the experiments. 

## Dependencies

This project requires that Python version is versioned between 3.5 and 
3.8. Requirements for the software, running tests, and notebooks are 
listed in `requirements.txt`, `requirements_ci.txt`, and `requirements_notebooks.txt`, 
and may be installed into a virtual environment in the following way: 

```bash
pip install -r requirements.txt
pip install -r requirements_ci.txt # for unit tests 
```

Some of the visualizations within the Jupyter notebooks require 
the [geos](https://trac.osgeo.org/geos/) library, which can be installed 
using homebrew on macOS:

```
brew install geos
```

On a linux machine, GEOS can be installed without root access if working 
in Anaconda environments: 

```
conda install -c anaconda geos
```

Then, install the notebook dependencies: 

```
pip install -r requirements_notebooks.txt
```

## The Data

The data includes data from both a 2012 Hawaii tsunami event and 
a [2015 Chilean earthquake](https://earthquake.usgs.gov/earthquakes/eventpage/us20003k7a/executive). 
Within the the `data` directory there is from each event, organized 
by subdirectories which correspond to the year and day of year. 

### Downloading the Data 

First, data must be downloaded and unpacked into the `data` directory:

```shell
curl -O https://tsunami-detection.s3-us-west-1.amazonaws.com/data.tar.gz && tar -xvzf data.tar.gz && rm data.tar.gz
```

Data used for experiments is located in `data/experiments`, while raw historical 
data is available in the `chile` and `hawaii` directories. The data is also used in a variety of tests. 

### About the Data

In every folder, you find a file for each satellite in view from a GPS 
station. The value of the slant total electron content 
(sTEC) encountered by the GPS signal during its path in the ionosphere 
from the satellite (e.g G10) to the GPS receiver (e.g ahup) is available 
for select days of the year.

The data files have 7 columns:
- **Sod**: it represents the second of the day, it is my time array
- **dStec/dt**: the variations in time of the slant total electron 
content (the parameter of interest) epoch by epoch (it is like a velocity)
- **Lon**: longitude epoch by epoch of the IPP, the point to which we refer 
the sTEC estimations
- **Lat**: latitude epoch by epoch of the IPP, the point to which we refer 
the sTEC estimations
- **Hipp**: height epoch by epoch of the IPP, the point to which we refer 
the sTEC estimations
- **Azi**: the azimuth of the satellite epoch by epoch
- **Ele**: the elevation of the satellite epoch by epoch (usually we 
don’t consider data with elevation under 20 degrees since they are too 
noisy)#

#### Hawaii
 
The day of the earthquake is 302. Data is available for days of year 290 through 304. 

#### Chile 

Day of year 259 is available which is the day of the earthquake.

## Contributing

Please use the issue tracker to report any erroneous behavior or desired 
feature requests. 

If you would like to contribute to development, please fork the repository and make 
any changes to a branch which corresponds to an open issue. Hot fixes 
and bug fixes can be represented by branches with the prefix `fix/` versus 
`feature/` for new capabilities or code improvements. Pull requests will 
then be made from these branches into the repository's `dev` branch 
prior to being pulled into `master`. Pull requests which are works in 
progress or ready for merging should be indicated by their respective 
prefixes (`[WIP]` and `[MRG]`). Pull requests with the `[MRG]` prefix will be 
reviewed prior to being pulled into the `master` branch. 

### Tests
When contributing, please ensure to run unit tests and add additional tests as 
necessary if adding new functionality. To run the unit tests, use `pytest`: 

```
python -m  pytest --cov=tidd --capture=no --log-cli-level=CRITICAL
```

This should report the result of your unit tests as well as information 
about code coverage. 

### Versioning
[Semantic versioning](http://semver.org/) is used for this project. 
If contributing, please conform to semantic versioning guidelines when 
submitting a pull request.

### Updating the Changelog
Core contributors are responsible for maintaining `changelog.md` in 
coordination with new releases. 

### Releasing New Versions 

To release a new version of the software, simply tag the realse after building 
the distribution and wheels: 

```bash
python setup.py sdist bdist_wheel
git tag x.x.x -m 'some commit message here'
git push origin --tags <branch_name>
git add dist/
git push origin <branch_name>
```

## License
This project is licensed under the 
[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.

## Research
If citing this work, use the following: 

```
# TBD
```

### Motivation

* Real-Time Detection of Tsunami Ionospheric Disturbances with a 
Stand-Alone GNSS Receiver: A Preliminary Feasibility Demonstration. 
Savastano G., et. al.. Nature Scientific Reports, 2017. [PDF](https://www.nature.com/articles/srep46607.pdf).
* Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding. Hundman K., et. al. 
Knowledge Discovery and Data Mining (KDD), 2018. [PDF](https://dl.acm.org/doi/pdf/10.1145/3219819.3219845).
* Imaging Time-Series to Improve Classification and Imputation. Whang Z., Oats T. 2015. [PDF](https://arxiv.org/pdf/1506.00327.pdf).

## Contributors
- [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/)
    - [Valentino Constantinou](https://www.github.com/vc1492a)
- [Sapienza University of Rome](https://www.uniroma1.it/en/pagina-strutturale/home)
    - [Michela Ravanelli](https://scholar.google.com/citations?user=w6sS0Q0AAAAJ&hl=it)
- [University of California - Los Angeles (UCLA)](https://www.ucla.edu/)
    - [Hamlin Liu](https://www.github.com/hamlinliu17)
    - [Dr. Jacob Bortnik](https://atmos.ucla.edu/people/faculty/jacob-bortnik)

## Acknowledgements
- [NASA Jet Propulsion Laboratory](https://jpl.nasa.gov/)
  - Bryan Bales
  - [Dr. Chris Mattmann](https://github.com/chrismattmann)
  - [Virisha Timmaraju](https://github.com/virisha22)
  - [Annie Didier](https://github.com/adidier17)
  - [Dr. Brian Wilson](https://github.com/BrianWilson1)