# Detecting Anomalies in Slant Total Electron Content

(abstract goes here)

## The Data

There are four folders in the `data` directory, where each folder 
corresponds to a day ([see GNSS Web](http://navigationservices.agi.com/GNSSWeb/)). 
The day of the earthquake is 302. We processed 5 days: two days before 
the earthquake (day 300 and 301), the day of the earthquake (302) and 
two days after the earthquake (303 and 304).

In every folder, you find a file for each satellite in view from a GPS 
station: so you have the value of the slant total electron content 
(sTEC) encountered by the GPS signal during its path in the ionosphere 
from the satellite (e.g G10) to the GPS receiver (e.g ahup) for every 
day.

The files have 7 columns:
- Sod: it represents the second of the day, it is my time array
- dStec/dt: the variations in time of the slant total electron 
content (the parameter of interest) epoch by epoch (it is like a velocity)
- Lon: longitude epoch by epoch of the IPP, the point to which we refer 
the sTEC estimations
- Lat: latitude epoch by epoch of the IPP, the point to which we refer 
the sTEC estimations
- Hipp: height epoch by epoch of the IPP, the point to which we refer 
the sTEC estimations
- Azi: the azimuth of the satellite epoch by epoch
- Ele: the elevation of the satellite epoch by epoch (usually we 
don’t consider data with elevation under 20 degrees since they are too 
noisy)

