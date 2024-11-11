import netCDF4 as nc # https://bobbyhadz.com/blog/python-note-this-error-originates-from-subprocess
import numpy as np
import os

lons = [29.861, 33.719]
lats = [35.162, 32.628]
# lons = [18.946, 26.731]
# lats = [38.120, 33.693]
# lons = [20.120010376, 28.291136863]
# lats = [37.479999542, 33.158442888]

def find_lon_lat(path, grd_name, lon_lat):
    (lon, lat) = lon_lat
    with nc.Dataset(os.path.join(path, grd_name)) as dat_grd:
        lon_array = dat_grd.variables['lon_rho'][0, :]
        lat_array = dat_grd.variables['lat_rho'][:, 0]

    closest_index_lon = np.argmin(np.abs(lon_array - lon))
    print('Longitude:%f\tIndex:%d' % (lon, closest_index_lon))
    closest_index_lat = np.argmin(np.abs(lat_array - lat))
    print('Latidute:%f\tIndex:%d' % (lat, closest_index_lat))
    return([closest_index_lon, closest_index_lat])

for lon, lat in zip(lons, lats):
     find_lon_lat(path, grd_name, (lon,lat))