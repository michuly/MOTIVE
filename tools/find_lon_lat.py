
from intro_file import *

# lons = [18.946, 26.731]
# lats = [38.120, 33.693]
lons = [-143, -137]
lats = [-2, 2]

def find_lon_lat(path, grd_name, lon_lat):
    (lon, lat) = lon_lat
    with nc.Dataset(os.path.join(path, grd_name)) as dat_grd:
        lon_array = dat_grd.variables['lon_rho'][0, :]
        lat_array = dat_grd.variables['lat_rho'][:, 0]

    closest_index_lon = np.argmin(np.abs(lon_array - lon))
    closest_index_lat = np.argmin(np.abs(lat_array - lat))
    return([closest_index_lon, closest_index_lat])

for lon, lat in zip(lons, lats):
    ind_lon, ind_lat = find_lon_lat(data_path, grd_name, (lon, lat))
    print('lon index for %f: %d', (lon, ind_lon))
    print('lat index for %f: %d', (lat, ind_lat))

