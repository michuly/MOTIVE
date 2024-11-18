from imports_file import *
from simulation_parameters import *

# lons = [18.946, 26.731]
# lats = [38.120, 33.693]
lons = [-140, -130, -150]
lats = [0, -2, 2]

def find_lon_lat(path, grd_name_tot, lon_lat):
    (lon, lat) = lon_lat
    with Dataset(os.path.join(path, grd_name_tot)) as dat_grd:
        lon_array = dat_grd.variables['lon_rho'][0, :]
        lat_array = dat_grd.variables['lat_rho'][:, 0]
    closest_index_lon = np.argmin(np.abs(lon_array - lon))
    closest_index_lat = np.argmin(np.abs(lat_array - lat))
    return([closest_index_lon, closest_index_lat])

for lon, lat in zip(lons, lats):
    ind_lon, ind_lat = find_lon_lat(grd_path, grd_name_tot, (lon, lat))
    print('lon index for %f: %d' % (lon, ind_lon))
    print('lat index for %f: %d' % (lat, ind_lat))

