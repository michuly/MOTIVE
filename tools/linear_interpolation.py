import numpy as np

def linear_interp(var, z_r, z_new):
    """
    looks for the value of var, in between sigma levels.
    Example, if z_new is in between z_r[i] and z_r[i+1] values,
    the interpolation will be:
    (z[i+1]-z_new)/(z[i+1]-z[i])*var[i]+(z_new-z[i])/(z[i+1]-z[i])*var[i+1]
    axis of interpolation is 2!
    """
    if type(var)==tuple:
        x_size, y_size, z_size = var[0].shape
    else:
        x_size, y_size, z_size = var.shape
    X2, Y2 = np.meshgrid(np.arange(x_size), np.arange(y_size), indexing='ij')

    # calculate the distance between z_r and z_new
    dz = z_r - z_new[:, :, np.newaxis]
    dz_up = dz.copy()
    dz_up[dz_up < 0] = 9999.
    idx_up = np.argmin(dz_up, axis=2)
    idx_up[np.all(dz<0, axis=2)] = z_size - 1  # if z_new>h, then take upper value
    idx_up[np.all(dz>0, axis=2)] = 1  # if z_new<h, then take bottom value
    idx_down = idx_up-1

    # dz_down = dz.copy()
    # dz_down[dz_down > 0] = -9999.
    # idx_down = np.argmax(dz_down, axis=2)

    # calculate the "weights" for the var[i], and v[i+1]
    weights_down = np.abs((z_r[X2.flatten(), Y2.flatten(), idx_up.flatten()] - z_new.flatten()))
    weights_up = np.abs((z_r[X2.flatten(), Y2.flatten(), idx_down.flatten()]- z_new.flatten()))
    weights_norm = weights_down + weights_up
    print(np.where(weights_norm==0))

    # the final result
    if type(var)==tuple:
        var_interp=[]
        for _var in var:
            var_tmp = _var[X2.flatten(), Y2.flatten(), idx_up.flatten()] * weights_up/weights_norm + \
                      _var[X2.flatten(), Y2.flatten(), idx_down.flatten()] * weights_down/weights_norm
            var_interp.append(var_tmp.reshape(x_size, y_size))
    else:
        var_interp = var[X2.flatten(), Y2.flatten(), idx_up.flatten()] * weights_up / weights_norm + \
                  var[X2.flatten(), Y2.flatten(), idx_down.flatten()] * weights_down / weights_norm
        var_interp = var_interp.reshape(x_size, y_size)

    return var_interp
