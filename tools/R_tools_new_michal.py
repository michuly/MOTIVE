##################################################################################
# R_TOOLS
###################################################################################
"""





"""

###################################################################################
# Load modules
###################################################################################

# for numeric functions
import numpy as np
from numpy import float32
from netCDF4 import Dataset

# copy data
from copy import copy

# ROMSTOOLS
import R_tools_fort as toolsF

# for plotting
import matplotlib.pyplot as py

import time as tm


#################################################################################
# Complimentary functions
#################################################################################
def rank(a):
    return len(a.shape)


####################################################
def Zeros(*args, **kwargs):
    kwargs.update(dtype=float32)
    return np.zeros(*args, **kwargs)


#################################################
# rho_eos (from rho_eos.F in romsucla)
#################################################


def rho_eos(T,S,z_r,z_w,rho0):

    (rho) = toolsF.rho_eos(T,S,z_r,z_w,rho0)

    return rho

#################################################
# rho_eos (from rho_eos.F in romsucla)
#################################################


def rho1_eos(T,S,z_r,z_w,rho0):

    (rho1) = toolsF.rho1_eos(T,S,z_r,rho0)

    return rho1


#################################################################################
# Compute all shear components on a rho-rho grid of the horizontal velocity fild
#################################################################################
# param u, v, w: velocities
# param z_r, z_w: z values of sigma, in p coordinates and rho coordiantes, respectively
# param pm, pn: 1/dx, 1/dy
# coor: rho coordinates or psi coordinates.
#

def vort(u, v, pm, pn, z_r=None, z_w=None, simple=False, mask=None, coord='p'):
    dvdx = DvDx(v, pm, z_r=z_r, z_w=z_w, simple=simple, mask=None, coord=coord)
    dudy = DuDy(u, pn, z_r=z_r, z_w=z_w, simple=simple, mask=None, coord=coord)
    if coord == 'p':
        return (dvdx - dudy)
    else:
        return psi2rho(dvdx - dudy)


def strain(u, v, pm, pn, z_r=None, z_w=None, simple=False, mask=None, coord='p'):
    dvdx = DvDx(v, pm, z_r=z_r, z_w=z_w, simple=simple, mask=None, coord=coord)
    dudy = DuDy(u, pn, z_r=z_r, z_w=z_w, simple=simple, mask=None, coord=coord)
    dudx = DuDx(u, pm, z_r=z_r, z_w=z_w, simple=simple, mask=None, coord=coord)
    dvdy = DvDy(v, pn, z_r=z_r, z_w=z_w, simple=simple, mask=None, coord=coord)
    # return dvdx, dudy, dudx, dvdy
    if coord == 'p':
        return (np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2))
    else:
        return psi2rho(np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2))


def DuDx(u, pm, z_r=None, z_w=None, simple=False, mask=None, coord='p'):
    if simple:
        dudx = Zeros((pm.shape[0], pm.shape[1], u.shape[2]))
        dudx[1:-1, :, :] = diffx(u, rho2u(pm))
    else:
        dudx = Zeros((pm.shape[0], pm.shape[1], u.shape[2]))
        dudx[1:-1, :, :] = (diffxi(u, rho2u(pm), rho2u(z_r), rho2u(z_w), mask=mask))

    return dudx


def DuDy(u, pn, z_r=None, z_w=None, simple=False, mask=None, coord='r'):
    if simple:
        return psi2rho(diffy(u, rho2u(pn)))
    else:
        return psi2rho(diffeta(u, rho2u(pn), rho2u(z_r), rho2u(z_w), mask=mask))


def w2rho_s(var_w, z_r, z_w):
    #print var_w.shape, z_r.shape
    w_r = z_r * 0
    w_r = var_w[:,:,:-1] * (z_w[:,:,1:] - z_r[:,:,:]) + var_w[:,:,1:] * (z_r[:,:,:] - z_w[:,:,:-1])
    w_r /= (z_w[:,:,1:] - z_w[:,:,:-1])
    return w_r


def DuDz(u, z_r, z_w, simple=False, coord='r'):
    dz_r = z_r[:, :, 1:] - z_r[:, :, :-1]
    dz_r[dz_r == 0] = np.nan
    dudz = z_w * 0
    dudz[:, :, 0] = 0
    dudz[:, :, -1] = 0
    dudz[:, :, 1:-1] = u2rho((u[:, :, 1:] - u[:, :, :-1]) / (0.5 * (dz_r[1:, :, :] + dz_r[:-1, :, :])))
    if coord == 'r':
        dudz = w2rho_s(dudz, z_r, z_w)

    return dudz


def DvDx(v, pm, z_r=None, z_w=None, simple=False, mask=None, coord='r'):
    if simple:
        return psi2rho(diffx(v, rho2v(pm)))
    else:
        return psi2rho(diffxi(v, rho2v(pm), rho2v(z_r), rho2v(z_w), mask=mask))


def DvDy(v, pn, z_r=None, z_w=None, simple=False, mask=None, coord='p'):
    if simple:
        dvdy = Zeros((pn.shape[0], pn.shape[1], v.shape[2]))
        dvdy[:, 1:-1, :] = diffy(v, rho2v(pn))
    else:
        dvdy = Zeros((pn.shape[0], pn.shape[1], v.shape[2]))
        dvdy[:, 1:-1, :] = (diffeta(v, rho2v(pn), rho2v(z_r), rho2v(z_w), mask=mask))

    return dvdy


def DvDz(v, z_r, z_w, simple=False, coord='r'):
    dz_r = z_r[:, :, 1:] - z_r[:, :, :-1]
    dz_r[dz_r == 0] = np.nan
    dvdz = z_w * 0
    dvdz[:, :, 0] = 0
    dvdz[:, :, -1] = 0
    dvdz[:, :, 1:-1] = v2rho((v[:, :, 1:] - v[:, :, :-1]) / (0.5 * (dz_r[:, 1:, :] + dz_r[:, :-1, :])))
    if coord == 'r':
        dvdz = w2rho_s(dvdz, z_r, z_w)

    return dvdz


#######################################################
# interpolate a 3D variable on horizontal levels of constant depths (FORTRAN version, much faster)
#######################################################
def zlevs(gd, nch, itime=None):
    if itime is not None:
        zeta = ncload(nch, 'zeta', itime)
        return toolsF.zlevs(gd['h'], zeta, nch.hc, nch.Cs_r, nch.Cs_w)
    else:
        return toolsF.zlevs(gd['h'], gd['h'] * 0, nch.hc, nch.Cs_r, nch.Cs_w)


def mooring_zlevs(h, zeta, hc, Cs_r, Cs_w):
    # (gd['h'], zeta, nch.hc, nch.Cs_r, nch.Cs_w)
    return toolsF.zlevs(h, zeta, hc, Cs_r, Cs_w)


#######################################################
def vinterp(var, depths, z_r, z_w=None, mask=None, imin=0, jmin=0, kmin=1, floattype=np.float32, interp_sfc=1,
            interp_bot=0, below=None, **kwargs):
    """
    L,M,N - are the shape of x,y,z. nz is the index of z on which to interpolate the variable.
    """


    if mask == None:  mask = np.ones((z_r.shape[0], z_r.shape[1]), order='F', dtype=floattype); mask[
        z_r[:, :, -1] == 0] = 0

    if z_w is None:
        print('no z_w specified')
        z_w = Zeros((z_r.shape[0], z_r.shape[1], z_r.shape[2] + 1), order='F')
        z_w[:, :, 1:-1] = 0.5 * (z_r[:, :, 1:] + z_r[:, :, :-1])
        z_w[:, :, 0] = z_r[:, :, 0] - (z_r[:, :, 1] - z_r[:, :, 0])
        z_w[:, :, -1] = z_r[:, :, -1] + (z_r[:, :, -1] - z_r[:, :, -2])

    if rank(depths) == 1:
        newz = np.asfortranarray(Zeros((z_r.shape[0], z_r.shape[1], len(depths))) + depths, dtype=floattype)
    else:
        newz = depths

    print(z_r.shape)
    print(newz.shape)
    print(below.shape)
    if interp_bot == 1:
        print("data will be interpolated below ground")
        below = 1000.
        vnew = toolsF.sigma_to_z_intr_bot(Lm, Mm, N, nz, z_r, z_w, mask, var, newz, below, imin, jmin, kmin, 9999.)
    elif interp_sfc == 1:
        print("no interpolation below ground")
        print(z_r.shape, z_w.shape, mask.shape, var.shape, newz.shape)
        vnew = toolsF.sigma_to_z_intr_sfc(Lm, Mm, N, nz, z_r, z_w, mask, var, newz, imin, jmin, kmin, 9999.)
    else:
        print("no interpolation below ground")
        vnew = toolsF.sigma_to_z_intr(Lm, Mm, N, nz, z_r, z_w, mask, var, newz, imin, jmin, kmin, 9999.)
        # vnew = toolsF.sigma_to_z_intr(z_r, z_w, mask, var, newz, imin, jmin, kmin, 9999.)

    vnew[np.abs(vnew) == 9999.] = np.nan

    return vnew


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


def closest_values(var, z_r, z_new):

    x_size, y_size, z_size = var.shape
    X2, Y2 = np.meshgrid(np.arange(x_size), np.arange(y_size), indexing='ij')
    idx_up = np.argmin(np.abs(z_r - z_new[:, :, np.newaxis]), axis=2)
    var_interp = var[X2.flatten(), Y2.flatten(), idx_up.flatten()]
    return var_interp.reshape(x_size, y_size)


def linear_interp_old(var, z_r, z_new):
    """
    looks for the value of var, in between sigma levels.
    Example, if z_new is in between z_r[i] and z_r[i+1] values, 
    the interpolation will be:
    (z[i+1]-z_new)/(z[i+1]-z[i])*var[i]+(z_new-z[i])/(z[i+1]-z[i])*var[i+1]
    axis of interpolation is 2!
    """
    x_size, y_size, z_size = var.shape
    X2, Y2 = np.meshgrid(np.arange(x_size), np.arange(y_size), indexing='ij')

    # calculate the distance between z_r and z_new
    dz = z_r - z_new[:, :, np.newaxis]
    dz_up = dz.copy()
    dz_up[dz_up < 0] = 9999.
    idx_up = np.argmin(dz_up, axis=2)
    idx_up[np.all(dz<0,axis=2)] = z_size-1  # if z_new>h, then take upper value
    idx_down = idx_up-1

    dz_down = dz.copy()
    dz_down[dz_down > 0] = -9999.
    idx_down = np.argmax(dz_down, axis=2)
    idx_down[np.all(dz>0, axis=2)] = 0  # if z_new>h, then take bottom value

    # calculate the "weights" for the var[i], and v[i+1]
    weights_norm = z_r[X2.flatten(), Y2.flatten(), idx_up.flatten()] - z_r[X2.flatten(), Y2.flatten(), idx_down.flatten()]
    print(np.where(weights_norm==0))
    weights_down = (z_r[X2.flatten(), Y2.flatten(), idx_up.flatten()] - z_new.flatten()) / weights_norm
    weights_up = (z_new.flatten() - z_r[X2.flatten(), Y2.flatten(), idx_down.flatten()]) / weights_norm

    # the final result
    var_interp = var[X2.flatten(), Y2.flatten(), idx_up.flatten()] * weights_up + var[X2.flatten(), Y2.flatten(), idx_down.flatten()] * weights_down

    return var_interp.reshape(x_size, y_size)




#######################################################
# Transfert a field at psi points to rho points
#######################################################
def psi2rho(var_psi):
    if rank(var_psi) < 3:
        var_rho = psi2rho_2d(var_psi)
    else:
        var_rho = psi2rho_3d(var_psi)

    return var_rho


##############################
def psi2rho_2d(var_psi):
    [M, L] = var_psi.shape
    Mp = M + 1
    Lp = L + 1
    Mm = M - 1
    Lm = L - 1

    var_rho = Zeros((Mp, Lp))
    var_rho[1:M, 1:L] = 0.25 * (var_psi[0:Mm, 0:Lm] + var_psi[0:Mm, 1:L] + var_psi[1:M, 0:Lm] + var_psi[1:M, 1:L])
    var_rho[0, :] = var_rho[1, :]
    var_rho[Mp - 1, :] = var_rho[M - 1, :]
    var_rho[:, 0] = var_rho[:, 1]
    var_rho[:, Lp - 1] = var_rho[:, L - 1]

    return var_rho


#############################
def psi2rho_3d(var_psi):
    [Mz, Lz, Nz] = var_psi.shape
    var_rho = Zeros((Mz + 1, Lz + 1, Nz))

    for iz in range(0, Nz, 1):
        var_rho[:, :, iz] = psi2rho_2d(var_psi[:, :, iz])

    return var_rho


#######################################################
# Transfert a field at rho points to psi points
#######################################################
def rho2psi(var_rho):
    if rank(var_rho) < 3:
        var_psi = rho2psi_2d(var_rho)
    else:
        var_psi = rho2psi_3d(var_rho)

    return var_psi


##############################
def rho2psi_2d(var_rho):
    var_psi = 0.25 * (var_rho[1:, 1:] + var_rho[1:, :-1] + var_rho[:-1, :-1] + var_rho[:-1, 1:])

    return var_psi


#############################
def rho2psi_3d(var_rho):
    var_psi = 0.25 * (var_rho[1:, 1:, :] + var_rho[1:, :-1, :] + var_rho[:-1, :-1, :] + var_rho[:-1, 1:, :])

    return var_psi


#######################################################
# x-derivative from rho-grid to u-grid
#######################################################
def diffx(var, pm, dn=1):
    if rank(var) < 3:
        dvardx = diffx_2d(var, pm, dn)
    else:
        dvardx = diffx_3d(var, pm, dn)

    return dvardx


###########################
def diffx_3d(var, pm, dn=1):
    [N, M, L] = var.shape

    dvardx = Zeros((N - dn, M, L))

    for iz in range(0, L):
        dvardx[:, :, iz] = diffx_2d(var[:, :, iz], pm, dn)

    return dvardx


###########################
def diffx_2d(var, pm, dn=1):
    if (rank(pm) == 2) and (var.shape[0] == pm.shape[0]):
        dvardx = (var[dn:, :] - var[:-dn, :]) * 0.5 * (pm[dn:, :] + pm[:-dn, :]) / dn
    else:
        dvardx = (var[dn:, :] - var[:-dn, :]) * pm / dn

    return dvardx


###########################
def diffx_2d_time(var, pm, dn=1):
    if (var.shape[0] == pm.shape[0]):
        dvardx = (var[dn:, :, :] - var[:-dn, :, :]) * 0.5 * (pm[dn:, :, None] + pm[:-dn, :, None]) / dn
    else:
        dvardx = (var[dn:, :, :] - var[:-dn, :, :]) * pm[:, :, None] / dn

    return dvardx


###########################
def diffy_2d_time(var, pn, dn=1):
    if (var.shape[1] == pn.shape[1]):
        dvardy = (var[:, dn:, :] - var[:, :-dn, :]) * 0.5 * (pn[:, dn:, None] + pn[:, :-dn, None]) / dn
    else:
        dvardy = (var[:, dn:, :] - var[:, :-dn, :]) * pn[:, :, None] / dn

    return dvardy


#######################################################
# y-derivative from rho-grid to v-grid
#######################################################
def diffy(var, pn, dn=1):
    if rank(var) < 3:
        dvardy = diffy_2d(var, pn, dn)
    else:
        dvardy = diffy_3d(var, pn, dn)

    return dvardy


###########################
def diffy_3d(var, pn, dn=1):
    [N, M, L] = var.shape
    dvardy = Zeros((N, M - dn, L))
    for iz in range(0, L): dvardy[:, :, iz] = diffy_2d(var[:, :, iz], pn, dn)

    return dvardy


###########################
def diffy_2d(var, pn, dn=1):
    if (rank(pn) == 2) and (var.shape[1] == pn.shape[1]):
        dvardy = (var[:, dn:] - var[:, :-dn]) * 0.5 * (pn[:, dn:] + pn[:, :-dn]) / dn
    else:
        dvardy = (var[:, dn:] - var[:, :-dn]) * pn / dn

    return dvardy


#######################################################
# Compute horizontal derivatives on sigma-levels (1st order)
#######################################################
'''
var on rho-rho grid
dvardxi on psi-rho grid
'''
def diffxi(var,pm,z_r,z_w=None,newz=None,mask=None):
    """
    using the chain rule. (dvar/dx)|z = (dvar/dx)|sigma -(dvar/dz)(dz/dx)|sigma
    but not why solve for the upper layer? (the surface)
    Vicky says its because the upper layer will not have dz, and so it must be zero.z
    """
    dvardxi = Zeros((var.shape[0]-1,var.shape[1],var.shape[2]))
    dz_r = z_r[:,:,1:]-z_r[:,:,:-1]
    dz_w = z_w[:,:,1:]-z_w[:,:,:-1]

    if (var.shape[2]==z_w.shape[2]):  # should it be a comparison to dz_w??
        #.....var on psi-rho points to facilitate taking derivatives at w points......#
        # tmp= 0.5*(rho2u(var)[:,:,1:] + rho2u(var)[:,:,:-1])
        tmp = w2rho_s(rho2u(var),rho2u(z_r),rho2u(z_w))

        #.............(dvar/dx)|z = (dvar/dx)|sigma -(dvar/dz)(dz/dx)|sigma.........................#
        #.........dvar/dx|sigma...............#
        dvardxi[:,:,1:-1] = ((var[1:,:,1:-1] - var[:-1,:,1:-1]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        #........-(dvar/dz)*(dz/dx)|sigma
        dvardxi[:,:,1:-1] = dvardxi[:,:,1:-1] - (((z_w[1:,:,1:-1] - z_w[:-1,:,1:-1]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2u(dz_r))

        tmp2 = (rho2u(var)[:,:,1] - rho2u(var[:,:,0]))/rho2u(dz_w[:,:,0])
        dvardxi[:,:,0] = ((var[1:,:,0] - var[:-1,:,0]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        dvardxi[:,:,0] = dvardxi[:,:,0] - (tmp2 + (rho2u(z_w)[:,:,0] - rho2u(z_r)[:,:,0])*(dvardxi[:,:,1] - tmp2)/(rho2u(z_w[:,:,1]) - rho2u(z_r[:,:,0])))*(((z_w[1:,:,0] - z_w[:-1,:,0]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)

    elif (var.shape[2]==z_r.shape[2]):
        #.....var on w points to facilitate taking derivatives at w points......#
        # tmp= 0.5*(rho2u(var)[:,:,1:] + rho2u(var)[:,:,:-1])
        tmp = rho2w(rho2u(var),rho2u(z_r),rho2u(z_w))[:,:,1:-1]
        #.........................................................................#
        dvardxi[:,:,1:-1] = ((var[1:,:,1:-1] - var[:-1,:,1:-1]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        dvardxi[:,:,1:-1] = dvardxi[:,:,1:-1] - (((z_r[1:,:,1:-1] - z_r[:-1,:,1:-1]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2u(dz_w[:,:,1:-1]))
        tmp2 = (rho2u(var)[:,:,1] - rho2u(var[:,:,0]))/rho2u(dz_r[:,:,0])
        dvardxi[:,:,0] = ((var[1:,:,0] - var[:-1,:,0]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        dvardxi[:,:,0] = dvardxi[:,:,0] - (tmp2 + (rho2u(z_r)[:,:,0] - rho2u(z_w)[:,:,1])*(dvardxi[:,:,1] - tmp2)/(rho2u(z_r[:,:,1]) - rho2u(z_w[:,:,1])))*(((z_r[1:,:,0] - z_r[:-1,:,0]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)
    return dvardxi




#######################################################
# Compute horizontal derivatives on sigma-levels (1st order)
#######################################################
'''
var on rho-rho grid
dvardxi on psi-rho grid
'''
def diffeta(var,pn,z_r,z_w=None,newz=None,mask=None):
    dvardeta = Zeros((var.shape[0],var.shape[1]-1,var.shape[2]))
    dz_r = z_r[:,:,1:]-z_r[:,:,:-1]
    dz_w = z_w[:,:,1:]-z_w[:,:,:-1]

    if (var.shape[2]==z_w.shape[2]):
        #.....var on rho points to facilitate taking derivatives at w points......#
#        tmp= 0.5*(rho2v(var)[:,:,1:] + rho2v(var)[:,:,:-1])
        tmp = w2rho_s(rho2v(var),rho2v(z_r),rho2v(z_w))
        #.........................................................................#
        dvardeta[:,:,1:-1] = ((var[:,1:,1:-1] - var[:,:-1,1:-1]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T
        dvardeta[:,:,1:-1] = dvardeta[:,:,1:-1] - (((z_w[:,1:,1:-1] - z_w[:,:-1,1:-1]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2v(dz_r))
        tmp2 = (rho2v(var)[:,:,1] - rho2v(var[:,:,0]))/rho2v(dz_w[:,:,0])
        dvardeta[:,:,0] = ((var[:,1:,0] - var[:,:-1,0]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T
        dvardeta[:,:,0] = dvardeta[:,:,0] - (tmp2 + (rho2v(z_w)[:,:,0] - rho2v(z_r)[:,:,0])*(dvardeta[:,:,1] - tmp2)/(rho2v(z_w[:,:,1]) - rho2v(z_r[:,:,0])))*(((z_w[:,1:,0] - z_w[:,:-1,0]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T)

    elif (var.shape[2]==z_r.shape[2]):
        #.....var on rho points to facilitate taking derivatives at w points......#
#        tmp= 0.5*(rho2v(var)[:,:,1:] + rho2v(var)[:,:,:-1])
        tmp = rho2w(rho2v(var),rho2v(z_r),rho2v(z_w))[:,:,1:-1]
        #.........................................................................#
        dvardeta[:,:,1:-1] = ((var[:,1:,1:-1] - var[:,:-1,1:-1]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T
        dvardeta[:,:,1:-1] = dvardeta[:,:,1:-1] - (((z_r[:,1:,1:-1] - z_r[:,:-1,1:-1]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2v(dz_w[:,:,1:-1]))
        tmp2 = (rho2v(var)[:,:,1] - rho2v(var[:,:,0]))/rho2v(dz_r[:,:,0])
        dvardeta[:,:,0] = ((var[:,1:,0] - var[:,:-1,0]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T
        dvardeta[:,:,0] = dvardeta[:,:,0] - (tmp2 + (rho2v(z_r)[:,:,0] - rho2v(z_w)[:,:,1])*(dvardeta[:,:,1] - tmp2)/(rho2v(z_r[:,:,1]) - rho2v(z_w[:,:,1])))*(((z_r[:,1:,0] - z_r[:,:-1,0]).T*0.5*(pn[:,1:] + pn[:,:-1]).T).T)

    return dvardeta


#######################################################
# Transfert a field at rho points to u points
#######################################################
def rho2u(var_rho):
    if rank(var_rho) == 1:
        var_u = 0.5 * (var_rho[1:] + var_rho[:-1])
    elif rank(var_rho) == 2:
        var_u = rho2u_2d(var_rho)
    else:
        var_u = rho2u_3d(var_rho)

    return var_u


##############################
def rho2u_2d(var_rho):
    var_u = 0.5 * (var_rho[1:, :] + var_rho[:-1, :])

    return var_u


#############################
def rho2u_3d(var_rho):
    var_u = 0.5 * (var_rho[1:, :, :] + var_rho[:-1, :, :])

    return var_u


#######################################################
# Transfert a field at rho points to v points
#######################################################
def rho2v(var_rho):
    if rank(var_rho) == 1:
        var_v = 0.5 * (var_rho[1:] + var_rho[:-1])
    elif rank(var_rho) == 2:
        var_v = rho2v_2d(var_rho)
    else:
        var_v = rho2v_3d(var_rho)

    return var_v


##############################
def rho2v_2d(var_rho):
    var_v = 0.5 * (var_rho[:, 1:] + var_rho[:, :-1])

    return var_v


#############################
def rho2v_3d(var_rho):
    var_v = 0.5 * (var_rho[:, 1:, :] + var_rho[:, :-1, :])

    return var_v


############################
def rho2w(var_r, z_r, z_w):
    #print var_r.shape, z_w.shape
    var_w = z_w * 0
    var_w[:,:,0] = var_r[:,:,0] + (var_r[:,:,1] - var_r[:,:,0])*(z_w[:,:,0] - z_r[:,:,0])/(z_r[:,:,1] - z_r[:,:,0])
    var_w[:,:,1:-1] = var_r[:,:,:-1] * (z_r[:,:,1:] - z_w[:,:,1:-1]) + var_r[:,:,1:] * (z_w[:,:,1:-1] - z_r[:,:,:-1])
    var_w[:,:,1:-1] /= (z_r[:,:,1:] - z_r[:,:,:-1])


    var_w2 = z_w * 0
    # first value, using linear extrapolation
    slope=(var_r[:,:,1] - var_r[:,:,0])/(z_r[:,:,1] - z_r[:,:,0])
    var_w2[:,:,0] = var_r[:,:,0] + (z_w[:,:,0] - z_r[:,:,0])*slope

    # all the values between first and last values, using linear interpolation
    weights_up = z_w[:,:,1:-1] - z_r[:,:,:-1]
    weights_down = z_r[:,:,1:] - z_w[:,:,1:-1]
    weights_norm = weights_down + weights_up  # = z_r[:,:,1:] - z_r[:,:,:-1]
    var_w2[:,:,1:-1] = (var_r[:,:,1:] * weights_up + var_r[:,:,:-1] * weights_down)/weights_norm

    # last value, using linear extrapolation
    slope=(var_r[:,:,-1] - var_r[:,:,-2])/(z_r[:,:,-1] - z_r[:,:,-2])
    var_w2[:,:,-1] = var_r[:,:,-1] + (z_w[:,:,-1] - z_r[:,:,-1])*slope

    print(np.all(var_w2[:,:,:-1]==var_w[:,:,:-1]))
    return var_w


############################
def rho2w(var_r, z_r, z_w):
    #print var_r.shape, z_w.shape
    w_w = z_w * 0
    w_w[:,:,0] = var_r[:,:,0] + (var_r[:,:,1] - var_r[:,:,0])*(z_w[:,:,0] - z_r[:,:,0])/(z_r[:,:,1] - z_r[:,:,0])
    w_w[:,:,1:-1] = var_r[:,:,:-1] * (z_r[:,:,1:] - z_w[:,:,1:-1]) + var_r[:,:,1:] * (z_w[:,:,1:-1] - z_r[:,:,:-1])
    w_w[:,:,1:-1] /= (z_r[:,:,1:] - z_r[:,:,:-1])
    return w_w


def rho2w_michal(var_r, z_r, z_w):

    weights_down = (z_w[:,:,1:] - z_r[:,:,:])
    weights_up = (z_r[:,:,:] - z_w[:,:,:-1])
    weights_norm = weights_down + weights_up
    var_w2 = (var_r[:,:,1:] * weights_up + var_r[:,:,:-1] * weights_down)/weights_norm

    return var

#######################################################
# Transfert a field at u points to the rho points
#######################################################
def v2rho(var_v):
    if rank(var_v) == 2:
        var_rho = v2rho_2d(var_v)
    elif rank(var_v) == 3:
        var_rho = v2rho_3d(var_v)
    else:
        var_rho = v2rho_4d(var_v)

    return var_rho


#######################################################
def v2rho_2d(var_v):
    [Mp, L] = var_v.shape
    Lp = L + 1
    Lm = L - 1
    var_rho = Zeros((Mp, Lp))
    var_rho[:, 1:L] = 0.5 * (var_v[:, 0:Lm] + var_v[:, 1:L])
    var_rho[:, 0] = var_rho[:, 1]
    var_rho[:, Lp - 1] = var_rho[:, L - 1]
    return var_rho


#######################################################
def v2rho_3d(var_v):
    [Mp, L, N] = var_v.shape
    Lp = L + 1
    Lm = L - 1
    var_rho = Zeros((Mp, Lp, N))
    var_rho[:, 1:L, :] = 0.5 * (var_v[:, 0:Lm, :] + var_v[:, 1:L, :])
    var_rho[:, 0, :] = var_rho[:, 1, :]
    var_rho[:, Lp - 1, :] = var_rho[:, L - 1, :]
    return var_rho


#######################################################
def v2rho_4d(var_v):
    [Mp, L, N, Nt] = var_v.shape
    Lp = L + 1
    Lm = L - 1
    var_rho = Zeros((Mp, Lp, N, Nt))
    var_rho[:, 1:L, :, :] = 0.5 * (var_v[:, 0:Lm, :, :] + var_v[:, 1:L, :, :])
    var_rho[:, 0, :, :] = var_rho[:, 1, :, :]
    var_rho[:, Lp - 1, :, :] = var_rho[:, L - 1, :, :]
    return var_rho


#######################################################
# Transfert a 2 or 2-D field at u points to the rho points
#######################################################
def u2rho(var_u):
    if rank(var_u) == 2:
        var_rho = u2rho_2d(var_u)
    elif rank(var_u) == 3:
        var_rho = u2rho_3d(var_u)
    else:
        var_rho = u2rho_4d(var_u)
    return var_rho


#######################################################
def u2rho_2d(var_u):
    [M, Lp] = var_u.shape
    Mp = M + 1
    Mm = M - 1
    var_rho = Zeros((Mp, Lp))
    var_rho[1:M, :] = 0.5 * (var_u[0:Mm, :] + var_u[1:M, :])
    var_rho[0, :] = var_rho[1, :]
    var_rho[Mp - 1, :] = var_rho[M - 1, :]

    return var_rho


#######################################################
def u2rho_3d(var_u):
    [M, Lp, N] = var_u.shape
    Mp = M + 1
    Mm = M - 1
    var_rho = Zeros((Mp, Lp, N))
    var_rho[1:M, :, :] = 0.5 * (var_u[0:Mm, :] + var_u[1:M, :, :])
    var_rho[0, :, :] = var_rho[1, :, :]
    var_rho[Mp - 1, :, :] = var_rho[M - 1, :, :]

    return var_rho


#################################################################################
def u2rho_4d(var_u):
    [M, Lp, N, Nt] = var_u.shape
    Mp = M + 1
    Mm = M - 1
    var_rho = Zeros((Mp, Lp, N, Nt))
    var_rho[1:M, :, :, :] = 0.5 * (var_u[0:Mm, :, :, :] + var_u[1:M, :, :, :])
    var_rho[0, :, :, :] = var_rho[1, :, :, :]
    var_rho[Mp - 1, :, :, :] = var_rho[M - 1, :, :, :]

    return var_rho


#######################################################
# Vertical integration of a 3D variable between depth1 and depth2
#######################################################

def vert_integ_weights(z_w, depth1, depth2):

    Hz = z_w[:, :, 1:] - z_w[:, :, :-1]

    if depth1 != 9999:  # 9999 = surface as upper limit
        #  all depths above the upper limit are zero:
        Hz[z_w[:, :, :-1] >= depth1] = 0
        #  deal with the condition when upper limit is between layer in z_w,
        #  Hz will need to represent the part of the layer to be considered:
        indices = np.where(np.logical_and(z_w[:, :, :-1] < depth1, depth1 < z_w[:, :, 1:]))
        if type(depth1) == np.ndarray:
            Hz[indices] = depth1[indices]-z_w[:, :, :-1][indices]
        else:
            Hz[indices] = depth1-z_w[:, :, :-1][indices]


    Hz[z_w[:, :, 1:] <= depth2] = 0
    indices = np.where(np.logical_and(z_w[:, :, :-1] < depth2, depth2 < z_w[:, :, 1:]))
    z_w_ind = (indices[0], indices[1], indices[2]+1)
    if type(depth2)==np.ndarray:
        depth_ind = (indices[0], indices[1], indices[0]*0)
        Hz[indices] = z_w[z_w_ind]-depth2[depth_ind]
    else:
        Hz[indices] = z_w[:, :, 1:][indices] - depth2

    return Hz


#######################################################
# Netcdf Handling
#######################################################
from os.path import join as pjoin


def Forder(var):
    return np.asfortranarray(var.T, dtype=np.float32)
    # return var


#######################################################
def gridDict(dr, gridfile, ij=None):
    # Returns dictionary containing basic info from ROMS grid file
    gdict = {}

    if ij is not None:
        imin, imax, jmin, jmax = ij[0], ij[1], ij[2], ij[3]
    else:
        imin, imax, jmin, jmax = 0, None, 0, None

    nc = Dataset(pjoin(dr, gridfile), 'r')
    try:
        gdict['lon_rho'] = Forder(nc.variables['lon_rho'][jmin:jmax, imin:imax])
        gdict['lat_rho'] = Forder(nc.variables['lat_rho'][jmin:jmax, imin:imax])
    except:
        gdict['lon_rho'] = Forder(nc.variables['x_rho'][jmin:jmax, imin:imax])
        gdict['lat_rho'] = Forder(nc.variables['y_rho'][jmin:jmax, imin:imax])

    gdict['Ny'] = gdict['lon_rho'].shape[1]
    gdict['Nx'] = gdict['lon_rho'].shape[0]
    gdict['pm'] = Forder(nc.variables['pm'][jmin:jmax, imin:imax])
    gdict['pn'] = Forder(nc.variables['pn'][jmin:jmax, imin:imax])
    gdict['dx'] = 1.0 / np.average(gdict['pm'])
    gdict['dy'] = 1.0 / np.average(gdict['pn'])
    gdict['f'] = Forder(nc.variables['f'][jmin:jmax, imin:imax])
    try:
        gdict['h'] = Forder(nc.variables['h'][jmin:jmax, imin:imax])
    except:
        print('No h in grid file')
    try:
        gdict['rho0'] = nc.rho0
    except:
        print('No rho0 in grid file')
    try:
        gdict['mask_rho'] = Forder(nc.variables['mask_rho'][jmin:jmax, imin:imax])
    except:
        print('No mask in file')
        pass
    nc.close()
    print(gdict['Nx'], gdict['Ny'])
    return gdict


####################################################################################
def ncload(nc, var, itime=None, ij=None, func=None):
    if ij is not None:
        imin, imax, jmin, jmax = ij[0], ij[1], ij[2], ij[3]
    else:
        imin, imax, jmin, jmax = 0, None, 0, None

    xx = nc.variables[var].dimensions[-1]
    yy = nc.variables[var].dimensions[-2]

    if yy == 'eta_v':
        if jmax is not None:
            jmax = jmax - 1
        # jmax = -1 if jmax is None else jmax = jmax - 1
    if xx == 'xi_u':
        if imax is not None:
            imax = imax - 1
        # imax = -1 if imax is None else imax = imax - 1

    if itime is None:
        var1 = Forder(nc.variables[var][..., jmin:jmax, imin:imax])
    else:
        if isinstance(itime, list) or isinstance(itime, tuple):
            # print 'here'
            var1 = np.squeeze(Forder(nc.variables[var][itime[0]:itime[1], ..., jmin:jmax, imin:imax]))
            # print('inside time list before function', var1.shape)
        else:
            var1 = Forder(nc.variables[var][itime, ..., jmin:jmax, imin:imax])
    if func is None:
        return var1
    else:
        return func(var1)
    # return var1 if func is None else return func(var1)


####################################################################################
def ncload3d(nc, var, time=None, func=None, ijrho=None, grd='rho'):
    if time is not None:
        var1 = Forder(nc.variables[var][time, :, :, :])
    else:
        var1 = Forder(nc.variables[var][:, :, :])

    if func is None:
        return var1
    else:
        return func(var1)


####################################################################################
def ncload2d(nc, var, time=None, func=None):
    if time is not None:
        var1 = Forder(nc.variables[var][time, :, :])
    else:
        var1 = Forder(nc.variables[var][:, :])

    if func is None:
        return var1
    else:
        return func(var1)


####################################################################################
# Plotting
###################################################################################
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter, MultipleLocator, FormatStrFormatter
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt


def colorbar_tight(ax, im, fontsize, label, notation=0, sizeb=2, extend='both'):
    # colorbar_tight(ax, im, 16, '[mW m$^{-2}$]', tick_locs, tick_labels, pad = 0.3, sizeb = 2, extend='both')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=str(sizeb) + "%", pad=0.3)
    cbar = plt.colorbar(im, cax=cax)
    # cbar.formatter.set_powerlimits((0, 0))
    cbar.ax.tick_params(labelsize=fontsize)
    # cbar.ax.set_position(position)
    cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
    cbar.ax.yaxis.get_offset_text().set_position((4.0, 1.0))
    cbar.set_label(label, fontsize=fontsize, labelpad=10, rotation=270)
    # cbar.formatter = matplotlib.ticker.FixedFormatter(tick_labels)
    cbar.update_ticks()


###################################################################################
# NetCDF
###################################################################################
import os

####################################################################################
def wrtNcfile_z_lev(dr, outfile, vardict):
    try:
        nco = Dataset(os.path.join(dr, outfile), 'a')
    except:
        nco = Dataset(os.path.join(dr, outfile), 'w')
    nco.createDimension('depth', 8)
    nco.createDimension('time', None)
    for var, value in zip(list(vardict.keys()), list(vardict.values())):
        nco.createVariable(var, np.dtype('float32').char, ('time', 'depth'))
        nco.variables[var][:] = value
    return nco


####################################################################################
def wrtNcVars(outfile, vardict):
    nco = Dataset(outfile, 'a')
    for var, value in zip(list(vardict.keys()), list(vardict.values())):
        print(value.shape)
        if var not in nco.variables.keys():
            nco.createVariable(var, np.dtype('float32').char, ('time', 'depth', 'eta_rho', 'xi_rho'))
        nco.variables[var][:] = value
    nco.close()



####################################################################################
def wrtNcfile_2d(outfile, grddict):
    try:
        nco = Dataset(outfile, 'a')
    except:
        nco = Dataset(outfile, 'w')

    nco.createDimension('eta_rho', grddict['Ny'])
    nco.createDimension('xi_rho', grddict['Nx'])
    nco.createDimension('time', None)
    nco.close()


####################################################################################
def wrtNcVars_2d(outfile, vardict, dim_names=('time', 'eta_rho', 'xi_rho')):
    nco = Dataset(outfile, 'a')
    for var, value in zip(list(vardict.keys()), list(vardict.values())):
        nco.createVariable(var, np.dtype('float32').char, dim_names)
        nco.variables[var][:] = value
    nco.close()


####################################################################################
def nccopy(src, dst, ivarname, varlist, exclude = None):
    src=Dataset(src, 'r')
    dst=Dataset(dst, 'w')
    ncdimcopy(src,dst,exclude)
    ncvarcopy(src, dst, ivarname, varlist, exclude)
    src.close()
    dst.close()


####################################################################################
#make a copy with dimensions same as source but variables in varlist with same dims as var
####################################################################################
####################################################################################
def ncdimcopy(src, dst, exclude = None):
    dst.setncatts(src.__dict__)
    for name, dimension in src.dimensions.items():
        if exclude is not None and name in exclude:
            continue
        else:
            dst.createDimension(name, len(dimension) if not dimension.isunlimited() else None)


####################################################################################
def ncvarcopy(src, dst, exclude =  None):
    for name, value in src.variables.items():
        if exclude is not None and name in exclude:
            continue
        else:
            try:
                dims = value.dimensions
                dst.createVariable(name, value.datatype, dims)
                dst[name][:]=value
            except:
                print(name + ' Failed')

