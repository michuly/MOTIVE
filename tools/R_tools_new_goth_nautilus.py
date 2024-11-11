##################################################################################
# R_TOOLS
###################################################################################
"""





"""

###################################################################################
#Load modules
###################################################################################

#for numeric functions
import numpy as np
from numpy import float32
#for netcdf files
#from Scientific.IO.NetCDF import *
from netCDF4 import Dataset

#copy data
from copy import copy

#ROMSTOOLS
import R_tools_fort as toolsF
#import R_tools_fort_gula as toolsF_g

#Simulations (path, data...)
#import R_vars as va

#for plotting
import matplotlib.pyplot as py

import time as tm
###################################################################################


#import R_smooth as sm
import R_plot as r_plt

def Zeros(*args, **kwargs):
  kwargs.update(dtype=float32)
  return np.zeros(*args, **kwargs) 



def csf(sc, theta_s, theta_b):
  if (theta_s > 0):          
      csrf=(1-np.cosh(sc*theta_s))/(np.cosh(theta_s)-1)
  else:                  
      csrf=-sc**2                                 
  if (theta_b > 0):
      h = (np.exp(theta_b*csrf)-1)/(1-np.exp(-theta_b))
  else:                                             
      h  = csrf                                     
  return h


def scordinate(theta_s,theta_b,N,hc):
  sc_r=Zeros(N-1)
  Cs_r=Zeros(N-1)
  sc_w=Zeros(N)
  Cs_w=Zeros(N)
  ds=1.0/N
  sc_r= ds*(np.arange(N)-N+0.5)
  Cs_r=csf(sc_r, theta_s,theta_b)

  sc_w[0]=-1
  sc_w[N]=0
  Cs_w[0]=-1
  Cs_w[N]=0
  sc_w[1:N]=ds*(np.arange(1,N)-N)
  Cs_w=csf(sc_w, theta_s,theta_b)
  return (sc_r,Cs_r,sc_w,Cs_w)
#################################################
# get_depths (from setdepth.F in romsucla)
#################################################


def get_depths(simul,**kwargs):


    if 'coord' in  kwargs: 
        coord = kwargs['coord']
        [ny1i,ny2i,nx1i,nx2i] = coord[0:4]
        [ny1,ny2,nx1,nx2] = simul.coord[0:4]
        topo = np.asfortranarray(simul.topo[nx1i-nx1:nx2i-nx1,ny1i-ny1:ny2i-ny1])

    else: 
        coord = simul.coord
        topo = simul.topo
        

    if hasattr(simul, 'zeta'): 
        zeta=simul.zeta[nx1i-nx1:nx2i-nx1,ny1i-ny1:ny2i-ny1]
    else: 
        zeta = va.var('zeta',simul,depths=[0],coord=coord[0:4]).data

    (z_r,z_w) = toolsF.zlevs(topo, zeta, simul.hc, simul.Cs_r, simul.Cs_w)

    if 'sub' in  kwargs: 
        z_r = np.asfortranarray(z_r[:,:,simul.coord[4]-1])
        z_w = np.asfortranarray(z_w[:,:,np.arange(np.min(simul.coord[4])-1,np.max(simul.coord[4])+1)])
        
    return [z_r,z_w]



#################################################
# get_depth (from setdepth.F in romsucla)
#################################################


def get_depth(simul,**kwargs):


    if 'coord' in  kwargs: 
        coord = kwargs['coord']
        [ny1i,ny2i,nx1i,nx2i] = coord[0:4]
        [ny1,ny2,nx1,nx2] = simul.coord[0:4]
        topo = np.asfortranarray(simul.topo[nx1i-nx1:nx2i-nx1,ny1i-ny1:ny2i-ny1])

    else: 
        coord = simul.coord
        topo = simul.topo

    if hasattr(simul, 'zeta'): zeta=simul.zeta[nx1i-nx1:nx2i-nx1,ny1i-ny1:ny2i-ny1]
    else: zeta = va.var('zeta',simul,depths=[0],coord=coord[0:4]).data

    (z_r) = toolsF.zlev(topo, zeta, simul.hc, simul.Cs_r, simul.Cs_w)

    return z_r


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


#################################################
# rho_grad (from rho_eos.F and prsgrd.F in romsucla)
#################################################


def rho_grad(T,S,z_r,z_w,rho0,pm,pn):

    (drdz,drdx,drdy) = toolsF.rho_grad(T,S,z_r,z_w,rho0,pm,pn)

    return [drdz,drdx,drdy]


#######################################################
#interpolate a 3D variable on horizontal levels of constant depths (FORTRAN version, much faster)
#######################################################
def zlevs(gd, nch, itime = None, ij = None, nozeta = False):
    try:
        Cs_w = nch.Cs_w
        Cs_r = nch.Cs_r
    except:
        Cs_w = nch.variables['Cs_w'][:]
        Cs_r = nch.variables['Cs_r'][:]
    if itime is not None:
        if nozeta:
            zeta = ncload(nch, 'zeta', itime, ij = ij)*0
        else:       
            zeta = ncload(nch, 'zeta', itime, ij = ij)
        return toolsF.zlevs(gd['h'], zeta, nch.hc, Cs_r, Cs_w)
    else:
        return toolsF.zlevs(gd['h'], gd['h']*0, nch.hc, Cs_r, Cs_w)
#######################################################


def vinterp(var, depths, z_r, z_w=None, mask=None,imin=0,jmin=0,kmin=1, floattype=np.float32,interp_sfc=1,interp_bot=0,below=None,**kwargs):


    if mask==None:  mask = np.ones((z_r.shape[0],z_r.shape[1]), order='F', dtype=floattype); mask[z_r[:,:,-1]==0] = 0

    if z_w is None: 
        print('no z_w specified')
        z_w=Zeros((z_r.shape[0],z_r.shape[1],z_r.shape[2]+1), order='F')
        z_w[:,:,1:-1] = 0.5*(z_r[:,:,1:] + z_r[:,:,:-1])
        z_w[:,:,0] = z_r[:,:,0] - (z_r[:,:,1]-z_r[:,:,0])
        z_w[:,:,-1] = z_r[:,:,-1] + (z_r[:,:,-1]-z_r[:,:,-2])
        
    if np.rank(depths)==1: newz = np.asfortranarray(Zeros((z_r.shape[0],z_r.shape[1],len(depths))) + depths, dtype=floattype)
    else: newz = depths

    if interp_bot==1:
        print("data will be interpolated below ground")
        below=1000.
        vnew=toolsF.sigma_to_z_intr_bot(z_r, z_w,mask,var,newz,below,imin,jmin,kmin,9999.)
    elif interp_sfc==1:
        print("no interpolation below ground")
        print(z_r.shape, z_w.shape,mask.shape,var.shape,newz.shape)
        vnew=toolsF.sigma_to_z_intr_sfc(z_r, z_w,mask,var,newz,imin,jmin,kmin,9999.)
    else:
        print("no interpolation below ground")
        vnew=toolsF.sigma_to_z_intr(z_r, z_w,mask,var,newz,imin,jmin,kmin,9999.)    

    
    vnew[np.abs(vnew)==9999.]=np.nan

    return vnew










#######################################################
#Transfert a field at psi points to rho points
#######################################################

def psi2rho(var_psi):

    if np.rank(var_psi)<3:
        var_rho = psi2rho_2d(var_psi)
    else:
        var_rho = psi2rho_3d(var_psi)

    return var_rho


##############################

def psi2rho_2d(var_psi):

    [M,L]=var_psi.shape
    Mp=M+1
    Lp=L+1
    Mm=M-1
    Lm=L-1

    var_rho=Zeros((Mp,Lp))
    var_rho[1:M,1:L]=0.25*(var_psi[0:Mm,0:Lm]+var_psi[0:Mm,1:L]+var_psi[1:M,0:Lm]+var_psi[1:M,1:L])
    var_rho[0,:]=var_rho[1,:]
    var_rho[Mp-1,:]=var_rho[M-1,:]
    var_rho[:,0]=var_rho[:,1]
    var_rho[:,Lp-1]=var_rho[:,L-1]

    return var_rho

##############################

def psi2rho_2dp1(var_psi):

    [M,L,Nt]=var_psi.shape
    Mp=M+1
    Lp=L+1
    Mm=M-1
    Lm=L-1

    var_rho=Zeros((Mp,Lp,Nt))
    var_rho[1:M,1:L,:]=0.25*(var_psi[0:Mm,0:Lm,:]+var_psi[0:Mm,1:L,:]+var_psi[1:M,0:Lm,:]+var_psi[1:M,1:L,:])
    var_rho[0,:,:]=var_rho[1,:,:]
    var_rho[Mp-1,:,:]=var_rho[M-1,:,:]
    var_rho[:,0,:]=var_rho[:,1,:]
    var_rho[:,Lp-1,:]=var_rho[:,L-1,:]

    return var_rho

#############################


def psi2rho_3d(var_psi):

    [Mz,Lz,Nz]=var_psi.shape
    var_rho=Zeros((Mz+1,Lz+1,Nz))

    for iz in range(0, Nz, 1):    
        var_rho[:,:,iz]=psi2rho_2d(var_psi[:,:,iz])

    return var_rho



#######################################################
#Transfert a field at rho points to psi points
#######################################################

def rho2psi(var_rho):

    if np.rank(var_rho)<3:
        var_psi = rho2psi_2d(var_rho)
    else:
        var_psi = rho2psi_3d(var_rho)

    return var_psi


##############################

def rho2psi_2d(var_rho):

    var_psi = 0.25*(var_rho[1:,1:]+var_rho[1:,:-1]+var_rho[:-1,:-1]+var_rho[:-1,1:])

    return var_psi

#############################

def rho2psi_3d(var_rho):

    var_psi = 0.25*(var_rho[1:,1:,:]+var_rho[1:,:-1,:]+var_rho[:-1,:-1,:]+var_rho[:-1,1:,:])

    return var_psi





#######################################################
#Transfert a field at rho points to u points
#######################################################

def rho2u(var_rho):

    if np.rank(var_rho)==1:
        var_u = 0.5*(var_rho[1:]+var_rho[:-1])
    elif np.rank(var_rho)==2:       
        var_u = rho2u_2d(var_rho)
    else:
        var_u = rho2u_3d(var_rho)

    return var_u


##############################

def rho2u_2d(var_rho):

    var_u = 0.5*(var_rho[1:,:]+var_rho[:-1,:])

    return var_u

#############################

def rho2u_3d(var_rho):

    var_u = 0.5*(var_rho[1:,:,:]+var_rho[:-1,:,:])

    return var_u



#######################################################
#Transfert a field at rho points to v points
#######################################################

def rho2v(var_rho):

    if np.rank(var_rho)==1:
        var_v = 0.5*(var_rho[1:]+var_rho[:-1])
    elif np.rank(var_rho)==2:
        var_v = rho2v_2d(var_rho)
    else:
        var_v = rho2v_3d(var_rho)

    return var_v


##############################

def rho2v_2d(var_rho):

    var_v = 0.5*(var_rho[:,1:]+var_rho[:,:-1])

    return var_v

#############################

def rho2v_3d(var_rho):

    var_v = 0.5*(var_rho[:,1:,:]+var_rho[:,:-1,:])

    return var_v





#######################################################
#Transfert a field at u points to the rho points
#######################################################

def v2rho(var_v):


    if np.rank(var_v) == 2:
        var_rho = v2rho_2d(var_v)
    elif np.rank(var_v) == 3:
        var_rho = v2rho_3d(var_v)
    else:
        var_rho = v2rho_4d(var_v)

    return var_rho

#######################################################

def v2rho_2d(var_v):

    [Mp,L]=var_v.shape
    Lp=L+1
    Lm=L-1
    var_rho=Zeros((Mp,Lp))
    var_rho[:,1:L]=0.5*(var_v[:,0:Lm]+var_v[:,1:L])
    var_rho[:,0]=var_rho[:,1]
    var_rho[:,Lp-1]=var_rho[:,L-1]
    return var_rho

#######################################################

def v2rho_3d(var_v):

    [Mp,L,N]=var_v.shape
    Lp=L+1
    Lm=L-1
    var_rho=Zeros((Mp,Lp,N))
    var_rho[:,1:L,:]=0.5*(var_v[:,0:Lm,:]+var_v[:,1:L,:])
    var_rho[:,0,:]=var_rho[:,1,:]
    var_rho[:,Lp-1,:]=var_rho[:,L-1,:]
    return var_rho


#######################################################
#######################################################

def v2rho_4d(var_v):

    [Mp,L,N,Nt]=var_v.shape
    Lp=L+1
    Lm=L-1
    var_rho=Zeros((Mp,Lp,N,Nt))
    var_rho[:,1:L,:,:]=0.5*(var_v[:,0:Lm,:,:]+var_v[:,1:L,:,:])
    var_rho[:,0,:,:]=var_rho[:,1,:,:]
    var_rho[:,Lp-1,:,:]=var_rho[:,L-1,:,:]
    return var_rho


#######################################################
#######################################################

def v2rho_2dp1(var_v):

    [Mp,L,Nt]=var_v.shape
    Lp=L+1
    Lm=L-1
    var_rho=Zeros((Mp,Lp,Nt))
    var_rho[:,1:L,:]=0.5*(var_v[:,0:Lm,:]+var_v[:,1:L,:])
    var_rho[:,0,:]=var_rho[:,1,:]
    var_rho[:,Lp-1,:]=var_rho[:,L-1,:]
    return var_rho


#######################################################
#Transfert a 2 or 2-D field at u points to the rho points
#######################################################

def u2rho(var_u):

    if np.rank(var_u) == 2:
        var_rho = u2rho_2d(var_u)
    elif np.rank(var_u) == 3:
        var_rho = u2rho_3d(var_u)
    else:
        var_rho = u2rho_4d(var_u)   
    return var_rho

#######################################################

def u2rho_2d(var_u):

    [M,Lp]=var_u.shape
    Mp=M+1
    Mm=M-1
    var_rho=Zeros((Mp,Lp))
    var_rho[1:M,:]=0.5*(var_u[0:Mm,:]+var_u[1:M,:])
    var_rho[0,:]=var_rho[1,:]
    var_rho[Mp-1,:]=var_rho[M-1,:]

    return var_rho

#######################################################

def u2rho_3d(var_u):

    [M,Lp,N]=var_u.shape
    Mp=M+1
    Mm=M-1
    var_rho=Zeros((Mp,Lp,N))
    var_rho[1:M,:,:]=0.5*(var_u[0:Mm,:]+var_u[1:M,:,:])
    var_rho[0,:,:]=var_rho[1,:,:]
    var_rho[Mp-1,:,:]=var_rho[M-1,:,:]

    return var_rho

#################################################################################

def u2rho_4d(var_u):

    [M, Lp, N, Nt]=var_u.shape
    Mp = M+1
    Mm = M-1
    var_rho = Zeros((Mp, Lp, N, Nt))
    var_rho[1:M,:,:,:]=0.5*(var_u[0:Mm,:,:,:]+var_u[1:M,:,:,:])
    var_rho[0,:,:,:]=var_rho[1,:,:,:]
    var_rho[Mp-1,:,:,:]=var_rho[M-1,:,:,:]

    return var_rho
#######################################################
#################################################################################

def u2rho_2dp1(var_u):

    [M, Lp, Nt]=var_u.shape
    Mp = M+1
    Mm = M-1
    var_rho = Zeros((Mp, Lp, Nt))
    var_rho[1:M,:,:]=0.5*(var_u[0:Mm,:,:]+var_u[1:M,:,:])
    var_rho[0,:,:]=var_rho[1,:,:]
    var_rho[Mp-1,:,:]=var_rho[M-1,:,:]

    return var_rho
#######################################################


def w2rho_s(var_w, z_r, z_w):
    #print var_w.shape, z_r.shape
    w_r = z_r * 0
    w_r = var_w[:,:,:-1] * (z_w[:,:,1:] - z_r[:,:,:]) + var_w[:,:,1:] * (z_r[:,:,:] - z_w[:,:,:-1])
    w_r /= (z_w[:,:,1:] - z_w[:,:,:-1])
    return w_r


def rho2w(var_r, z_r, z_w):
    #print var_r.shape, z_w.shape
    w_w = z_w * 0
    w_w[:,:,0] = var_r[:,:,0] + (var_r[:,:,1] - var_r[:,:,0])*(z_w[:,:,0] - z_r[:,:,0])/(z_r[:,:,1] - z_r[:,:,0])
    w_w[:,:,1:-1] = var_r[:,:,:-1] * (z_r[:,:,1:] - z_w[:,:,1:-1]) + var_r[:,:,1:] * (z_w[:,:,1:-1] - z_r[:,:,:-1])
    w_w[:,:,1:-1] /= (z_r[:,:,1:] - z_r[:,:,:-1])
    return w_w

#######################################################
#Transfert a 3-D field from verical w points to vertical rho-points
#######################################################

def w2rho(var_w):


    [M,L,N]=var_w.shape
    
    var_rho = Zeros((M,L,N-1))
    
    for iz in range(1,N-2):
        var_rho[:,:,iz]  = 0.5625*(var_w[:,:,iz+1] + var_w[:,:,iz]) -0.0625*(var_w[:,:,iz+2] + var_w[:,:,iz-1])
    
    var_rho[:,:,0]  = -0.125*var_w[:,:,2] + 0.75*var_w[:,:,1] +0.375*var_w[:,:,0] 
    var_rho[:,:,N-2]  = -0.125*var_w[:,:,N-3] + 0.75*var_w[:,:,N-2] +0.375*var_w[:,:,N-1] 
    

    return var_rho




'''
####################################################################################################################################
#Load variables
###################################################################################


    def load(self,varname,ncfile,simul,**kwargs):

        [ny1,ny2,nx1,nx2,depths] = self.coord

        if 'coord' in  kwargs: [ny1,ny2,nx1,nx2] = kwargs['coord'][0:4]
        if 'depths' in  kwargs: depths = kwargs['depths']

        [imin,jmin,kmin] = self.dico.get(varname)[2]; depth = np.array(depths)-1
        if len(depth)==1: depth = depth[0]


        try:
            data = np.squeeze(simul.Forder(ncfile.variables[varname][simul.infiletime,depth,ny1:ny2-jmin,nx1:nx2-imin]))
        except:
            data = np.squeeze(simul.Forder(ncfile.variables[varname][simul.infiletime,ny1:ny2-jmin,nx1:nx2-imin]))


        return data
# get_diagsPV 
#################################################


def get_pv_sol1(T,S,u,v,z_r,z_w,rho0,pm,pn,f):

    #print toolsF.get_diagspv.__doc__

    (pv) = toolsF.get_diagspv_sol1(T,S,u,v,z_r,z_w,rho0,pm,pn,f)

    return pv




#################################################
# get_diagsPV 
#################################################


def get_pv_sol2(T,S,u,v,z_r,z_w,rho0,pm,pn,f):

    #print toolsF.get_diagspv.__doc__

    (pv) = toolsF.get_diagspv_sol2(T,S,u,v,z_r,z_w,rho0,pm,pn,f)
    pv[pv==-9999.] = np.nan

    return pv

'''

#################################################
# get_J1
#################################################


def get_j1_sol1(stflx,ssflx,u,v,z_r,z_w,rho0,pm,pn,hbls,f):

    (J1) = toolsF.get_j1_sol1(stflx,ssflx,u,v,z_r,z_w,rho0,pm,pn,hbls,f)

    return J1

#################################################
# get_J1
#################################################


def get_j1_sol2(stflx,ssflx,u,v,z_r,z_w,rho0,pm,pn,hbls,f):

    (J1) = toolsF.get_j1_sol2(stflx,ssflx,u,v,z_r,z_w,rho0,pm,pn,hbls,f)

    return J1

#################################################
# get_J2
#################################################


def get_j2_sol1(T,S,u,v,z_r,z_w,rho0,pm,pn,hbls):

    (J2) = toolsF.get_j2_sol1(T,S,u,v,z_r,z_w,rho0,pm,pn,hbls)

    return J2

#################################################


def get_j2_sol2(T,S,u,v,z_r,z_w,rho0,pm,pn,hbls):

    (J2) = toolsF.get_j2_sol2(T,S,u,v,z_r,z_w,rho0,pm,pn,hbls)

    return J2



#################################################
# get_Jbot
#################################################


def get_jbot_sol1(T,S,u,v,z_r,z_w,rho0,pm,pn,hbbls,rdrg):


    (Jbot) = toolsF.get_jbot_sol1(T,S,u,v,z_r,z_w,rho0,pm,pn,hbbls,rdrg)

    return Jbot


#################################################


def get_jbot_sol2(T,S,u,v,z_r,z_w,rho0,pm,pn,hbbls,rdrg):


    (Jbot) = toolsF.get_jbot_sol2(T,S,u,v,z_r,z_w,rho0,pm,pn,hbbls,rdrg)

    return Jbot

#################################################
# get bottom drag
#################################################


def get_bottom_drag(u,v,Hz,rdrg):

    (ubot,vbot) = toolsF.get_bot(u,v,Hz,rdrg)

    return [ubot,vbot]


'''

#################################################
# get bottom drag
#################################################


def get_bottom_drag(u,v,Hz,rdrg):

    (ubot,vbot) = toolsF.get_bot(u,v,Hz,rdrg)

    return [ubot,vbot]



#################################################
# get bottom pressure torque
#################################################


def get_bpt(T,S, z_r,z_w,rho0,pm,pn):


    (bpt) = toolsF.get_bpt(T,S, z_r,z_w,rho0,pm,pn)


    #joe = Zeros((T.shape[0]-1,T.shape[1]-1))
    #joe[1:-1,1:-1] = np.asfortranarray(bpt[1:-1,1:-1])

    return bpt


#################################################
# get planetary term from vorticity balance
#################################################


def get_vortplantot(u,v,H,pm,pn,f):

    (vrtp) = toolsF.get_vortplantot(u,v,H,pm,pn,f)

    return vrtp


#################################################
# get planetary term from vorticity balance
#################################################


def get_vortplanet(u,v,H,pm,pn,f):

    (vrtp) = toolsF.get_vortplanet(u,v,H,pm,pn,f)

    return vrtp

#################################################
# get planetary stretching term from vorticity balance
#################################################


def get_vortstretch(u,v,H,pm,pn,f):

    (vrts) = toolsF.get_vortstretch(u,v,H,pm,pn,f)

    return vrts

#################################################
# get NL advective term from vort. balance equ.
#################################################


#def get_vortadv(u,v, z_r,z_w,pm,pn):

    #NOT READY YET
    #(vrta) = toolsF.get_vortadv(u,v, z_r,z_w,pm,pn)


    #return vrta

'''













#################################################
# Rotationnel
#################################################

def rot(u,v,pm,pn):
    
    if u.shape == v.shape:
        u=rho2u(u); v=rho2v(v)
    
    (rot) = toolsF.get_rot(u,v,pm,pn)

    return rot




#################################################
# Gradient (amplitude)
#################################################


def grad(psi,pm,pn, coord = 'p'):

    grad = toolsF.get_grad(psi,pm,pn)
    if coord == 'r':
        grad = psi2rho(grad)
    return grad















#######################################################
#x-derivative from rho-grid to u-grid
#######################################################

def diffx(var,pm,dn=1):

    if np.rank(var)<3:
        dvardx = diffx_2d(var,pm,dn)
    else:
        dvardx = diffx_3d(var,pm,dn)

    return dvardx

###########################

def diffx_3d(var,pm,dn=1):

    [N,M,L]=var.shape

    dvardx = Zeros((N-dn,M,L))

    for iz in range(0, L):    
        dvardx[:,:,iz]=diffx_2d(var[:,:,iz],pm,dn)

    return dvardx

###########################

def diffx_2d(var,pm,dn=1):

    if (np.rank(pm)==2) and (var.shape[0]==pm.shape[0]): 
        dvardx = (var[dn:,:]-var[:-dn,:])*0.5*(pm[dn:,:]+pm[:-dn,:])/dn
    else: 
        dvardx = (var[dn:,:]-var[:-dn,:])*pm/dn

    return dvardx




#######################################################
#y-derivative from rho-grid to v-grid
#######################################################

def diffy(var,pn,dn=1):

    if np.rank(var)<3: dvardy = diffy_2d(var,pn,dn)
    else: dvardy = diffy_3d(var,pn,dn)

    return dvardy

    #######################

def diffy_3d(var,pn,dn=1):

    [N,M,L]=var.shape
    dvardy = Zeros((N,M-dn,L))
    for iz in range(0, L): dvardy[:,:,iz]=diffy_2d(var[:,:,iz],pn,dn)

    return dvardy

    #######################


def diffy_2d(var,pn,dn=1):

    if (np.rank(pn)==2) and (var.shape[1]==pn.shape[1]):
        dvardy = (var[:,dn:]-var[:,:-dn])*0.5*(pn[:,dn:]+pn[:,:-dn])/dn
    else: 
        dvardy = (var[:,dn:]-var[:,:-dn])*pn/dn

    return dvardy



    
    
    

#######################################################
#Compute horizontal derivatives on sigma-levels (1st order)
#######################################################
'''
var on rho-rho grid
dvardxi on psi-rho grid
'''

############################################################
#....Compute horizontal derivatives using the chain rule as below.....# 
#......(d/dx)_z = (d/dx)_sigma - [(dz/dx)_sigma]* [(d/dz)]
#......(d/dy)_z = (d/dy)_sigma - [(dz/dy)_sigma]* [(d/dz)]
#....z_r and z_w passed to the func. must have same shape in x-y as var......#
############################################################

def diffxi(var,pm,z_r,z_w=None,newz=None,mask=None):
    dvardxi = Zeros((var.shape[0]-1,var.shape[1],var.shape[2]))
    dz_r = z_r[:,:,1:]-z_r[:,:,:-1]
    dz_w = z_w[:,:,1:]-z_w[:,:,:-1]
    if (var.shape[2]==z_w.shape[2]):
        #.....var on psi-rho points to facilitate taking derivatives at w points......#
#        tmp= 0.5*(rho2u(var)[:,:,1:] + rho2u(var)[:,:,:-1])
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
#        tmp= 0.5*(rho2u(var)[:,:,1:] + rho2u(var)[:,:,:-1])
        tmp = rho2w(rho2u(var),rho2u(z_r),rho2u(z_w))[:,:,1:-1]
        #.........................................................................#
        dvardxi[:,:,1:-1] = ((var[1:,:,1:-1] - var[:-1,:,1:-1]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        dvardxi[:,:,1:-1] = dvardxi[:,:,1:-1] - (((z_r[1:,:,1:-1] - z_r[:-1,:,1:-1]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)*((tmp[:,:,1:] - tmp[:,:,:-1])/rho2u(dz_w[:,:,1:-1]))
        tmp2 = (rho2u(var)[:,:,1] - rho2u(var[:,:,0]))/rho2u(dz_r[:,:,0])
        dvardxi[:,:,0] = ((var[1:,:,0] - var[:-1,:,0]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T
        dvardxi[:,:,0] = dvardxi[:,:,0] - (tmp2 + (rho2u(z_r)[:,:,0] - rho2u(z_w)[:,:,1])*(dvardxi[:,:,1] - tmp2)/(rho2u(z_r[:,:,1]) - rho2u(z_w[:,:,1])))*(((z_r[1:,:,0] - z_r[:-1,:,0]).T*0.5*(pm[1:,:] + pm[:-1,:]).T).T)
    return dvardxi

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


def diffxi_old(var,pm,z_r,z_w=None,newz=None,mask=None):


    if z_r.shape[2]<=2:
        dvardxi = diffxi_2d(var,pm,z_r,z_w,newz,mask)
    else:
        dvardxi = diffxi_3d(var,pm,z_r,z_w,newz,mask)

    ##############################################

    return dvardxi

#######################################################
#######################################################


def diffxi_3d(var,pm,z_r,z_w=None,newz=None,mask=None):


    if newz==None: newz = 0.5*(z_r[1:,:,:] + z_r[:-1,:,:])
    else: newz = rho2u(newz)

    dvardxi = Zeros((var.shape[0]-1,var.shape[1],var.shape[2]))

    ##############################################

    varzp = vinterp(var[1:,:,:],newz,z_r[1:,:,:],z_w[1:,:,:],interp_bot=1)
    varzm = vinterp(var[:-1,:,:],newz,z_r[:-1,:,:],z_w[:-1,:,:],interp_bot=1)

    dvardxi = ((varzp - varzm ).T*0.5*(pm[1:,:]+pm[:-1,:]).T ).T

    ##############################################

    return dvardxi


#######################################################
#######################################################


def diffxi_2d(var,pm,z_r,z_w=None,newz=None,mask=None):

    dvardxi = Zeros((z_r.shape[0]-1,z_r.shape[1]))

    ##############################################

    if newz==None: newz = 0.5*(z_r[:-1,:,0] + z_r[1:,:,0])
    else: newz = rho2u(newz)

    dz0 = (z_r[1:,:,0]-newz)
    dz1 = (newz-z_r[1:,:,1])
    varzp = (dz1*var[1:,:,0] + dz0*var[1:,:,1])/(z_r[1:,:,0]-z_r[1:,:,1])

    dz0 = (z_r[:-1,:,0]-newz)
    dz1 = (newz-z_r[:-1,:,1])
    varzm = (dz1*var[:-1,:,0] + dz0*var[:-1,:,1])/(z_r[:-1,:,0]-z_r[:-1,:,1])

    dvardxi = (varzp - varzm )*0.5*(pm[1:,:]+pm[:-1,:])
    ##############################################

    return dvardxi





#######################################################
#Compute horizontal derivatives on sigma-levels (1st order)
#######################################################

'''
var on rho-rho grid
dvardxi on psi-rho grid
'''

def diffeta_old(var,pn,z_r,z_w=None,newz=None,mask=None):


    if z_r.shape[2]<=2:
        dvardeta = diffeta_2d(var,pn,z_r,z_w,newz,mask)
    else:
        dvardeta = diffeta_3d(var,pn,z_r,z_w,newz,mask)

    ##############################################

    return dvardeta


#######################################################
#######################################################


def diffeta_3d(var,pn,z_r,z_w=None,newz=None,mask=None):


    if newz==None: newz = 0.5*(z_r[:,:-1,:] + z_r[:,1:,:])
    else: newz = rho2v(newz)

    dvardeta = Zeros((var.shape[0],var.shape[1]-1,var.shape[2]))

    ##############################################

    varzp = vinterp(var[:,1:,:],newz,z_r[:,1:,:],z_w[:,1:,:],interp_bot=1)
    varzm = vinterp(var[:,:-1,:],newz,z_r[:,:-1,:],z_w[:,:-1,:],interp_bot=1)

    dvardeta = ((varzp - varzm).T*0.5*(pn[:,:-1]+pn[:,1:]).T).T

    ##############################################


    return dvardeta



#######################################################
#Compute horizontal derivatives on sigma-levels (1st order)
#######################################################



def diffeta_2d(var,pn,z_r,z_w=None,newz=None,mask=None):

    dvardeta = Zeros((z_r.shape[0],z_r.shape[1]-1))

    ##############################################

    if newz==None: newz = 0.5*(z_r[:,:-1,0] + z_r[:,1:,0])
    else: newz = rho2v(newz)

    dz0 = (z_r[:,1:,0]-newz)
    dz1 = (newz-z_r[:,1:,1])
    varzp = (dz1*var[:,1:,0] + dz0*var[:,1:,1])/(z_r[:,1:,0]-z_r[:,1:,1])

    dz0 = (z_r[:,:-1,0]-newz)
    dz1 = (newz-z_r[:,:-1,1])
    varzm = (dz1*var[:,:-1,0] + dz0*var[:,:-1,1])/(z_r[:,:-1,0]-z_r[:,:-1,1])

    dvardeta = (varzp - varzm )*0.5*(pn[:,:-1]+pn[:,1:])

    ##############################################


    return dvardeta

#######################################################
#Compute Jacobian on sigma-levels (1st order)
#######################################################


def jacob_sig(var1,var2,pm,pn,z_r,z_w=None,newz=None,mask=None):

    print('jacob ,var, var2', var1.shape, var2.shape)

    var = rho2v(diffxi(var1,pm,z_r,z_w,newz,mask)) * rho2u(diffeta(var2,pn,z_r,z_w,newz,mask))\
        - rho2v(diffxi(var2,pm,z_r,z_w,newz,mask)) * rho2u(diffeta(var1,pn,z_r,z_w,newz,mask))

    print('jacob ,final', var.shape)

    return var
    

 #######################################################
#Compute Jacobian on sigma-levels (1st order)
#######################################################


def jacob(var1,var2,pm,pn):

    var = rho2v(diffx(var1,pm)) * rho2u(diffy(var2,pn))\
        - rho2v(diffx(var2,pm)) * rho2u(diffy(var1,pn))

    return var   
    
    
    
   
#######################################################
#Compute mean
#######################################################



def nanmean(data,axis=None, *args):

    dataout = np.ma.filled(np.ma.masked_array(data,np.isnan(data)).mean(*args,axis=axis), fill_value=np.nan)
    if dataout.shape==(): dataout = float(dataout)

    return dataout



#######################################################
#Strain (amplitude only)
#######################################################


def strain_uvxy(ux, uy, vx, vy):
    
    s1 = ux-vy; s2= vx+uy;
    #Rotational part of strain
    s_rot=s1*s1+s2*s2
    s=2*(ux**2+vy**2)+s2*s2
    #Total strain, s^2-s_rot^2=delta^2
    #return (s,ux,uy,vx,vy)
    return (s,s_rot)

#######################################################
#Strain (amplitude only)
#######################################################


def strain(u,v,pm=1,pn=1):
    
    if u.shape==v.shape: 
        ux=u2rho( diffx(u,pm))
        vy=v2rho( diffy(v,pn))
        uy=v2rho( diffy(u,pn))
        vx=u2rho( diffx(v,pm))   
        
    else:    
        ux = Zeros(pm.shape)*np.nan
        vy,uy,vx = copy(ux),copy(ux),copy(ux)
        
        ux[1:-1,:]=diffx(u,rho2u(pm))
        vy[:,1:-1]=diffy(v,rho2v(pn))
        uy=psi2rho( diffy(u,rho2u(pn)))
        vx=psi2rho( diffx(v,rho2v(pm)))
    
    s1 = ux-vy; s2= vx+uy;
    #Rotational part of strain
    s_rot=s1*s1+s2*s2
    s=2*(ux**2+vy**2)+s2*s2
    #Total strain, s^2-s_rot^2=delta^2
    #return (s,ux,uy,vx,vy)
    return (s,s_rot)


    
#######################################################
#Strain (direction + amplitude)
#######################################################


def straindir(u,v,pm=1,pn=1):
    
    if u.shape[0]==v.shape[0]: 
        ux=u2rho( diffx(u,pm))
        vy=v2rho( diffy(v,pn))
        uy=v2rho( diffy(u,pn))
        vx=u2rho( diffx(v,pm))   
        
    else:    
        if pm==1:
            ux=u2rho( u2rho( diffx(u,pm)))
            vy=v2rho( v2rho( diffy(v,pn)))
            uy=psi2rho( diffy(u,pn))
            vx=psi2rho( diffx(v,pm))           
        else:
            ux = Zeros(pm.shape)*np.nan
            vy,uy,vx = copy(ux),copy(ux),copy(ux)
            
            ux[1:-1,:]=diffx(u,rho2u(pm))
            vy[:,1:-1]=diffy(v,rho2v(pn))
            uy=psi2rho( diffy(u,rho2u(pn)))
            vx=psi2rho( diffx(v,rho2v(pm)))
        
    s1 = ux-vy; s2= vx+uy;
    thetas = np.arctan(s2/s1)/2; thetaps = np.arctan(-1*s1/s2)/2;
    #see if division by 0
    eps = 1e-15; 
    thetas[np.abs(s1)<eps] = np.sign(s2[np.abs(s1)<eps])*np.pi/4
    #check if s1'>0 (s1<0 means that you are on the perpendicular axis)
    s1bis = s1 * np.cos(2*thetas) + s2*np.sin(2*thetas)
    thetas[s1bis<0] = thetas[s1bis<0]+np.pi/2
    s1bis = s1 * np.cos(2*thetas) + s2*np.sin(2*thetas)
    return thetas,s1bis
    
#######################################################
#2d divergence
#######################################################

def div(u,v,pm=1,pn=1):
    if u.shape[0]==v.shape[0]: 
        ux=u2rho( diffx(u,pm))
        vy=v2rho( diffy(v,pn))
        
    else:    
        ux = Zeros(pm.shape)*np.nan
        vy = copy(ux)
         
        ux[1:-1,:]=diffx(u,rho2u(pm))
        vy[:,1:-1]=diffy(v,rho2v(pn))
    s=(ux+vy)
    return s

#######################################################
#2d divergence on z coordinates, u and v on rho points
#######################################################
def div3d_z(u,v,pm,pn):
    div=Zeros(u2rho(u).shape)
    for iz in range(div.shape[2]):
        div[:,:,iz]=div2d(u[:,:,iz],v[:,:,iz],pm,pn)
    return div    
def grad3d_z(buoy,pm,pn):
    gradb=Zeros(buoy.shape)
    for iz in range(gradb.shape[2]):
        gradb[:,:,iz]=psi2rho(toolsF.get_grad(buoy[:,:,iz],pm,pn))
    return gradb
def div2d(u,v,pm=1,pn=1):
  #div1=pm*0
  #div1=(u[1:,1:-1]-u[0:-1,1:-1])*pm[1:-1,1:-1]+(v[1:-1,1:]-v[1:-1,0:-1])*pn[1:-1,1:-1]      
  
  dy_u=rho2u(1/pn);# compute dy in u points
  dx_v=rho2v(1/pm);
  duxi=Zeros(pm.shape);#always return fields with the same size as where they are located (in this case rho points)
  dveta=Zeros(pm.shape);
  duxi[1:-1,:]=pm[1:-1,:]*pn[1:-1,:]*(u[1:,:]*dy_u[1:,:]-u[0:-1,:]*dy_u[0:-1,:])
  dveta[:,1:-1]=pm[:,1:-1]*pn[:,1:-1]*(v[:,1:]*dx_v[:,1:]-v[:,0:-1]*dx_v[:,0:-1])
  div=duxi+dveta
  return div

#######################################################
#vertical velocity on z-points
#######################################################

def get_wvlcty_z(u,v,w0,pm,pn,depths):
    div1=u2rho(u)*0
    for iz in range(u.shape[2]):
        div1[:,:,iz]=div(u[:,:,iz],v[:,:,iz],pm,pn)     
        #div[:,:,iz]=div_2d(u[:,:,iz],v[:,:,iz],pm,pn)     
    #print w0.shape,div.shape,depths.size
    w=div1*0
    w[:,:,1:]=integrate.cumtrapz(-div1,depths)  
    #print w.shape
    return w  
#######################################################
# Vertical integration of a 3D variable only on a slope 
#######################################################

def vertIntSeamount(var, z_w, z_r, h, hbmax):

    #cff1 = np.max([depth1,depth2])
    #print z_r.shape, z_w.shape, var.shape
    hmax = np.amax(h)
    if var.shape[-1] == z_r.shape[-1]:
       Hz = z_w[:,:,1:] - z_w[:,:,:-1]
       cff = z_r + hbmax
       Hz[cff < 0] = 0
    elif var.shape[-1] == z_w.shape[-1]:
       Hz = z_w*0
       Hz[:,:,1:-1] = z_r[:,:,1:] - z_r[:,:,:-1]
       cff = z_w + hbmax
       #print cff.shape, Hz.shape, np.max(cff), np.min(cff)
       Hz[cff < 0] = 0
   
    try:
        varint = np.nansum(Hz * var,2)
        return varint
    except:
        pass
    try:
        varint = np.nansum(rho2u(Hz) * var,2)
        return varint
    except:
        pass
    try:
        varint = np.nansum(rho2v(Hz) * var,2)
        return varint
    except:
        pass
    try:
        varint = np.nansum(rho2psi(Hz) * var,2)
        return varint
    except:
        print(Hz.shape, var.shape)
        print('Failed...')
        sys.exit() 
#######################################################

#######################################################
#Geostrophic streamfunction 
#######################################################
def get_geoStreamfun_z(b,zeta,f,g,depths,pm,pn):
    psiG=b*0
    psiG[:,:,0]=g*zeta[:,:]
    psiG[:,:,1:]=g*zeta[:,:,None]+integrate.cumtrapz(b,depths)  
    #print w.shape
    psiG=psiG/np.mean(f)
    vg=rho2v(u2rho(diffx(psiG,pm)))
    ug=-rho2u(v2rho(diffy(psiG,pn)))
    
    #vrt on psi-rho grid
    
    v_x = psi2rho(diffx(vg,rho2v(pm)))
    u_y = psi2rho(diffy(ug,rho2u(pn)))
    vrt_g=v_x-u_y
    print(vg.shape, ug.shape)
    return (psiG,u2rho(ug),v2rho(vg),vrt_g)
#######################################################
#Geostrophic streamfunction from diags 
#######################################################

def get_geoStreamfun_diags(ncdiag, zeta, f,g,zr,zw,pm,pn):
    psiG=b*0
    psiG[:,:,0]=g*zeta[:,:]
    print(psiG.shape,zw.shape,zeta.shape) 
    psiG[:,:,1:]=g*zeta[:,:,None]+integrate.cumtrapz(b,zw, axis=2 )
    #print w.shape
    psiG=psiG/np.mean(f)
    vg=rho2v(u2rho(diffx(psiG,pm)))
    ug=-rho2u(v2rho(diffy(psiG,pn)))
    
    #vrt on psi-rho grid
    
    v_x = psi2rho(diffx(vg,rho2v(pm)))
    u_y = psi2rho(diffy(ug,rho2u(pn)))
    vrt_g=v_x-u_y
    print(vg.shape, ug.shape)
    return (psiG,u2rho(ug),v2rho(vg),vrt_g)

#######################################################
# Compute stress, tau from ROMS tau_z = (umix, vmix)
#######################################################
def tau(umix, vmix, taubx, tauby, z_r, z_w, coord='uv'):
    taux, tauy = Zeros(umix.shape), Zeros(vmix.shape)
    taux[:,:,0], tauy[:,:,0]  = taubx, tauby
    taux[:,:,1:]=taubx[:,:,None] + integrate.cumtrapz(umix,rho2u(z_r), axis=2 )
    tauy[:,:,1:]=tauby[:,:,None] + integrate.cumtrapz(vmix,rho2v(z_r), axis=2 )
    if coord == 'r':
        return u2rho(taux), v2rho(tauy) 
    return taux, tauy

#######################################################
#Geostrophic streamfunction 
#######################################################

def get_geoStreamfun(b,zeta,f,g,zr,zw,pm,pn):
    psiG=b*0
    psiG[:,:,-1]=g*zeta[:,:]
    print(psiG.shape,zw.shape,zeta.shape) 
    psiG[:,:,-2::-1]=g*zeta[:,:,None]+integrate.cumtrapz(b,zr, axis=2 )
    #print w.shape
    psiG=psiG/np.mean(f)
    vg=rho2v(u2rho(diffx(psiG,pm)))
    ug=-rho2u(v2rho(diffy(psiG,pn)))
    
    #vrt on psi-rho grid
    
    v_x = psi2rho(diffx(vg,rho2v(pm)))
    u_y = psi2rho(diffy(ug,rho2u(pn)))
    vrt_g=v_x-u_y
    print(vg.shape, ug.shape)
    return (psiG,u2rho(ug),v2rho(vg),vrt_g)
#######################################################
#Geostrophic streamfunction 
#######################################################
def surfGeoVel(zeta,f,pm,pn, fmt = 'rho' ):
    g=9.81
    if fmt == 'rho':
        vg=(g/f)*(u2rho(diffx(zeta,pm)))
        ug=(-g/f)*(v2rho(diffy(zeta,pn)))
    if fmt == 'uv':
        ug, vg = rho2u(ug), rho2v(vg)
    return ug, vg

#######################################################
#######################################################
def get_pressure(b,zeta,g,zr,pm,pn,rho0):
    p=b*0
    p[:,:,-1]=rho0*g*zeta[:,:]
    print(p.shape,zr.shape,zeta.shape) 
    print(zr[0,0,0],zr[0,0,-1])
    print(b[0,0,0],b[0,0,-1])

    p[:,:,-2::-1]=rho0*g*zeta[:,:,None]+rho0*integrate.cumtrapz(b,zr, axis=2 )
    #print w.shape
    return p 

#######################################################
#Pressure at depth
#######################################################
def get_pressure_z(b,depths,pm,pn,rho0=1025.0, zeta = None, g = 9.81):
    p=b*0
    if zeta is not None:
        p[:,:,0]=rho0*g*zeta[:,:]
        p[:,:,1:]=rho0*g*zeta[:,:,None]+rho0*integrate.cumtrapz(b,depths)  
    else:
        p[:,:,0] = rho0 * b[:,:,0] * depths[0] 
        p[:,:,1:] = np.expand_dims(p[:,:,0], axis =-1) + rho0*integrate.cumtrapz(b, depths)  
        #print w.shape
    return p

#######################################################
# Shear on constant z surface 
#######################################################
def shear2d(u,v,pm=1,pn=1):
    
    if u.shape[0]==v.shape[0]: 
        ux=u2rho( diffx(u,pm))
        vy=v2rho( diffy(v,pn))
        uy=v2rho( diffy(u,pn))
        vx=u2rho( diffx(v,pm))   
        
    else:    
        ux = Zeros(pm.shape)
        vy,uy,vx = copy(ux),copy(ux),copy(ux)
        
        ux[1:-1,:]=diffx(u,rho2u(pm))
        vy[:,1:-1]=diffy(v,rho2v(pn))
        uy=psi2rho( diffy(u,rho2u(pn)))
        vx=psi2rho( diffx(v,rho2v(pm)))
        
    return (ux,uy,vx,vy) 
    
    
    
#######################################################
#Shear (direction + amplitude)
#######################################################


def sheardir(u,v,pm=1,pn=1):
    
    if u.shape[0]==v.shape[0]: 
        ux=u2rho( diffx(u,pm))
        vy=v2rho( diffy(v,pn))
        uy=v2rho( diffy(u,pn))
        vx=u2rho( diffx(v,pm))   
        
    else:    
        if pm==1:
            ux=u2rho( u2rho( diffx(u,pm)))
            vy=v2rho( v2rho( diffy(v,pn)))
            uy=psi2rho( diffy(u,pn))
            vx=psi2rho( diffx(v,pm))           
        else:   
            ux = Zeros(pm.shape)*np.nan
            vy,uy,vx = copy(ux),copy(ux),copy(ux)
            
            ux[1:-1,:]=diffx(u,rho2u(pm))
            vy[:,1:-1]=diffy(v,rho2v(pn))
            uy=psi2rho( diffy(u,rho2u(pn)))
            vx=psi2rho( diffx(v,rho2v(pm)))
        
    s1 = ux-vy; s2= vx+uy; div=ux+vy
    #thetas = np.arctan(s2/s1)/2; 
    thetas = np.arctan(-1*s1/s2)/2;
    #see if division by 0
    eps = 1e-15; 
    thetas[np.abs(s2)<eps] = 0.
    #check if s2'>0 (s2'>0 means that you are on the perpendicular axis)
    s2bis = -s1 * np.sin(2*thetas) + s2*np.cos(2*thetas)
    thetas[s2bis>0] = thetas[s2bis>0]+np.pi/2
    s2bis = -s1 * np.sin(2*thetas) + s2*np.cos(2*thetas)


    return thetas,s2bis
    
    
    
    


#######################################################
#Rotate winds or u,v to lat,lon coord -> result on psi grid
#######################################################



def rotuv(simul,u,v,**kwargs):


    if isinstance(simul, float):
        angle = simul 
    else:
        if 'coord' in  kwargs: 
            [ny1,ny2,nx1,nx2]= kwargs['coord']
        else:
            [ny1,ny2,nx1,nx2] = simul.coord[0:4]
        
        ncfile = Dataset(simul.ncname.grd, 'r', format='NETCDF3_CLASSIC')
        angle = rho2psi(simul.Forder( np.array(ncfile.variables['angle'][ny1:ny2,nx1:nx2]) ))

    if u.shape!=v.shape:
        u=rho2v(u)
        v=rho2u(v)

    'rotate vectors by geometric angle'
    urot = u*np.cos(angle) - v*np.sin(angle)
    vrot = u*np.sin(angle) + v*np.cos(angle)

    return [urot,vrot]


#######################################################
#Get distance from lon-lat grid
#######################################################

def lonlat_to_m(lon,lat):

    lon = lon*2*np.pi/360.
    lat = lat*2*np.pi/360.

    dx = np.arccos(np.sin(lat[1:,:])*np.sin(lat[:-1,:]) + np.cos(lat[1:,:])*np.cos(lat[:-1,:])*np.cos(lon[1:,:]-lon[:-1,:]))*6371000.

    dy = np.arccos(np.sin(lat[:,1:])*np.sin(lat[:,:-1]) + np.cos(lat[:,1:])*np.cos(lat[:,:-1])*np.cos(lon[:,1:]-lon[:,:-1]))*6371000.

    return dx,dy






#######################################################
#Put nan at boundaries
#######################################################

def nanbnd(var,nbp=1):


    if len(var.shape)==1:
        var[:nbp] = np.nan
        var[-nbp:] = np.nan       

    elif len(var.shape)==2:
        var[:nbp,:] = np.nan
        var[-nbp:,:] = np.nan       
        var[:,:nbp] = np.nan       
        var[:,-nbp:] = np.nan
        
    elif len(var.shape)==3:
        var[:nbp,:,:] = np.nan
        var[-nbp:,:,:] = np.nan       
        var[:,:nbp,:] = np.nan       
        var[:,-nbp:,:] = np.nan    
         
    elif len(var.shape)==4:
        var[:nbp,:,:,:] = np.nan
        var[-nbp:,:,:,:] = np.nan       
        var[:,:nbp,:,:] = np.nan       
        var[:,-nbp:,:,:] = np.nan           
        
        
    return var

    
    
    
    
    
    
    
    
    
#######################################################
#Put nan in a sub region
#######################################################
from matplotlib.path import Path

def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)

def inpolygon1(x, y, xsub, ysub):
    print('Computing mask...')
    shape = x.shape
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x,y)).T 
    tupverts = [(xsub[0,0], ysub[0,0]), (xsub[0,-1], ysub[0,-1]), (xsub[-1,-1], ysub[-1,-1]),(xsub[-1,0], ysub[-1,0])] 
    p = Path(tupverts) # make a polygon
    print('Done')
    return p.contains_points(points).reshape(shape)



def nansub(var,x,y,xsub,ysub):

    var_bool = inpolygon1(x,y,xsub,ysub)
    
    if isinstance(var,int) or isinstance(var,float): var[~var_bool] = np.nan
    else: var.data[~var_bool] = 0 
    
    return var.data 
    
    

#######################################################


def nansub_bool(x,y,xsub,ysub):

    #polygon=np.array([(xsub[0,0],ysub[0,0]),(xsub[0,-1],ysub[0,-1]),(xsub[-1,-1],ysub[-1,-1]),(xsub[-1,0],ysub[-1,0])]) 
    
    # get contour of the sub region
    polygon=np.array((np.hstack((xsub[0,:],xsub[:,-1],xsub[-1,::-1],xsub[::-1,0])),np.hstack((ysub[0,:],ysub[:,-1],ysub[-1,::-1],ysub[::-1,0])))).T  
    
    # get all points from the region
    points = np.array((x.ravel(),y.ravel())).T
    
    # test wether points are inside the sub region -> return a boolean array
    var_bool = nxutils.points_inside_poly(points, polygon).reshape(x.shape)
    
    return var_bool  
#################################################################################
#Compute correlation tendency 
def STtendency(u,v,w,T,S,depths,pm,pn,mask=None):
  Tx,Ty,Tz=gradients_z(T,depths,pm,pn,mask)
  Sx,Sy,Sz=gradients_z(S,depths,pm,pn,mask)
  vx = psi2rho(diffx(v,rho2v(pm)))
  uy = psi2rho(diffy(u,rho2u(pn)))
  vy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  vy[:,1:-1,:] = diffy(v ,rho2v(pn))
  ux = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  ux[1:-1,:,:] = diffx(u ,rho2u(pm))
  wx = u2rho(diffx(w,pm))
  wy = v2rho(diffy(w,pn))
  TermH1=-2*(ux*Sx*Tx+vy*Sy*Ty)
  TermH2=-(vx+uy)*(Sy*Tx+Sx*Ty)
  TermV=-(wx*(Sz*Tx+Sx*Tz)+wy*(Sz*Ty+Sy*Tz))
  corr=(Sx*Tx+Sy*Ty)
  return TermH1,TermH2,TermV,corr
#For z-coordinates
#################################################################################

    
#################################################################################
#Compute all shear components on a rho-rho grid of the horizontal velocity fild
#For z-coordinates
#################################################################################


def gradients_z(T,depths,pm,pn,mask=None):
    
  dz=depths[2:]-depths[:-2]
  #dz[dz==0] = np.nan
  dTdx = u2rho(diffx(T,pm))
  dTdy = v2rho(diffy(T,pn))
  #Alternatively
  #'dvdz on rho-w grid'
  dTdz=T*0  
  dTdz[:,:,0]=0
  dTdz[:,:,-1]=0
  dTdz[:,:,1:-1] = (T[:,:,2:]-T[:,:,:-2])/dz[None,None,:]
  
  return (dTdx,dTdy,dTdz)
   
#################################################################################


def gradients(T, z_r, z_w, pm, pn, grid = 'r', mask = None):
    
  #dTdx = u2rho(diffx(T,pm))
  #dTdy = v2rho(diffy(T,pn))
  #dTdz = d_dz(T, z_r, z_w, coord = 'r')
  
  return (u2rho(diffx(T,pm)),v2rho(diffy(T,pn)),d_dz(T, z_r, z_w, coord = 'r'))
 
   
#################################################################################
#Compute all shear components on a rho-rho grid of the horizontal velocity fild
#For z-coordinates
#################################################################################

def diff_z(v,depths,mask=None):
  dz=depths[2:]-depths[:-2]
  dvdz=v*0  
  dvdz[...,0]=0
  dvdz[...,-1]=0
  try:
        dvdz[...,1:-1] = (v[...,2:]-v[...,:-2])/dz[None,None,:]
  except:
        dvdz[...,1:-1] = (v[...,2:]-v[...,:-2])/dz[:]
  return dvdz

def shear_z(u,v,depths,pm,pn,mask=None):
    
  #dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz=depths[2:]-depths[:-2]
  #dz[dz==0] = np.nan
  dvdx = psi2rho(diffx(v,rho2v(pm)))
  dudy = psi2rho(diffy(u,rho2u(pn)))
  #Alternatively
  #dvdx=psi2rho(v2rho(v),pm,z_r,z_w,mask=mask)
  #dudy=psi2rho(u2rho(u),pn,z_r,z_w,mask=mask)
  #'dvdz on rho-w grid'
  dvdz=v*0  
  dvdz[:,:,0]=0
  dvdz[:,:,-1]=0
  dvdz[:,:,1:-1] = (v[:,:,2:]-v[:,:,:-2])/dz[None,None,:]
  dvdz = v2rho(dvdz)
  #'dvdz on rho-w grid'
  dudz=u*0
  dudz[:,:,0] = 0
  dudz[:,:,-1] = 0
  dudz[:,:,1:-1] = (u[:,:,2:]-u[:,:,:-2])/dz[None,None,:]
  dudz=u2rho(dudz)
  #dudx=psi2rho(diffxi(u,rho2u(pm),rho2u(z_r),rho2u(z_w),mask=mask))
  #dvdy=psi2rho(diffeta(v,rho2v(pn),rho2v(z_r),rho2v(z_w),mask=mask))
  
  dvdy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  dvdy[:,1:-1,:] = diffy(v ,rho2v(pn))
  dudx = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  dudx[1:-1,:,:] = diffx(u ,rho2u(pm))
  return (dudx,dudy,dudz,dvdx,dvdy,dvdz)
   
def strainVortUz_z(u,v,depths,pm,pn,mask=None):
    
  #dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz=depths[2:]-depths[:-2]
  #dz[dz==0] = np.nan
  dvdx = psi2rho(diffx(v,rho2v(pm)))
  dudy = psi2rho(diffy(u,rho2u(pn)))
  #Alternatively
  #dvdx=psi2rho(v2rho(v),pm,z_r,z_w,mask=mask)
  #dudy=psi2rho(u2rho(u),pn,z_r,z_w,mask=mask)
  #'dvdz on rho-w grid'
  dvdz=v*0  
  dvdz[:,:,0]=0
  dvdz[:,:,-1]=0
  dvdz[:,:,1:-1] = (v[:,:,2:]-v[:,:,:-2])/dz[None,None,:]
  dvdz = v2rho(dvdz)
  #'dvdz on rho-w grid'
  dudz=u*0
  dudz[:,:,0] = 0
  dudz[:,:,-1] = 0
  dudz[:,:,1:-1] = (u[:,:,2:]-u[:,:,:-2])/dz[None,None,:]
  dudz=u2rho(dudz)
  #dudx=psi2rho(diffxi(u,rho2u(pm),rho2u(z_r),rho2u(z_w),mask=mask))
  #dvdy=psi2rho(diffeta(v,rho2v(pn),rho2v(z_r),rho2v(z_w),mask=mask))
  
  dvdy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  dvdy[:,1:-1,:] = diffy(v ,rho2v(pn))
  dudx = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  dudx[1:-1,:,:] = diffx(u ,rho2u(pm))
  strain, _ =  strain_uvxy(dudx, dudy, dvdx, dvdy) 
  return strain, dvdx - dudy, (dudz**2 + dvdz**2)**0.5, dudx + dvdy
   
def okubo_weiss_z(u,v,depths,pm,pn,f0,mask=None):
    
  #dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz=depths[2:]-depths[:-2]
  #dz[dz==0] = np.nan
  dvdx = psi2rho(diffx(v,rho2v(pm)))
  dudy = psi2rho(diffy(u,rho2u(pn)))
  dvdy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  dvdy[:,1:-1,:] = diffy(v ,rho2v(pn))
  dudx = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  dudx[1:-1,:,:] = diffx(u ,rho2u(pm))
  #Alternatively
  #dvdx=psi2rho(v2rho(v),pm,z_r,z_w,mask=mask)
  #dudy=psi2rho(u2rho(u),pn,z_r,z_w,mask=mask)
  #'dvdz on rho-w grid'
  s1 = dudx-dvdy; s2= dvdx+dudy;
  S2=s1*s1+s2*s2
  vrt=dvdx-dudy
  return(vrt/f0,(vrt*vrt-S2)/f0**2)
  #dudx=psi2rho(diffxi(u,rho2u(pm),rho2u(z_r),rho2u(z_w),mask=mask))
   
   
    
#################################################################################
#Compute vertical  shear components on a rho-rho grid of the horizontal velocity fild
#################################################################################


def vertshear(u, v, z_r, z_w, pm, pn, grid = 'r', mask = None, old = False):
    
  dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz_r[dz_r==0] = np.nan
  #'dvdz on rho-w grid'
  dvdz=z_w*0  
  dvdz[:,:,1:-1] = v2rho((v[:,:,1:]-v[:,:,:-1])/(0.5*(dz_r[:,1:,:]+ dz_r[:,:-1,:])))
  if not old:
    dvdz[:,:,0] = dvdz[:,:,1]  
    dvdz[:,:,-1] = dvdz[:,:,-2]    
  if grid == 'r':
      dvdz = w2rho_s(dvdz, z_r, z_w)
  #'dvdz on rho-w grid'
  dudz=z_w*0
  dudz[:,:,1:-1] = u2rho((u[:,:,1:]-u[:,:,:-1])/(0.5*(dz_r[1:,:,:]+ dz_r[:-1,:,:])))
  if not old:
    dudz[:,:,0] = dudz[:,:,1]   
    dudz[:,:,-1] = dudz[:,:,-2]    
  if grid == 'r':
    dudz=w2rho_s(dudz, z_r, z_w)
  
  return (dudz,dvdz)
#################################################################################
####################################################
def vertshear4d(u, v, z_r, z_w, pm, pn, grid = None, mask = None):
    
  dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz_r[dz_r==0] = np.nan
  #Alternatively
  #'dvdz on rho-w grid'
  Nx, Ny, Nz, Nt = v.shape 
  dvdz = Zeros((Nx, Ny, Nz+1, Nt))
  dvdz[:,:,0,:]=0
  dvdz[:,:,-1,:]=0
  dvdz[:,:,1:-1,:] = v2rho((v[:,:,1:,:]-v[:,:,:-1,:])/(0.5*(dz_r[:,1:,:, None]+ dz_r[:,:-1,:, None])))
  Nx, Ny, Nz, Nt = u.shape 
  dudz = Zeros((Nx, Ny, Nz+1, Nt))
  #'dvdz on rho-w grid'
  dudz[:,:,0,:] = 0
  dudz[:,:,-1,:] = 0
  dudz[:,:,1:-1,:] = u2rho((u[:,:,1:,:]-u[:,:,:-1,:])/(0.5*(dz_r[1:,:,:,None]+ dz_r[:-1,:,:,None])))
  #dudx=psi2rho(diffxi(u,rho2u(pm),rho2u(z_r),rho2u(z_w),mask=mask))
  #dvdy=psi2rho(diffeta(v,rho2v(pn),rho2v(z_r),rho2v(z_w),mask=mask))
  
  return (dudz,dvdz)
#################################################################################
#Compute  advective flux for rho-rho field
#################################################################################


def advFlux(q, u, v, w, z_r, z_w, pm, pn, mask=None):
  #Assumed that q is in rho-rho format  
  dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz_r[dz_r==0] = np.nan
  #'dvdz on rho-w grid'
  ddy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  ddx = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  ddx=u2rho(u*diffxi(q,pm,z_r,z_w,mask=mask))
  ddy=v2rho(v*diffeta(q,pn,z_r,z_w,mask=mask))
  ddz = q*d_dz(w, z_r, z_w, coord = 'r')
  #'dvdz on rho-w grid'
  return (ddx+ddy+ddz)
  
#################################################################################
#Compute divergent advective flux for rho-rho field
#################################################################################


def divFlux(q, u, v, w, z_r, z_w, pm, pn, simple = True, mask=None):
  #Assumed that q is in rho-rho format  
  dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz_r[dz_r==0] = np.nan
  #'dvdz on rho-w grid'
  if simple:
    ddy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
    ddy[:,1:-1,:] = diffy(v*rho2v(q) ,rho2v(pn))
    ddx = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
    ddx[1:-1,:,:] = diffx(u*rho2u(q) ,rho2u(pm))
  else:
    ddy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
    ddx = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
    ddx[1:-1,:,:]=(diffxi(u*rho2u(q),rho2u(pm),rho2u(z_r),rho2u(z_w),mask=mask))
    ddy[:,1:-1,:]=(diffeta(v*rho2v(q),rho2v(pn),rho2v(z_r),rho2v(z_w),mask=mask))
  ddz = d_dz(q*w, z_r, z_w, coord = 'r')
  #'dvdz on rho-w grid'
  
  return (ddx+ddy+ddz)
#################################################################################
#Compute all shear components on a rho-rho grid of the horizontal velocity fild
#################################################################################


def shear(u, v, z_r, z_w, pm, pn, mask=None, simple = False, coord = 'r'):
    
  dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz_r[dz_r==0] = np.nan
  vx = psi2rho(diffxi(v,rho2v(pm),rho2v(z_r),rho2v(z_w),mask=mask))
  uy = psi2rho(diffeta(u,rho2u(pn),rho2u(z_r),rho2u(z_w),mask=mask))
  #'vz on rho-w grid'
  vz=z_w*0  
  vz[:,:,1:-1] = v2rho((v[:,:,1:]-v[:,:,:-1])/(0.5*(dz_r[:,1:,:]+ dz_r[:,:-1,:])))
  vz[:,:,0] = vz[:,:,1]  
  vz[:,:,-1] = vz[:,:,-2]    
  #'vz on rho-w grid'
  uz=z_w*0
  uz[:,:,1:-1] = u2rho((u[:,:,1:]-u[:,:,:-1])/(0.5*(dz_r[1:,:,:]+ dz_r[:-1,:,:])))
  uz[:,:,0] = uz[:,:,1]   
  uz[:,:,-1] = uz[:,:,-2]    
  if coord=='r':
      uz=w2rho_s(uz, z_r, z_w)
      vz=w2rho_s(vz, z_r, z_w)
  vy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  ux = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  ux[1:-1,:,:]=(diffxi(u,rho2u(pm),rho2u(z_r),rho2u(z_w),mask=mask))
  vy[:,1:-1,:]=(diffeta(v,rho2v(pn),rho2v(z_r),rho2v(z_w),mask=mask))
  
  return (ux,uy,uz,vx,vy,vz)
#################################################################################
#Compute all shear components on a rho-rho grid of the horizontal velocity fild
#################################################################################


def horzshear(u,v,z_r,z_w,pm,pn, simple = False, mask=None):
    
  dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz_r[dz_r==0] = np.nan
  if simple:
      dvdx = psi2rho(diffx(v,rho2v(pm)))
      dudy = psi2rho(diffy(u,rho2u(pn)))
  else:         
      dvdx = psi2rho(diffxi(v,rho2v(pm),rho2v(z_r),rho2v(z_w),mask=mask))
      dudy = psi2rho(diffeta(u,rho2u(pn),rho2u(z_r),rho2u(z_w),mask=mask))
  #Alternatively
  #dvdx=psi2rho(v2rho(v),pm,z_r,z_w,mask=mask)
  #dudy=psi2rho(u2rho(u),pn,z_r,z_w,mask=mask)
  #'dvdz on rho-w grid'
  if simple:
    dvdy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
    dvdy[:,1:-1,:] = diffy(v ,rho2v(pn))
    dudx = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
    dudx[1:-1,:,:] = diffx(u ,rho2u(pm))
  else:
    dvdy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
    dudx = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
    dudx[1:-1,:,:]=(diffxi(u,rho2u(pm),rho2u(z_r),rho2u(z_w),mask=mask))
    dvdy[:,1:-1,:]=(diffeta(v,rho2v(pn),rho2v(z_r),rho2v(z_w),mask=mask))
  return (dudx,dudy,dvdx,dvdy)
  
#################################################################################
#Compute all shear components on a rho-rho grid of the horizontal velocity fild
#################################################################################


def vort(u,v,z_r,z_w,pm,pn, simple = False, mask=None, coord='p'):
    
  dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz_r[dz_r==0] = np.nan
  if simple:
      dvdx = psi2rho(diffx(v,rho2v(pm)))
      dudy = psi2rho(diffy(u,rho2u(pn)))
  else:         
      dvdx = (diffxi(v,rho2v(pm),rho2v(z_r),rho2v(z_w),mask=mask))
      dudy = (diffeta(u,rho2u(pn),rho2u(z_r),rho2u(z_w),mask=mask))
  if coord == 'p':
      return (dvdx-dudy)
  else:
      return psi2rho(dvdx-dudy)

def div3d(u,v,z_r,z_w,pm,pn,mask=None):
    
  dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
  dz_r[dz_r==0] = np.nan
  
  dvdy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  dudx = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
  dudx[1:-1,:,:]=(diffxi(u,rho2u(pm),rho2u(z_r),rho2u(z_w),mask=mask))
  dvdy[:,1:-1,:]=(diffeta(v,rho2v(pn),rho2v(z_r),rho2v(z_w),mask=mask))
  return (dudx + dvdy)


#######################################################
#Horintal gradients for rho variable
#######################################################

def iden(var):
  return var

def dxdy(T, z_r, z_w, pm, pn,mask=None, coord = 'r'):
   if coord=='r':
        funcx, funcy = u2rho, v2rho
   else:    
        funcx, funcy = iden, iden
   Tx = funcx(diffxi(T,pm,z_r,z_w,mask=mask)) 
   Ty = funcy(diffeta(T,pn,z_r,z_w,mask=mask))
   return Tx, Ty

def mixFlux(T,S,AKv,z_r,z_w,mask=None):
    
  dz_r=z_w*0
  dz_r[:,:,1:-1] = z_r[:,:,1:]- z_r[:,:,:-1]
  dTdz=AKv*0
  dTdz[:,:,0] = 0
  dTdz[:,:,-1] = 0
  dTdz[:,:,1:-1] = (T[:,:,1:]-T[:,:,:-1])/dz_r[:,:,1:-1]
  
  dSdz=AKv*0
  dSdz[:,:,0] = 0
  dSdz[:,:,-1] = 0
  dSdz[:,:,1:-1] = (S[:,:,1:]-S[:,:,:-1])/dz_r[:,:,1:-1]
  
  return AKv*dTdz,AKv*dSdz

#######################################################
#Compute \nu dT/dZ and \nu dS/dz at every sigma level
#######################################################

def d_dz(T,z_r,z_w, coord = 'w'):
    
  dz_r=z_w*0
  dz_r[:,:,1:-1] = z_r[:,:,1:]- z_r[:,:,:-1]
  dTdz=z_w*0
  dTdz[:,:,0] = 0
  dTdz[:,:,-1] = 0
  dTdz[:,:,1:-1] = (T[:,:,1:]-T[:,:,:-1])/dz_r[:,:,1:-1]
  
  if coord == 'r':
          return w2rho_s(dTdz, z_r, z_w)
  else:
          return dTdz


def d_dzw(Akv,z_r,z_w):
  print(z_w.shape, z_r.shape)  
  dz_w=z_r*0
  dz_w[:,:,:] = z_w[:,:,1:]- z_w[:,:,:-1]
  dAkvdz=z_r*0
  dAkvdz[:,:,:] = (Akv[:,:,1:]-Akv[:,:,:-1])/dz_w[:,:,:]
  
  return dAkvdz



#######################################################
#Compute Potential Vorticity of a 3-D field on psi-w grid
#######################################################
'''

Compute ertel potential vorticity using buoyancy (b=-g rho/rho0)

T and S on horizontal rho grids and vertical rho-grid (specified by z_r)
U and V on horizontal u- and v- grids and vertical rho-grid (specified by z_r)

PV is computed on horizontal psi-grid and vertical w-grid (specified by z_w)

'''
def PV(temp,salt,u,v,z_r,z_w,f,g,rho0,pm,pn,mask=None):

    #print 'we are using python version for PV'

    #rho on rho-rho grid      bvf on rho-w grid
    [dbdx,dbdy,dbdz] = toolsF.rho_grad(temp,salt,z_r,z_w,rho0,pm,pn)
    dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
    dz_r[dz_r==0] = np.nan
    pv=Zeros((z_w.shape[0]-1,z_w.shape[1]-1,z_w.shape[-1]))

##########################
#Ertel potential vorticity, term 1: [f + (dv/dx - du/dy)]*db/dz


    #dudy and dvdx on psi-rho grid
    dvdx = diffxi(v,rho2v(pm),rho2v(z_r),rho2v(z_w),mask=mask)
    dudy = diffeta(u,rho2u(pn),rho2u(z_r),rho2u(z_w),mask=mask)

    dbdz = rho2psi(dbdz)[:,:,1:-1]

    #vrt on psi-rho grid
    vrt = dvdx - dudy

    #PV1 on psi-w grid
    pv[:,:,1:-1] =  ((rho2psi(f).T + 0.5*(vrt[:,:,1:] + vrt[:,:,:-1]).T).T * dbdz)
    del vrt,dbdz

##########################
#'Ertel potential vorticity, term 2: (dv/dz)*(db/dx)'

    #'dvdz on psi-w grid'
    dvdz = rho2u((v[:,:,1:]-v[:,:,:-1])/(0.5*(dz_r[:,1:,:]+ dz_r[:,:-1,:])))

    #'dbdx on psi-rho grid'
    dbdx = rho2v(dbdx)

    #PV1 on psi-w grid
    pv[:,:,1:-1] = pv[:,:,1:-1] -1*dvdz*0.5*(dbdx[:,:,1:] + dbdx[:,:,:-1])
    del dbdx,dvdz

##########################
#'Ertel potential vorticity, term 3: (du/dz)*(db/dy)'

    #'dudz on psi-w grid'
    dudz = rho2v((u[:,:,1:]-u[:,:,:-1])/(0.5*(dz_r[1:,:,:]+ dz_r[:-1,:,:])))

    #'dbdy on psi-rho grid'
    dbdy = rho2u(dbdy)

    #PV3 on psi-w grid
    pv[:,:,1:-1] = pv[:,:,1:-1] + dudz*0.5*(dbdy[:,:,1:] + dbdy[:,:,:-1])
    
    del dbdy,dudz

##########################

    return pv



#######################################################
#Compute Potential Vorticity of a 3-D field on psi-w grid
#######################################################
'''

Compute ertel potential vorticity using buoyancy (b=-g rho/rho0)

T and S on horizontal rho grids and vertical rho-grid (specified by z_r)
U and V on horizontal u- and v- grids and vertical rho-grid (specified by z_r)

PV is computed on horizontal psi-grid and vertical w-grid (specified by z_w)

'''

def PV_terms(temp,salt,u,v,z_r,z_w,f,g,rho0,pm,pn,mask=None):

    #print 'we are using python version for PV'

    #rho on rho-rho grid      bvf on rho-w grid
    [dbdx,dbdy,dbdz] = toolsF.rho_grad(temp,salt,z_r,z_w,rho0,pm,pn)
    dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
    dz_r[dz_r==0] = np.nan

    pv1=Zeros((z_w.shape[0]-1,z_w.shape[1]-1,z_w.shape[-1]))*np.nan
    pv2=Zeros((z_w.shape[0]-1,z_w.shape[1]-1,z_w.shape[-1]))*np.nan
    pv3=Zeros((z_w.shape[0]-1,z_w.shape[1]-1,z_w.shape[-1]))*np.nan

    
##########################
#Ertel potential vorticity, term 1: [f + (dv/dx - du/dy)]*db/dz


    #dudy and dvdx on psi-rho grid
    dvdx = diffxi(v,rho2v(pm),rho2v(z_r),rho2v(z_w),mask=mask)
    dudy = diffeta(u,rho2u(pn),rho2u(z_r),rho2u(z_w),mask=mask)

    dbdz = rho2psi(dbdz)[:,:,1:-1]

    #vrt on psi-rho grid
    vrt = dvdx - dudy

    #PV1 on psi-w grid
    pv1[:,:,1:-1] =  ((rho2psi(f).T + 0.5*(vrt[:,:,1:] + vrt[:,:,:-1]).T).T * dbdz)
    del vrt,dbdz

##########################
#'Ertel potential vorticity, term 2: (dv/dz)*(db/dx)'

    #'dvdz on psi-w grid'
    dvdz = rho2u((v[:,:,1:]-v[:,:,:-1])/(0.5*(dz_r[:,1:,:]+ dz_r[:,:-1,:])))

    #'dbdx on psi-rho grid'
    dbdx = rho2v(dbdx)

    #PV1 on psi-w grid
    pv2[:,:,1:-1] =  -1*dvdz*0.5*(dbdx[:,:,1:] + dbdx[:,:,:-1])
    del dbdx,dvdz

##########################
#'Ertel potential vorticity, term 3: (du/dz)*(db/dy)'

    #'dudz on psi-w grid'
    dudz = rho2v((u[:,:,1:]-u[:,:,:-1])/(0.5*(dz_r[1:,:,:]+ dz_r[:-1,:,:])))

    #'dbdy on psi-rho grid'
    dbdy = rho2u(dbdy)

    #PV3 on psi-w grid
    pv3[:,:,1:-1] =  dudz*0.5*(dbdy[:,:,1:] + dbdy[:,:,:-1])
    
    del dbdy,dudz

##########################

    return [pv1,pv2,pv3]
    
    
#######################################################
#Compute Potential Vorticity of a 3-D field on an equidistant z-grid 
#######################################################
    
def PV_terms_z(bx,by,bz,u,v,dz,f,pm,pn,mask=None):
    #print 'we are using python version for PV'

    pv1=bx*0
    pv2=bx*0
    pv3=bx*0
    
#Ertel potential vorticity, term 1: [f + (dv/dx - du/dy)]*db/dz

    #dudy and dvdx on psi-rho grid
    dvdx = u2rho(diffx(v2rho(v),pm))
    dvdy = v2rho(diffy(v2rho(v),pn))
    dudx = u2rho(diffx(u2rho(u),pm))
    dudy = v2rho(diffy(u2rho(u),pn))

    #vrt on psi-rho grid
    vrt = dvdx - dudy

    #PV1 on psi-w grid
    print(f.shape,bz.shape,vrt.shape)
    pv1 = (vrt+f[:,:,None])*bz

##########################
#'Ertel potential vorticity, term 2: (dv/dz)*(db/dx)'
    
    dvdz=bx*0
    dvdz[:,:,1:-1]=v2rho((v[:,:,:-2]-v[:,:,2:])/dz)
    pv2 =  -1*dvdz*bx
    del dvdz

##########################
#'Ertel potential vorticity, term 3: (du/dz)*(db/dy)'

    dudz=bx*0
    dudz[:,:,1:-1]=u2rho((u[:,:,:-2]-u[:,:,2:])/dz)
    pv2 =  pv2+dudz*by
    del dudz
    #'dudz on psi-w grid'

    #'dbdy on psi-rho grid'
    #PV3 on psi-w grid
    
    #return (pv1+pv2)
    return pv1,pv2,vrt
#######################################################
#Compute Potential Vorticity of a 3-D field on psi-w grid
#######################################################
'''

Compute ertel potential vorticity using buoyancy (b=-g rho/rho0)

T and S on horizontal rho grids and vertical rho-grid (specified by z_r)
U and V on horizontal u- and v- grids and vertical rho-grid (specified by z_r)

PV is computed on horizontal psi-grid and vertical w-grid (specified by z_w)

'''

def PV_terms_r(dbdx,dbdy,dbdz,u,v,z_r,z_w,f,pm,pn,mask=None):

    #print 'we are using python version for PV'

    #rho on rho-rho grid      bvf on rho-w grid
    dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
    dz_r[dz_r==0] = np.nan

    pv1=Zeros((z_w.shape[0]-1,z_w.shape[1]-1,z_w.shape[-1]))
    pv2=Zeros((z_w.shape[0]-1,z_w.shape[1]-1,z_w.shape[-1]))
    pv3=Zeros((z_w.shape[0]-1,z_w.shape[1]-1,z_w.shape[-1]))

    
##########################
#Ertel potential vorticity, term 1: [f + (dv/dx - du/dy)]*db/dz


    #dudy and dvdx on psi-rho grid
    dvdx = diffxi(v,rho2v(pm),rho2v(z_r),rho2v(z_w),mask=mask)
    dudy = diffeta(u,rho2u(pn),rho2u(z_r),rho2u(z_w),mask=mask)

    dbdz = rho2psi(dbdz)[:,:,1:-1]

    #vrt on psi-rho grid
    vrt = dvdx - dudy

    #PV1 on psi-w grid
    pv1[:,:,1:-1] =  ((rho2psi(f).T + 0.5*(vrt[:,:,1:] + vrt[:,:,:-1]).T).T * dbdz)
    del vrt

##########################
#'Ertel potential vorticity, term 2: (dv/dz)*(db/dx)'

    #'dvdz on psi-w grid'
    dvdz = rho2u((v[:,:,1:]-v[:,:,:-1])/(0.5*(dz_r[:,1:,:]+ dz_r[:,:-1,:])))

    #'dbdx on psi-rho grid'
    dbdx = rho2v(dbdx)

    #PV1 on psi-w grid
    pv2[:,:,1:-1] =  -1*dvdz*0.5*(dbdx[:,:,1:] + dbdx[:,:,:-1])
    del dvdz

##########################
#'Ertel potential vorticity, term 3: (du/dz)*(db/dy)'

    #'dudz on psi-w grid'
    dudz = rho2v((u[:,:,1:]-u[:,:,:-1])/(0.5*(dz_r[1:,:,:]+ dz_r[:-1,:,:])))

    #'dbdy on psi-rho grid'
    dbdy = rho2u(dbdy)

    #PV3 on psi-w grid
    pv3[:,:,1:-1] =  dudz*0.5*(dbdy[:,:,1:] + dbdy[:,:,:-1])
    
    del dudz

##########################

    return [pv1,pv2,pv3]
    
    
#######################################################
#Compute Potential Vorticity of a 3-D field on psi-w grid
#######################################################
'''

Compute ertel potential vorticity using buoyancy (b=-g rho/rho0)

T and S on horizontal rho grids and vertical rho-grid (specified by z_r)
U and V on horizontal u- and v- grids and vertical rho-grid (specified by z_r)

PV is computed on horizontal psi-grid and vertical w-grid (specified by z_w)

'''

def PV_r(dbdx,dbdy,dbdz,u,v,z_r,z_w,f,pm,pn,mask=None):

    #print 'we are using python version for PV'

    #rho on rho-rho grid      bvf on rho-w grid
    dz_r = z_r[:,:,1:]- z_r[:,:,:-1]
    dz_r[dz_r==0] = np.nan
    pv=Zeros((z_w.shape[0]-1,z_w.shape[1]-1,z_w.shape[-1]))

##########################
#Ertel potential vorticity, term 1: [f + (dv/dx - du/dy)]*db/dz


    #dudy and dvdx on psi-rho grid
    dvdx = diffxi(v,rho2v(pm),rho2v(z_r),rho2v(z_w),mask=mask)
    dudy = diffeta(u,rho2u(pn),rho2u(z_r),rho2u(z_w),mask=mask)

    dbdz = rho2psi(dbdz)[:,:,1:-1]

    #vrt on psi-rho grid
    vrt = dvdx - dudy

    #PV1 on psi-w grid
    pv[:,:,1:-1] =  ((rho2psi(f).T + 0.5*(vrt[:,:,1:] + vrt[:,:,:-1]).T).T * dbdz)
    del vrt

##########################
#'Ertel potential vorticity, term 2: (dv/dz)*(db/dx)'

    #'dvdz on psi-w grid'
    dvdz = rho2u((v[:,:,1:]-v[:,:,:-1])/(0.5*(dz_r[:,1:,:]+ dz_r[:,:-1,:])))

    #'dbdx on psi-rho grid'
    dbdx = rho2v(dbdx)

    #PV1 on psi-w grid
    pv[:,:,1:-1] = pv[:,:,1:-1] -1*dvdz*0.5*(dbdx[:,:,1:] + dbdx[:,:,:-1])

##########################
#'Ertel potential vorticity, term 3: (du/dz)*(db/dy)'

    #'dudz on psi-w grid'
    dudz = rho2v((u[:,:,1:]-u[:,:,:-1])/(0.5*(dz_r[1:,:,:]+ dz_r[:-1,:,:])))

    #'dbdy on psi-rho grid'
    dbdy = rho2u(dbdy)

    #PV3 on psi-w grid
    pv[:,:,1:-1] = pv[:,:,1:-1] + dudz*0.5*(dbdy[:,:,1:] + dbdy[:,:,:-1])
    
    del dudz

##########################

    return pv


def grad_h_z(topo,depth,pm,pn):
    #Sort the topography into one dimension skipping repeat values
    hiz,ii=np.unique(topo[:],return_index='True')
    grad_h=psi2rho(grad(topo,pm,pn))
    temp_gh=grad_h.flatten()[ii]
    gh_z=depth*0
    ndepth=depth.size
    for iz in range(0,ndepth):
      #Find the grad_h_z at the actual depth locations 
      gh_z[iz]=temp_gh[np.argmin(abs(-depth[iz]-hiz))]
    return gh_z 
#######################################################
#Compute absolute vorticity of a 3-D field on psi grid
#######################################################

def get_absvrt(u,v,z_r,z_w,f,pm,pn,mask=None):

##########################
#Absolute vorticity,  [f + (dv/dx - du/dy)]

    vrt = get_vrt(u,v,z_r,z_w,pm,pn,mask)
    
    var =  (rho2psi(f).T + vrt.T).T 
    
    return var

#######################################################
#Compute relative vorticity of a 3-D field on psi grid
#######################################################

def get_vrt(u,v,z_r,z_w,pm,pn,mask=None):

    if len(u.shape)==3:
        #dudy and dvdx on psi grid
        dvdx = diffxi(v,rho2v(pm),rho2v(z_r),rho2v(z_w),mask)
        dudy = diffeta(u,rho2u(pn),rho2u(z_r),rho2u(z_w),mask)       
    else:      
        dvdx = diffx(v,rho2v(pm))
        dudy = diffy(u,rho2u(pn))  
        
    #vrt on psi grid
    vrt = dvdx - dudy    
    
    return vrt


def makenan(x, val = np.nan):
    x[x==0] = val 
    x[x<-100000] = val 
    x[x>1000000] = val

def varDict(nc, varlist, itime, opt = None, funclist=None, val = np.nan, ij = None):
    vd = {}
    if funclist is None:
        funclist=[ident for var in varlist]
    print('Loading..')
    #print(varlist)     
    for var, func in zip(varlist, funclist):
        if isinstance(nc, list) or isinstance(nc, tuple):
            try:
              vd[var] = np.squeeze(ncload(nc[0], var, itime, func = func, ij = ij))
            except:
              vd[var] = np.squeeze(ncload(nc[1], var, itime, func = func, ij = ij))
        else:
            try:
                vd[var] = np.squeeze(ncload(nc, var, itime, func = func, ij = ij))
            except:
                print(var, ' not found. Find another way to load it')
                continue
        #print(var, vd[var].shape, vd[var].dtype)     
        makenan(vd[var], val) 
    return vd
                

#######################################################
# Compute offline diags for vmix and hdiss
#######################################################
# vd consists of u, v, T, S, Akv, sustr, svstr
def get_uv_vmix(nc, vd, gd, rdrg=0, rho0=1025.0, itime = None, ij = None):
    print('Inside python get_uv_vmix    ')
    z_r,z_w = zlevs(gd, nc, itime = itime, ij=ij)
    try:
        print(vd['omega'].shape)
    except:     
        omega = toolsF.get_omega (vd['u'],vd['v'],z_r,z_w,gd['pm'],gd['pn']) 
        print('omega:', omega.shape, omega.dtype)
    #(MHdiss, MVmix, MPres) = toolsF_g.get_uv_vmix (vd['u'],vd['v'],vd['temp'],vd['salt'],z_r,z_w,gd['pm'],gd['pn'],gd['f']\
    (MHdiss, MVmix) = toolsF_g.get_uv_vmix (vd['u'],vd['v'],vd['temp'],vd['salt'],z_r,z_w,gd['pm'],gd['pn'],gd['f']\
                                           ,nc.dt,gd['mask_rho'],rdrg,rho0\
                                                         ,omega,vd['AKv'],vd['sustr'],vd['svstr']) 
    #print (MHdiss.shape, MVmix.shape) 
    return map(np.squeeze, [MHdiss[...,0], MHdiss[...,1], MVmix[...,0], MVmix[...,1]])
    #return map(np.squeeze, [MHdiss[...,0], MHdiss[...,1], MVmix[...,0], MVmix[...,1]])

#######################################################
def get_uv_vmix_est(nc, vd, gd, rdrg=0, rho0=1020.0, itime = None, ij = None):
    print('Inside python get_uv_vmix    ')
    z_r,z_w = zlevs(gd, nc, itime = itime, ij=ij)
    omega = toolsF.get_omega (vd['u'],vd['v'],z_r,z_w,gd['pm'],gd['pn'])
    print('omega:', omega.shape, omega.dtype)
    varlistu = ['u', 'v', 'w', 'AKv', 'temp', 'salt', 'sustr', 'svstr']
    print ('omega', np.max(omega), np.min(omega), np.mean(omega) )
    print ('z_r', np.max(z_r), np.min(z_r), np.mean(z_r) )
    print ('z_w', np.max(z_w), np.min(z_w), np.mean(z_w) )
    for var in varlistu:
        print(var, np.max(vd[var]), np.min(vd[var]), np.mean(vd[var]))    
    (MHdiss, MVmix) = toolsF_g.get_uv_vmix (vd['u'],vd['v'],vd['temp'],vd['salt'],z_r,z_w,gd['pm'],gd['pn'],gd['f']\
                                           ,nc.dt,gd['mask_rho'],rdrg,nc.rho0\
                                                            ,omega,vd['AKv'],vd['sustr'],vd['svstr'])
    return map(np.squeeze, [MHdiss[...,0], MHdiss[...,1], MVmix[...,0], MVmix[...,1]])
#######################################################
# Compute offline diags
#######################################################
# vd consists of u, v, T, S, Akv, sustr, svstr
def get_uv_diags(nc, vd, gd, rdrg=0, rho0=1025.0):
    print('Inside python get_uv_diags    ')
    z_r,z_w = zlevs(gd, nc)
    try:
        print(vd['omega'].shape)
    except:     
        vd['omega'] = toolsF.get_omega (vd['u'],vd['v'],z_r,z_w,gd['pm'],gd['pn']) 
        print('omega:', vd['omega'].shape, vd['omega'].dtype)
    #(MHdiss, MVmix, MPres) = toolsF_g.get_uv_vmix (vd['u'],vd['v'],vd['temp'],vd['salt'],z_r,z_w,gd['pm'],gd['pn'],gd['f']\
    dt = 180
    (MXadv, MYadv, MVadv, MHdiss, MCor, MVmix, MPrsgrd) = toolsF_g.get_uv_evolution (vd['u'],vd['v'],vd['temp'],vd['salt'],z_r,z_w,gd['pm'],gd['pn'],gd['f']\
                                           ,dt,gd['mask_rho'],rdrg,rho0\
                                                         ,vd['omega'],vd['AKv'],vd['sustr'],vd['svstr']) 
    
    return  MXadv, MYadv, MVadv, MHdiss, MVmix, MCor, MPrsgrd, z_r, z_w
#######################################################
# Compute pressure from Roms fortran 
#######################################################
def get_pressure_direct(nc, vd, gd):
    print('Inside python get_pressure    ')
    z_r,z_w = zlevs(gd, nc)
    P = toolsF.get_pressure(vd['temp'], vd['salt'], z_r, z_w, nc.rho0, gd['pm'], gd['pn'])
    return P            


#######################################################
# Get winds 
#######################################################
def get_winds(nch, ncwnd, itime, ij = None):
    oceantime = int(np.array(nch.variables['ocean_time'][itime]))%(360*24*3600)
    oceanday=oceantime/(24*3600.)
    datewind1=int(np.floor(oceanday-0.5))%360
    datewind2=int(np.ceil(oceanday-0.5))%360
    if ij is not None:
            imin, imax, jmin, jmax = ij[0], ij[1], ij[2], ij[3]
    else:
            imin, imax, jmin, jmax = 0, None, 0, None

    if datewind1==datewind2:
        coef1=0.5
        coef2=0.5
    else:
        coef1=abs(oceanday-0.5 - np.ceil(oceanday-0.5))
        coef2=abs(oceanday-0.5 - np.floor(oceanday-0.5))
        
    uwind1=Forder( np.array(ncwnd.variables['sustr'][datewind1,:,:]) )
    vwind1=Forder( np.array(ncwnd.variables['svstr'][datewind1,:,:]) )

    uwind2=Forder( np.array(ncwnd.variables['sustr'][datewind2,:,:]) )
    vwind2=Forder( np.array(ncwnd.variables['svstr'][datewind2,:,:]) )

    uwind=coef1*uwind1+coef2*uwind2
    vwind=coef1*vwind1+coef2*vwind2
    return uwind, vwind
#######################################################
# Vertical integration of a 3D variable between depth1 and depth2
#######################################################

def vert_int(var,z_w,depth1,depth2):

    cff2 = np.min([depth1,depth2])
    cff1 = np.max([depth1,depth2])

    Hz = z_w[:,:,1:] - z_w[:,:,:-1]
    
    cff = z_w[:,:,:-1] - cff1
    Hz[cff>0] = 0.
    Hz[np.logical_and(cff<Hz,cff>0)] = cff[np.logical_and(cff<Hz,cff>0)]
    
    cff = z_w[:,:,1:] - cff2
    Hz[cff<0] = 0.
    Hz[np.logical_and(cff<Hz,cff>0)] = cff[np.logical_and(cff<Hz,cff>0)]
    
    varint = nansum(Hz * var,2)

    print(z_w[10,10,:])
    print(Hz[10,10,:])   
    
    
    return varint
    
#######################################################
# Compute solution of TTW equation on sigma levels
# 
#######################################################


import scipy.integrate as integrate
 
def cumtrapz(var,z,inv=False):
    
    varint = Zeros((var.shape[0],var.shape[1],var.shape[2]))
    
    if np.rank(z)==0:
    
        if inv:
            varint[:,:,1:] = integrate.cumtrapz(var*z,axis=2)
        else:
            varint[:,:,:-1] = integrate.cumtrapz(var[:,:,::-1]*z[:,:,::-1],axis=2)[:,:,::-1]
            
    elif np.rank(z)==1:
    
        for i in range(var.shape[0]):
            for j in range(var.shape[1]):
                var[i,j,:] = var[i,j,:]*z
                
        if inv:
            varint[:,:,1:] = integrate.cumtrapz(var,axis=2)
        else:
            varint[:,:,:-1] = integrate.cumtrapz(var[:,:,::-1],axis=2)[:,:,::-1]           
            
    elif np.rank(z)==3:
    
        if inv:
            varint[:,:,1:] = integrate.cumtrapz(var*z,axis=2)
        else:
            varint[:,:,:-1] = integrate.cumtrapz(var[:,:,::-1]*z[:,:,::-1],axis=2)[:,:,::-1]
            
    return varint
    
#######################################################
#######################################################
#Compute tendency given u,v,buoy
#######################################################

def get_tendency(u,v,buoy,pm,pn,T=None,S=None):
    
    if u.shape==v.shape:
        
        vx = u2rho(diffx(v ,pm))
        uy = v2rho(diffy(u ,pn))
        ux = u2rho(diffx(u ,pm))
        vy = v2rho(diffy(v ,pn))

    else:

        vx = psi2rho(diffx(v ,rho2v(pm)))
        uy = psi2rho(diffy(u ,rho2u(pn)))
        
        if len(u.shape)==3:
            vy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
            vy[:,1:-1,:] = diffy(v ,rho2v(pn))
            ux = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
            ux[1:-1,:,:] = diffx(u ,rho2u(pm))
            #print 'Here in tendency'
        else:
            vy = Zeros(pm.shape)*np.nan
            vy[:,1:-1] = diffy(v ,rho2v(pn))
            ux = Zeros(pm.shape)*np.nan
            ux[1:-1,:] = diffx(u ,rho2u(pm))
            
    ##############
    
    bx = u2rho(diffx(buoy,pm))
    by = v2rho(diffy(buoy,pn))
    if T is not None:
        Tx = u2rho(diffx(T,pm))
        Ty = v2rho(diffy(T,pn))
        Sx = u2rho(diffx(S,pm))
        Sy = v2rho(diffy(S,pn))
        tend_ts=+1*((ux*Sx*Tx+vy*Sy*Ty)+0.5*(vx+uy)*(Sy*Tx+Sx*Ty))/(Sx*Tx+Sy*Ty)
    
    tend = +1*(bx * ux * bx + by * uy * bx + bx * vx * by + by * vy * by)/(bx**2+by**2)
    tend_u = +1*(ux **3+vy**3+(ux+vy)*(uy**2+uy*vx+vx**2))/(ux**2+uy**2+vx**2+vy**2)
#    if opt>0:
#      return(bx,by,bx * ux * bx+by * vy * by,by * uy * bx + bx * vx * by)
#    else:
    if T is not None:
      return tend,tend_u,tend_ts
    else: 
      return tend,tend_u
     
#######################################################
#Compute tendency given u,v, rho, compute bx, by from rho_grad
#######################################################

def get_tendency_r(u,v,bx,by,pm,pn):
    
    if u.shape==v.shape:
        
        vx = u2rho(diffx(v ,pm))
        uy = v2rho(diffy(u ,pn))
        ux = u2rho(diffx(u ,pm))
        vy = v2rho(diffy(v ,pn))

    else:

        vx = psi2rho(diffx(v ,rho2v(pm)))
        uy = psi2rho(diffy(u ,rho2u(pn)))
        
        if len(u.shape)==3:
            vy = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
            vy[:,1:-1,:] = diffy(v ,rho2v(pn))
            ux = Zeros((pm.shape[0],pm.shape[1],u.shape[2]))
            ux[1:-1,:,:] = diffx(u ,rho2u(pm))
            #print 'Here in tendency'
        else:
            vy = Zeros(pm.shape)*np.nan
            vy[:,1:-1] = diffy(v ,rho2v(pn))
            ux = Zeros(pm.shape)*np.nan
            ux[1:-1,:] = diffx(u ,rho2u(pm))
            
    ##############
    
    tend = -1*(bx * ux * bx + by * uy * bx + bx * vx * by + by * vy * by)
    
    return tend
#######################################################
# Compute divergent part of the flow 
# by solving Poisson equation for velocity potential
#######################################################
'''

Note that we are using uniform grid spacing, which is fine for small
scale grids but not for very large grids such as Pacific.

We are using Dirichlet boundary conditions

'''
#from pyamg import *
#from pyamg.gallery import *
from scipy import *
from scipy.linalg import *  


def div2uvs(u,v,pm,pn):
    

    if len(u.shape)>2:
        
        udiv = Zeros(u.shape)*np.nan
        vdiv = Zeros(v.shape)*np.nan
        
        for iz in range(u.shape[2]):
            udiv[:,:,iz],vdiv[:,:,iz] = div2uv(u[:,:,iz],v[:,:,iz],pm,pn)
            
    else:
        
        udiv,vdiv = div2uv(u,v,pm,pn)

    return udiv,vdiv

##################


def div2uv(u,v,pm,pn):


    pm = np.ones(pm.shape)*np.mean(pm)
    pn = np.ones(pm.shape)*np.mean(pn)
  
  
    # compute div
    div = Zeros(pm.shape)
    div[1:-1,:] = div[1:-1,:] + diffx(u,rho2u(pm)) 
    div[:,1:-1] = div[:,1:-1] + diffy(v,rho2v(pn))
    div[isnan(div)] =0
    
    # solve poisson
    A = poisson(div.shape, format='csr')     # 2D Poisson problem 
    ml =ruge_stuben_solver(A)                # construct the multigrid hierarchy
    print(ml)                                 # print hierarchy information
    b = -1*div.flatten()*1/np.mean(pm)**2    # right hand side
    x = ml.solve(b, tol=1e-10)               # solve Ax=b to a tolerance of 1e-8
    print("residual norm is", norm(b - A*x))  # compute norm of residual vector
    
    udiv = diffx(x.reshape(div.shape),pm)
    vdiv = diffy(x.reshape(div.shape),pn)
    
    return udiv,vdiv
    

    
#######################################################
# Compute solution of omega equation
# 
#######################################################
'''


'''
#from pyamg import *
#from pyamg.gallery import *
from scipy import *
from scipy.linalg import *  
from scipy.sparse import *
from scipy.sparse.linalg import spsolve
    
def solve_omega(buoy,pm,pn,f,N2,depths,ur=None,vr=None,nh=1,forcing=0.,mixrotuv=True):
    
    
    #######################################################
    #Create forcing (Q vector divergence)
    #######################################################
    rotuv=True; 
    if ur==None: rotuv=False

    new = Zeros(buoy.shape)

    nz=len(depths); [nx,ny] = pm.shape; ndim = (nz-1)*ny*nx;
    print('number of points is ',nx,ny,nz)
    dz = depths[1:]-depths[:-1]

    #get gradients
    bx,by = copy(new),copy(new)
    bx[1:-1,:,:]= diffx(buoy,pm,2); by[:,1:-1,:] = diffy(buoy,pn,2)  
    
    #if periodicity in y (used for jet example)
    #by[:,0,:] = (buoy[:,1,:]-buoy[:,-1,:])*0.5*(pn[:,1]+pn[:,-1])/2
    #by[:,-1,:] = (buoy[:,0,:]-buoy[:,-2,:])*0.5*(pn[:,1]+pn[:,-1])/2   
    
    ux,uy,uz = copy(new),copy(new),copy(new)
    vx,vy,vz = copy(new),copy(new),copy(new)

    if rotuv:
        u = ur; v=vr
        ux[1:-1,:,:] = diffx(u,pm,2); uy[:,1:-1,:] = diffy(u,pn,2)   
        vx[1:-1,:,:] = diffx(v,pm,2); vy[:,1:-1,:] = diffy(v,pn,2)       
        
        if mixrotuv:
            print('using velocity and buoyancy field to define Q vector')
            uz = -(by.T/f.T).T; vz =  (bx.T/f.T).T;
        
        else:
            #using only non-divergent velocity field
            print('using only velocity field to define Q vector')
            z_depths = copy(uz[:,:,:-1]);
            for i in range(nx): 
                for j in range(ny):
                    z_depths[i,j,:] = 0.5*(depths[1:]+depths[:-1])
            uz = vinterp((u[:,:,1:] - u[:,:,:-1])/dz,depths,z_depths)
            vz = vinterp((v[:,:,1:] - v[:,:,:-1])/dz,depths,z_depths)
        
    else:
        #or thermal wind      
        print('using buoyancy gradient to define Q vector')
        uz = -(by.T/f.T).T; vz =  (bx.T/f.T).T;
        #u = np.cumsum(uz,2)*dz;
        #v = np.cumsum(vz,2)*dz;
        u = cumtrapz(uz,depths)
        v = cumtrapz(vz,depths)
        ux[1:-1,:,:] = diffx(u,pm,2); uy[:,1:-1,:] = diffy(u,pn,2)   
        vx[1:-1,:,:] = diffx(v,pm,2); vy[:,1:-1,:] = diffy(v,pn,2)
        
        #if periodicity in y (used for jet example)
        #uy[:,0,:] = (u[:,1,:]-u[:,-1,:])*0.5*(pn[:,1]+pn[:,-1])/2
        #uy[:,-1,:] = (u[:,0,:]-u[:,-2,:])*0.5*(pn[:,1]+pn[:,-1])/2      
        #vy[:,0,:] = (v[:,1,:]-v[:,-1,:])*0.5*(pn[:,1]+pn[:,-1])/2
        #vy[:,-1,:] = (v[:,0,:]-v[:,-2,:])*0.5*(pn[:,1]+pn[:,-1])/2
        
    #Components of Q vector = (Qx,Qy)
    Qx = 2*(f.T*(vx*uz + vy*vz).T).T;
    Qy =-2*(f.T*(ux*uz + uy*vz).T).T;

    Qxx, Qyy =  copy(new),copy(new)
    Qxx[1:-1,:,:]= diffx(Qx,pm,2); Qyy[:,1:-1,:] = diffy(Qy,pn,2)
    #if periodicity in y (used for jet example)    
    #Qyy[:,0,:] = (Qy[:,1,:]-Qy[:,-1,:])*0.5*(pn[:,1]+pn[:,-1])/2
    #Qyy[:,-1,:] = (Qy[:,0,:]-Qy[:,-2,:])*0.5*(pn[:,1]+pn[:,-1])/2
    
    print('N2.shape', N2.shape)
    divQ = (Qxx + Qyy)/N2
    
    # smooothing...
    if nh>1: 
        divQ = sm.moy_h(divQ,nh)
        f=  sm.moy_h(f,nh); 
        pm = sm.moy_h(pm,nh)/nh
        pn = sm.moy_h(pn,nh)/nh
        if len(N2.shape)==3: N2=sm.moy_h(N2,nh);
        new = Zeros(divQ.shape)
        [nx,ny] = pm.shape; ndim = (nz-1)*ny*nx;
        print('nh is', nh)
        print('number of points is now ',nx,ny,nz)


    #######################################################
    # reorder forcings from (i,j,k) to vector
    R = Zeros(ndim);
    for i in range(nx): 
        for j in range(ny): 
            for k in range(nz-1):
                idx = i*ny*(nz-1) + k*ny + j;
                R[idx] = divQ[i,j,k]

    R[np.isnan(R)]=0

    #######################################################
    #Create matrix A
    #######################################################    
    
    print('creating matrix A')
    A = omega_matrix(pm,pn,depths,f,N2)

    
    #######################################################
    #Solve matrix A
    #######################################################   

    A = A.tocsr()
    
    #print 'solving equation'
    
    tstart = tm.time() 

    
    # Method 1 
    ml =ruge_stuben_solver(A)                # construct the multigrid hierarchy
    print(ml)                                 # print hierarchy information
    X = ml.solve(R, tol=1e-8)               # solve Ax=b to a tolerance of 1e-8
    print("residual norm is", norm(R - A*X))  # c
    print('Using ruge_stuben_solver.........', tm.time()-tstart)

    
    # Method 2 
    #X = spsolve(A,R)
    #print "residual norm is", norm(R - A*X)  # c
    #print 'Using spsolve.........', tm.time()-tstart
    #tstart = tm.time()  
    
    
    #######################################################  
    # reorder results in (i,j,k)
    
    w = Zeros((nx,ny,nz))
    for i in range(nx): 
        for j in range(ny): 
            for k in range(nz-1):
                idx = i*ny*(nz-1) + k*ny + j; 
                w[i,j,k] = X[idx];
    
    
    return w
    
####################################################### 

######################################################  

def omega_matrix(pm,pn,depths,f,N2):    
    
    
    
    # elliptic equation matrix divided by N^2: (f/N)^2 d_zz + d_xx + d_yy
    dx =1/np.mean(pm)
    dy =1/np.mean(pn)   
    dz = depths[1]-depths[0]
    
    nz=len(depths)
    [nx,ny] = pm.shape

    ############################
    

    
    dx2i = 1./(dx*dx);
    dy2i = 1./(dy*dy);
    dz2i = 1./(dz*dz);


    ndim = (nz-1)*ny*nx;
    i_s  = ny*(nz-1);
    js  = 1;
    bjs = ny-1;
    ks  = ny;
    #A = Zeros((ndim,ndim));
    #A=csc_matrix((ndim,ndim))
    A=lil_matrix((ndim,ndim))
    
    ############################
    
    for i in range(nx): 
        print(str(i) +  ' of '+ str(nx))
        for j in range(ny): 
        
            for k in range(nz-1):
                
                if len(N2.shape)==3: 
                    f2N2 = f[i,j]**2/N2[i,j,k]
                else: 
                    f2N2 = f[i,j]**2/N2[k]
                    
                if len(pm.shape)==2: 
                    dx2i= pm[i,j]**2
                    dy2i= pn[i,j]**2
      
                
                idx = i*ny*(nz-1) + k*ny + j;
                diag = 0.;

                if j>0:
                    A[idx,idx-js] = dy2i;
                    diag = diag - dy2i;
                #else:
                    #A[idx,idx+bjs] = dy2i;
                    #diag = diag - dy2i;

                if k>0:
                    dz2m = 1./((depths[k]-depths[k-1])*0.5*(depths[k+1]- depths[k-1]))
                    A[idx,idx-ks] = f2N2*dz2m;
                    diag = diag - f2N2*dz2m;
                else:
                    dz2m = 1./((depths[1]-depths[0])**2)
                    diag = diag - f2N2*dz2m;

                if i>0:
                    A[idx,idx-i_s] = dx2i;
                    diag = diag - dx2i;

                if i<nx-1:
                    A[idx,idx+i_s] = dx2i;
                    diag = diag - dx2i;

                if k==0:
                    dz2p = 1./((depths[k+1]-depths[k])*(depths[k+1]- depths[k]))
                    A[idx,idx+ks] = f2N2*dz2p;
                    diag = diag - f2N2*dz2p;
                elif k<nz-2:                   
                    dz2p = 1./((depths[k+1]-depths[k])*0.5*(depths[k+1]- depths[k-1]))
                    A[idx,idx+ks] = f2N2*dz2p;
                    diag = diag - f2N2*dz2p;                   
                else:
                    dz2p = 1./((depths[k+1]-depths[k])*0.5*(depths[k+1]- depths[k-1]))
                    diag = diag - f2N2*dz2p;
                    

                if j<ny-1:
                    A[idx,idx+js] = dy2i;
                    diag = diag - dy2i;
                #else:
                    #A[idx,idx-bjs] = dy2i;
                    #diag = diag - dy2i;

                A[idx,idx] = diag;

                

    return A
    
    
    
'''   
#######################################################
# Compute solution of TTW equation
# 
#######################################################

#from pyamg import *
#from pyamg.gallery import *
from scipy import *
from scipy.linalg import *  
from scipy.sparse import *
from scipy.sparse.linalg import spsolve
import time as tm
    
def solve_ttw(bx,by,AKv0,sustr,svstr,f,pm,pn,depths,timing=False):
    

    if timing: tstart = tm.time() 
    #######################################################
    #Create forcing (bx,by)
    #######################################################
    new = Zeros(AKv0.shape)

    nz=len(depths); [nx,ny] = pm.shape; 
    print 'number of points is ',nx,ny,nz
    dz = depths[1]-depths[0]
    dz2 = dz**2
    ks = 2;
    
    #get gradients
    #bx,by = copy(new),copy(new)
    #bx[1:-1,:,:]= diffx(buoy,pm,2); by[:,1:-1,:] = diffy(buoy,pn,2)
    
    AKv = Zeros((AKv0.shape[0],AKv0.shape[1],AKv0.shape[2]+1))
    AKv[:,:,:-1] = AKv0[:]; AKv[:,:,-1] = AKv0[:,:,-2]; AKv[:,:,-2] = AKv0[:,:,-2]
    del AKv0
    
    # Solutions
    uz,vz = copy(new)*np.nan,copy(new)

    #######################################################
    # Create Matrix
    #######################################################

    ndim = 2*(nz+1);
    A = lil_matrix((ndim,ndim))
    R = Zeros(ndim);

    for i in range(nx): 
    
        if i%20==0: print 'solving equation:', round(100.* i/(nx-1)) , ' %'
        
        for j in range(ny): 
        
            A = lil_matrix((ndim,ndim))
            
            #idx = 0
            #A[idx,idx+1] =  f[i,j];
            #A[idx+1,idx] = -f[i,j];

            #for k in range(1,nz):
                #idx = 2*k;
                #A[idx,idx+ks] = AKv[i,j,k+1]/dz2;
                #A[idx,idx] =-AKv[i,j,k]/dz2 - AKv[i,j,k]/dz2;
                #A[idx,idx-ks] = AKv[i,j,k-1]/dz2;
                #A[idx,idx+1] = f[i,j];
        
                #A[idx+1,idx+1+ks] = AKv[i,j,k+1]/dz2;
                #A[idx+1,idx+1] =-AKv[i,j,k]/dz2 - AKv[i,j,k]/dz2;
                #A[idx+1,idx+1-ks] = AKv[i,j,k-1]/dz2;
                #A[idx+1,idx] =-f[i,j];
    
            #idx = 2*nz;
            #A[idx,idx] = AKv[i,j,-1];
            #A[idx+1,idx+1] = AKv[i,j,-1];

            #######################################################

            idx = 0
            A[idx+1,idx] =  f[i,j];
            A[idx,idx+1] = -f[i,j];

            for k in range(1,nz):
                idx = 2*k;
                A[idx+ks,idx] = AKv[i,j,k+1]/dz2;
                A[idx,idx] =-AKv[i,j,k]/dz2 - AKv[i,j,k]/dz2;
                A[idx-ks,idx] = AKv[i,j,k-1]/dz2;
                A[idx+1,idx] = f[i,j];
        
                A[idx+1+ks,idx+1] = AKv[i,j,k+1]/dz2;
                A[idx+1,idx+1] =-AKv[i,j,k]/dz2 - AKv[i,j,k]/dz2;
                A[idx+1-ks,idx+1] = AKv[i,j,k-1]/dz2;
                A[idx,idx+1] =-f[i,j];
    
            idx = 2*nz;
            A[idx+1,idx] = AKv[i,j,-1];
            A[idx,idx+1] = AKv[i,j,-1];
            A = A*-1
            
            #######################################################

            for k in range(nz):   
                idx = 2*k
                R[idx] = bx[i,j,k]
                R[idx+1] = by[i,j,k]
                
   
            # add winds (for Ekman effect)
            #idx = 2*nz-4    
            #R[idx] = R[idx]-sustr[i,j]
            #R[idx+1] = R[idx+1]-svstr[i,j]            
            
            #idx = 2*nz-2    
            #R[idx] = R[idx]+2*sustr[i,j]
            #R[idx+1] = R[idx+1]+2*svstr[i,j]       
            
            #idx = 2*nz-2    
            #R[idx] = R[idx]-sustr[i,j]
            #R[idx+1] = R[idx+1]-svstr[i,j]       
                  
            idx = 2*nz    
            R[idx] = sustr[i,j]
            R[idx+1] = svstr[i,j]
            
            R[np.isnan(R)]=0
            if timing: print 'Matrix definition OK.........', tm.time()-tstart               
            if timing: tstart = tm.time()         

            #######################################################
            #Solve matrix A
            #######################################################   

            A = A.tocsr() 
            
            if timing: print 'Starting computation.........', tm.time()-tstart
            if timing: tstart = tm.time()   
  

            X = spsolve(A,R)
            del A
            
            if timing: print 'computation OK.........', tm.time()-tstart
            if timing: tstart = tm.time()         
            
            #######################################################
            
            # reorder results in (i,j,k)
            for k in range(nz):
                idx = 2*k
                uz[i,j,k] = X[idx];
                vz[i,j,k] = X[idx+1];
                
            if timing: print 'allocation OK.........', tm.time()-tstart               
            if timing: tstart = tm.time()  
            
    u = np.cumsum(uz,2)*dz
    v = np.cumsum(vz,2)*dz  
    
    return u,v
    
    
'''
    
  
    
#######################################################
# Compute solution of TTW equation
# 
#######################################################

#from pyamg import *
#from pyamg.gallery import *
from scipy import *
from scipy.linalg import *  
from scipy.sparse import *
from scipy.sparse.linalg import spsolve
import time as tm
import scipy.integrate as integrate
 
def solve_ttw(bx,by,AKv0,sustr,svstr,f,pm,pn,depths,timing=False):
    

    if timing: tstart = tm.time() 
    #######################################################
    #Create forcing (bx,by)
    #######################################################
    new = Zeros(AKv0.shape)

    nz=len(depths); [nx,ny] = pm.shape; 
    print('number of points is ',nx,ny,nz)
    dz = depths[1]-depths[0]
    dz2 = dz**2
    ks = 2;
    
    #get gradients
    #bx,by = copy(new),copy(new)
    #bx[1:-1,:,:]= diffx(buoy,pm,2); by[:,1:-1,:] = diffy(buoy,pn,2)
    
    AKv = Zeros((AKv0.shape[0],AKv0.shape[1],AKv0.shape[2]+1))
    #AKv[:,:,:-1] = AKv0[:]; AKv[:,:,-1] = AKv0[:,:,-2]; AKv[:,:,-2] = AKv0[:,:,-2]
    AKv[:,:,:-1] = AKv0[:]; AKv[:,:,-1] = AKv0[:,:,-3]; AKv[:,:,-2] = AKv0[:,:,-2]  
    #AKv[:,:,:-1] = AKv0[:,:,:]; AKv[:,:,-1] = AKv0[:,:,-1]; del AKv0  


    
    # Solutions
    uz,vz = copy(new)*np.nan,copy(new)

    #######################################################
    # Create Matrix
    #######################################################

    ndim = 2*(nz+1);
    #A = lil_matrix((ndim,ndim))
    #R = Zeros(ndim);

    for i in range(nx): 
    
        if i%20==0: print('solving equation:', round(100.* i/(nx-1)) , ' %')
        
        for j in range(ny): 
        
            A = lil_matrix((ndim,ndim))
            R = Zeros(ndim);
            
            idx = 0
            A[idx,idx+1] =  f[i,j];
            A[idx+1,idx] = -f[i,j];

            for k in range(1,nz):
                idx = 2*k;
                A[idx,idx+ks] = AKv[i,j,k+1]/dz2;
                A[idx,idx] =-AKv[i,j,k]/dz2 - AKv[i,j,k]/dz2;
                A[idx,idx-ks] = AKv[i,j,k-1]/dz2;
                A[idx,idx+1] = f[i,j];
        
                A[idx+1,idx+1+ks] = AKv[i,j,k+1]/dz2;
                A[idx+1,idx+1] =-AKv[i,j,k]/dz2 - AKv[i,j,k]/dz2;
                A[idx+1,idx+1-ks] = AKv[i,j,k-1]/dz2;
                A[idx+1,idx] =-f[i,j];
    
            idx = 2*nz;
            A[idx,idx+1] = AKv[i,j,-1];
            A[idx+1,idx] = AKv[i,j,-1];


            
            #######################################################

            for k in range(nz):   
                idx = 2*k
                R[idx] = bx[i,j,k]
                R[idx+1] = by[i,j,k]

            idx = 2*nz    
            R[idx] = svstr[i,j]
            R[idx+1] = sustr[i,j]

            
            if i==20 and j==80: 
                print(R)
                print(AKv[i,j,:])         
            if timing: print('Matrix definition OK.........', tm.time()-tstart)               
            if timing: tstart = tm.time()         

            #######################################################
            #Solve matrix A
            #######################################################   

            A = A.tocsr() 
            
            if timing: print('Starting computation.........', tm.time()-tstart)
            if timing: tstart = tm.time()   
  

            X = spsolve(A,R)
            del A
            
            if timing: print('computation OK.........', tm.time()-tstart)
            if timing: tstart = tm.time()         
            
            #######################################################
            
            # reorder results in (i,j,k)
            for k in range(nz):
                idx = 2*k
                uz[i,j,k] = X[idx];
                vz[i,j,k] = X[idx+1];
                
            if timing: print('allocation OK.........', tm.time()-tstart)               
            if timing: tstart = tm.time()  
            
    #u = np.cumsum(uz,2)*dz
    #v = np.cumsum(vz,2)*dz  
    
    u,v = copy(new), copy(new)
    print(uz.shape,len(depths))
    u[:,:,1:] = integrate.cumtrapz(uz,dx=dz, axis=2 )
    v[:,:,1:] = integrate.cumtrapz(vz,dx=dz, axis=2 )
    
    #ut[:,:,:-1] = integrate.cumtrapz(uz[:,:,::-1],z_w[:,:,::-1], axis=2 )[:,:,::-1]
    #vt[:,:,:-1] = integrate.cumtrapz(vz[:,:,::-1],z_w[:,:,::-1], axis=2 )[:,:,::-1]
    #ug[:,:,:-1] = (integrate.cumtrapz(-by[:,:,::-1],z_w[:,:,::-1], axis=2 )[:,:,::-1].T/f.T).T
    #vg[:,:,:-1] = (integrate.cumtrapz(bx[:,:,::-1],z_w[:,:,::-1], axis=2 )[:,:,::-1].T/f.T).T
    
    return u,v
    
      
 
    
#######################################################
# Compute solution of TTW equation on sigma levels
# 
#######################################################


from scipy.sparse import *
from scipy.sparse.linalg import spsolve
import scipy.integrate as integrate
import time as tm
 
def solve_ttw_sig(bx,by,AKv,sustr,svstr,f,pm,pn,z_w,timing=False,debug=0):
    
    '''
    AKv and by,by are on vertical w levels
    
    uz,vz (solutions of TTW) also

    '''

    
    if timing: tstart = tm.time() 
    #######################################################
    #Create forcing (bx,by)
    #######################################################
    new = Zeros(bx.shape)

    nz=AKv.shape[2]-1;  [nx,ny] = pm.shape; 
    print('number of points is ',nx,ny,nz)
    dz = z_w[:,:,1:] - z_w[:,:,:-1]
    ks = 2;

    # Solutions
    uz,vz = copy(new),copy(new)
    
    #######################################################
    # Create Matrix
    #######################################################

    ndim = 2*(nz+1);
    #A = lil_matrix((ndim,ndim))
    #R = Zeros(ndim);

    for i in range(nx): 
    
        if i%20==0: print('solving equation:', round(100.* i/(nx-1)) , ' %')
        
        for j in range(ny): 
        
            A = lil_matrix((ndim,ndim))
            R = Zeros(ndim);
            
            idx = 0
            A[idx,idx+1] =  f[i,j];
            A[idx+1,idx] = -f[i,j];

            for k in range(1,nz):
                idx = 2*k;
                dz2 = 0.5*(dz[i,j,k]+dz[i,j,k-1])
                
                A[idx,idx+ks] = AKv[i,j,k+1]/dz[i,j,k]/dz2;
                A[idx,idx] =-AKv[i,j,k]/dz[i,j,k]/dz2 - AKv[i,j,k]/dz[i,j,k-1]/dz2;
                A[idx,idx-ks] = AKv[i,j,k-1]/dz[i,j,k-1]/dz2;
                A[idx,idx+1] = f[i,j];
                
                A[idx+1,idx+1+ks] = AKv[i,j,k+1]/dz[i,j,k]/dz2;
                A[idx+1,idx+1] =-AKv[i,j,k]/dz[i,j,k]/dz2 - AKv[i,j,k]/dz[i,j,k-1]/dz2;
                A[idx+1,idx+1-ks] = AKv[i,j,k-1]/dz[i,j,k-1]/dz2;
                A[idx+1,idx] =-f[i,j];
    
            idx = 2*nz;
            A[idx,idx+1] = AKv[i,j,-1];
            A[idx+1,idx] = AKv[i,j,-1];
     
            #######################################################

            for k in range(nz):   
                idx = 2*k
                R[idx] = bx[i,j,k]
                R[idx+1] = by[i,j,k]

            idx = 2*nz    
            R[idx] = svstr[i,j]
            R[idx+1] = sustr[i,j]

                
            if timing: print('Matrix definition OK.........', tm.time()-tstart)               
            if timing: tstart = tm.time()         

            #######################################################
            #Solve matrix A
            #######################################################   

            A = A.tocsr() 
            
            if timing: print('Starting computation.........', tm.time()-tstart)
            if timing: tstart = tm.time()   
  

            X = spsolve(A,R)
            del A
            
            if timing: print('computation OK.........', tm.time()-tstart)
            if timing: tstart = tm.time()         
            
            #######################################################
            
            # reorder results in (i,j,k)
            for k in range(nz+1):
                idx = 2*k
                uz[i,j,k] = X[idx];
                vz[i,j,k] = X[idx+1];
                
            if timing: print('allocation OK.........', tm.time()-tstart)               
            if timing: tstart = tm.time()  
            
            
    #######################################################
    # Integrate vertically to get u,v,ug,vg
    print(uz.shape, z_w.shape)
    #print uz[10,10,10],bx[10,10,10], by[10,10,10], AKv[10,10,10]
    
    ut,vt,ug,vg = copy(new), copy(new), copy(new), copy(new)
    ut[:,:,1:] = integrate.cumtrapz(uz,z_w, axis=2 )
    vt[:,:,1:] = integrate.cumtrapz(vz,z_w, axis=2 )
    ug[:,:,1:] = (integrate.cumtrapz(-by,z_w, axis=2 ).T/f.T).T
    vg[:,:,1:] = (integrate.cumtrapz(bx,z_w, axis=2 ).T/f.T).T
    
    #ut[:,:,:-1] = integrate.cumtrapz(uz[:,:,::-1],z_w[:,:,::-1], axis=2 )[:,:,::-1]
    #vt[:,:,:-1] = integrate.cumtrapz(vz[:,:,::-1],z_w[:,:,::-1], axis=2 )[:,:,::-1]
    #ug[:,:,:-1] = (integrate.cumtrapz(-by[:,:,::-1],z_w[:,:,::-1], axis=2 )[:,:,::-1].T/f.T).T
    #vg[:,:,:-1] = (integrate.cumtrapz(bx[:,:,::-1],z_w[:,:,::-1], axis=2 )[:,:,::-1].T/f.T).T
    
    if debug==1:
        return ut,vt,ug,vg,uz,vz
    else:
        return ut,vt,ug,vg  
    
    

def Forder(var):
   return np.asfortranarray(var.T,dtype=np.float32)
   #return var

###################################################################################
# Reverse fortran order for arrays
####################################################################################
def revForder(var):
   return np.ascontiguousarray(var,dtype=np.float32).T

####################################################################################
#A highly specific function possibly useful for nonuniform grids
###################################################################################
def sliceSort(f,mask,Index=None):
  if len(f.shape)>2:
    print('******************Error******************* More than two dimensions')
  else:
    [Nx,Ny]=f.shape
  
  f1=np.reshape(f,(1,Nx*Ny))    
  f1=f1[~np.isnan(mask)] 
  print(f1.shape)
  if Index is None:
    #print 'Index not found'
    Ind=np.argsort(f1)
    print(Ind.shape,Ind)
    return f1[Ind],Ind
  else:
    return f1[Index]  
   #return var
###################################################################################

#Function generates file names
import matplotlib.pyplot as plt
import matplotlib as mpl
    
def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In: 
    cmap, name 
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """        
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(list(zip(k,reverse)))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r    
    
   
###################################################################################
###################################################################################
def cmap_inv(name):
  my_cmap=r_plt.nc_colormap(name)
  my_cmap_r = reverse_colourmap(my_cmap)
  my_cmap_r.set_bad('gray')
  return my_cmap_r  

def cmap_fwd(name, color = 'gray'):
  my_cmap=r_plt.nc_colormap(name)
  my_cmap.set_bad(color)
  return my_cmap  

# Initialize 
####################################################################################
#from mpl_toolkits.basemap import Basemap
def Basemap_init(x_rho,y_rho,axis=None,proj='merc',font=16,labelx=1,labely=1, nticks=[3,5]):
    lon0=(np.amin(x_rho)+np.amax(x_rho))/2.0
    lat_ts=(np.amin(y_rho)+np.amax(y_rho))/2.0
    if proj=='cyl':
      m = Basemap(projection=proj,llcrnrlat=np.amin(y_rho),urcrnrlat=np.amax(y_rho),llcrnrlon=np.amin(x_rho),urcrnrlon=np.amax(x_rho),lon_0=lon0,resolution='i',ax=axis) 
    elif proj=='merc':
      m = Basemap(projection=proj,llcrnrlat=np.amin(y_rho),urcrnrlat=np.amax(y_rho),llcrnrlon=np.amin(x_rho),urcrnrlon=np.amax(x_rho),lat_ts=lat_ts,resolution='i',ax=axis) 
    elif proj=='stere':
      m = Basemap(projection=proj,llcrnrlat=np.amin(y_rho),urcrnrlat=np.amax(y_rho),llcrnrlon=np.amin(x_rho),urcrnrlon=np.amax(x_rho),lat_ts=lat_ts,resolution='i',ax=axis) 
    elif proj=='ortho':
      m = Basemap(projection=proj,lon_0=lon0,lat_0=lat_ts,resolution='i',ax=axis) 
      #m = Basemap(projection=proj,lon_0=lon0,lat_0=lat_ts,llcrnrlat=np.amin(y_rho),urcrnrlat=np.amax(y_rho),llcrnrlon=np.amin(x_rho),urcrnrlon=np.amax(x_rho),resolution='i',ax=axis) 
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=1)
    lataxis=np.round(np.linspace(np.amin(y_rho),np.amax(y_rho),nticks[1]))
    lonaxis=np.round(np.linspace(np.amin(x_rho),np.amax(x_rho),nticks[0]))
    print(lataxis,lonaxis)
    np.insert(lataxis,0,1)
    m.drawparallels(lataxis,linewidth=0.3,labels=[labely,0,0,0],fontsize=font)
    m.drawmeridians(lonaxis,linewidth=0.3,labels=[0,0,0,labelx],fontsize=font)
    return m
    '''    if proj=='ortho':
        xmin, ymin = m(np.amin(x_rho), np.amin(y_rho))
        xmax, ymax = m(np.amax(x_rho), np.amax(y_rho))
        ax = plt.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    '''

import numpy.ma as ma
def mask_invalid(q,maxval=1000):
    q[q==0]=np.nan
    q[q>maxval]=np.nan
    q[q<-maxval]=np.nan
    return(ma.masked_invalid(q))

###################################################################################
import numpy.ma as ma
def prepVarPlot(var,maxval=10000, val = 0):
  var[var==0]=val
  var[var>maxval]=val
  var[var<-maxval]=val
  var[np.isinf(var)]=val
  var[np.isnan(var)]=val
  #varm = ma.masked_invalid(var)
  return var
###################################################################################
def makenan(x, val = np.nan):
    x[x==0] = val 
    x[x<-100000] = val 
    x[x>1000000] = val
###################################################################################


from os.path import join as pjoin
def gridDict(dr,gridfile, ij = None):
        #Returns dictionary containing basic info from ROMS grid file
        gdict={}
        print('gridfile: '+pjoin(dr,gridfile))
                
        if ij is not None:
                imin, imax, jmin, jmax = ij[0], ij[1], ij[2], ij[3]
        else:
                imin, imax, jmin, jmax = 0, None, 0, None
    
        nc = Dataset(pjoin(dr,gridfile), 'r')
        try:
                gdict['lon_rho']=Forder(nc.variables['lon_rho'][jmin:jmax,imin:imax])
                gdict['lat_rho']=Forder(nc.variables['lat_rho'][jmin:jmax,imin:imax])
        except:
                gdict['lon_rho']=Forder(nc.variables['x_rho'][jmin:jmax,imin:imax])
                gdict['lat_rho']=Forder(nc.variables['y_rho'][jmin:jmax,imin:imax])

        gdict['Ny']=gdict['lon_rho'].shape[1]
        gdict['Nx']=gdict['lon_rho'].shape[0]
        gdict['pm']=Forder(nc.variables['pm'][jmin:jmax,imin:imax])
        gdict['pn']=Forder(nc.variables['pn'][jmin:jmax,imin:imax])
        gdict['dx']=1.0/np.average(gdict['pm'])
        gdict['dy']=1.0/np.average(gdict['pn'])
        gdict['f']=Forder(nc.variables['f'][jmin:jmax,imin:imax])
        try:
            gdict['h']=Forder(nc.variables['h'][jmin:jmax,imin:imax])
        except:
            print('No h in grid file')    
        try:
            gdict['rho0'] = nc.rho0 
        except:
            print('No rho0 in grid file')    
        try:
            gdict['mask_rho'] = Forder(nc.variables['mask_rho'][jmin:jmax,imin:imax])
        except:
            print('No mask in file')
            pass
        nc.close()
        print(gdict['Nx'],gdict['Ny']) 
        return gdict

###################################################################################
###################################################################################
def wrtNcfile(dr, outfile, grddict, numRecords):
    g = grddict
    try :
          nco = Dataset(outfile, 'a')
    except :
          nco = Dataset(outfile, 'w')
    
    nco.createDimension('eta_rho',g['Ny'])
    nco.createDimension('eta_v',g['Ny']-1)
    nco.createDimension('xi_rho',g['Nx'])
    nco.createDimension('xi_u',g['Nx']-1)
    nco.createDimension('time',None)
    return nco

def wrtNcVars(nco, vardict):
        for var, value in zip(list(vardict.keys()), list(vardict.values())):
                nco.createVariable(var, np.dtype('float32').char, ('time','eta_rho','xi_rho'))
                nco.variables[:] = value    

###################################################################################
###################################################################################

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import NullFormatter,MultipleLocator, FormatStrFormatter
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
def colorbar_tight(ax,im,fontsize,numTicks,notation=0, sizeb = 2):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=str(sizeb)+"%", pad=0.03)
    cbar=plt.colorbar(im, cax=cax,extend='both')
    cbar.ax.tick_params(labelsize=fontsize) 
    tick_locator = ticker.MaxNLocator(nbins=numTicks)
    cbar.locator = tick_locator
    cbar.update_ticks()
    if notation==1:
       cbar.formatter.set_powerlimits((0, 0))
       cbar.update_ticks()

###################################################################################
def readvars(nch, varlist, time, glob, funclist = None, ij = None):
    vardict = {}
    if isinstance(varlist, list):
        for I, var in enumerate(varlist):
           if funclist is None:
              print(var)
              vardict[var] = ncload(nch, var, itime = time, ij = ij)
           else:
              vardict[var] = ncload(nch, var, itime = time, func = funclist[I], ij = ij)
    else:
        for I, var in enumerate(varlist.keys()):
           if funclist is None:
              print(var)
              vardict[varlist[var]] = ncload(nch, var, itime = time, ij = ij)
           else:
              vardict[varlist[var]] = ncload(nch, var, itime = time, func = funclist[I], ij = ij)
    return glob(vardict)
####################################################################################
def ncload(nc, var, itime = None, ij = None, func = None):
    
    if ij is not None:
        imin, imax, jmin, jmax = ij[0], ij[1], ij[2], ij[3]
    else:
        imin, imax, jmin, jmax = 0, None, 0, None
    
    xx = nc.variables[var].dimensions[-1]
    yy = nc.variables[var].dimensions[-2]
    
    if yy=='eta_v':
        if jmax is not None:
            jmax = jmax - 1
    if xx=='xi_u':
        if imax is not None:
            imax = imax - 1
    
    if itime is None:
        var1 = Forder(nc.variables[var][...,jmin:jmax,imin:imax]) 
    else:
        if isinstance(itime, list) or isinstance(itime, tuple):
            #print 'here'
            var1 = Forder(nc.variables[var][itime[0]:itime[1],...,jmin:jmax,imin:imax]) 
        else:
            var1 = Forder(nc.variables[var][itime,...,jmin:jmax,imin:imax]) 
    if func is None:
        return var1
    else:
        return func(var1)
    #return var1 if func is None else return func(var1)

####################################################################################
def ncloadz(nc, var, itime = None, iz = None, ij = None, func = None):
    
    if ij is not None:
        imin, imax, jmin, jmax = ij 
    else:
        imin, imax, jmin, jmax = 0, None, 0, None
    #print imin, imax, jmin, jmax, iz, itime
    xx = nc.variables[var].dimensions[-1]
    yy = nc.variables[var].dimensions[-2]
    
    if yy=='eta_v':
        if jmax is not None:
            jmax = jmax - 1
        #jmax = -1 if jmax is None else jmax = jmax - 1
    if xx=='xi_u':
        if imax is not None:
            imax = imax - 1
        #imax = -1 if imax is None else imax = imax - 1
    
    if itime is None:
        var1 = Forder(nc.variables[var][iz,:,:,jmin:jmax,imin:imax]) 
    else:
        if isinstance(itime, list) or isinstance(itime, tuple):
            #print 'here'
            var1 = Forder(np.squeeze(nc.variables[var][itime[0]:itime[1],iz,jmin:jmax,imin:imax]))
        else:
            if iz is None:
                var1 = Forder(np.squeeze(nc.variables[var][itime,:,jmin:jmax,imin:imax])) 
            else:    
                var1 = Forder(np.squeeze(nc.variables[var][itime,iz,jmin:jmax,imin:imax])) 
    if func is None:
        return var1
    else:
        return func(var1)
    #return var1 if func is None else return func(var1)

####################################################################################
def ncload3d(nc, var, time = None, func = None, ijrho = None, grd = 'rho'):
    if time is not None:
        var1 = Forder(nc.variables[var][time,:,:,:])
    else:
        var1 = Forder(nc.variables[var][:,:,:])

    if func is None:
        return var1
    else:
        return func(var1)
####################################################################################
def ncload2d(nc, var, time = None, func = None):
    if time is not None:
        var1 = Forder(nc.variables[var][time,:,:])
    else:
        var1 = Forder(nc.variables[var][:,:])

    if func is None:
        return var1
    else:
        return func(var1)

####################################################################################
#make a copy with dimensions same as source but variables in varlist with same dims as var
####################################################################################

def addDim(nco, name, dimname = None):
    if dimname is None:
        dimname  = name
    nco.createDimension(dimname, len(globals(name))) 
    nco.createVariable(name, np.dtype('float32').char, (dimname))

####################################################################################

def ncdimcopy(src, dst, exclude = None, ij = None, add  = None):
    
    #Important if you want to only copy a horizontal subset of data
    if ij is not None:
        dims = {}
        try:
            xdim, ydim  = (ij[1] - ij[0]), (ij[3] - ij[2])
        except:
            xdim, ydim = ij[0], ij[1]
        dims = {'xi_rho':xdim, 'eta_rho':ydim, 'xi_u':xdim-1, 'eta_v':ydim-1}
    
    dst.setncatts(src.__dict__)
    for name, dimension in src.dimensions.items():
        try:
            ndim = dims[name]
        except:
            ndim  = len(dimension) if not dimension.isunlimited() else None
                
        if exclude is not None and exclude == name:
          continue
        else:
          dst.createDimension(name, ndim)
    if add is not None:
        addDim(dst, add)
####################################################################################
def subset(nc, var, ij, itime=None):
    imin, imax, jmin, jmax = ij[0], ij[1], ij[2]. ij[3]
    xx = nc.variables[var].dimensions[-1]
    yy = nc.variables[var].dimensions[-2]
    if yy=='eta_v':
        jmin = jmin -1
    if xx=='xi_u':
        imin = imin -1
    if itime is None:
        return nc.variables[var][...,jmin:jmax,imin:imax] 
    else:
        return nc.variables[var][itime,...,jmin:jmax,imin:imax] 
####################################################################################
def ncvarcopy(src, dst, ivarname, varlist, exclude =  None, dtype = None):
    ivar = src.variables[ivarname]
    dims = ivar.dimensions
    if exclude is not None:
        #No idea why I need three steps to do this
        ldims = list(dims)
        dim = [dim for dim in ldims if dim != exclude]
        dims = tuple(dim)
    for vname in varlist:
        try:
            if dtype is None:
                dst.createVariable(vname, ivar.datatype, dims)
            else:
                dst.createVariable(vname, dtype , dims)

            #dst[name].setncatts(src[name].__dict__)
        except:
            print(vname + ' Failed')
    #Add ocean_time id present
    try:
        var = src.variables['ocean_time']
        dst.createVariable('ocean_time', var.datatype, var.dimensions)
    except:
        print('No ocean_time in file')
####################################################################################
def nc1varcopy(src, dst, name, namef = None):
    ivar = src.variables[name]
    if namef is not None:
        dst.createVariable(namef, ivar.datatype, ivar.dimensions)
        dst[namef].setncatts(src[name].__dict__)
    else:
        dst.createVariable(name, ivar.datatype, ivar.dimensions)
        dst[name].setncatts(src[name].__dict__)
    try:
        var = src.variables['ocean_time']
        dst.createVariable('ocean_time', var.datatype, var.dimensions)
    except:
        print('No ocean_time in file or already exits')
####################################################################################
####################################################################################
def ncallvarscopy(src, dst, varlist):
    for name in varlist:
        ivar = src.variables[name]
        dst.createVariable(name, ivar.datatype, ivar.dimensions)
        dst[name].setncatts(src[name].__dict__)
    #Add ocean_time id present
    try:
        var = src.variables['ocean_time']
        dst.createVariable('ocean_time', var.datatype, var.dimensions)
    except:
        print('No ocean_time in file')

####################################################################################
def nccopy(src, dst, ivarname, varlist, exclude = None, ij = None):
    ncdimcopy(src,dst,exclude, ij = ij) 
    ncvarcopy(src, dst, ivarname, varlist, exclude)    
####################################################################################

def advectiveFlx(u, v, w, z_r, z_w, pm, pn, vtype = 'zeta', simple = False, mask = None):
  ux, uy, uz, vx, vy, vz = shear(u, v, z_r, z_w, pm, pn, simple = simple, mask = mask) 
  #dx(uq)+dy(vq)+dz(wq)
  #Compute dx(uu), dy(vu), dz(wu) 
  #Or Compute udx(u), vdy(u), wdz(u) 
  #and Compute udx(v), vdy(v), wdz(v) 
  return u2rho(u)*ux + v2rho(v)*uy + w*uz, u2rho(u)*vx + v2rho(v)*vy + w*vz
####################################################################################
def nwrite(nc, var, time = None, func = None):
    if time is not None:
        var1 = Forder(nc.variables[var][itime,:,:,:])
    else:
        var1 = Forder(nc.variables[var][:,:,:])

    if func is None:
        return var1
    else:
        return func(var1)
####################################################################################
def intx(var, gd):
    dx = np.min(gd['pm'])
    return np.nansum(var, axis=0)*dx

def int_xz(var, z_r, z_w, h, gd ):
    #dx = np.min(gd['pm'])
    return np.nansum(vertIntSeamount(var, z_w, z_r, h, 1000), axis=0)*gd['dx']
    ####################################################################################
#trivial identity function
def ident(var):
        return var
    ####################################################################################

def ekeFlux(dr, gridfile, hisfileStr, start, end, step):
    outfile = dr + 'ekeTimeMeanFluxes.nc'
    print('output file: ' + outfile)
    gdc = gridDict(dr,gridfile)

    hisfile = dr + hisfileStr + '.{0:04}'.format(start)+'.nc'
    nch = Dataset(hisfile, 'r')
    varnames = ['hrs', 'vrs', 'wb', 'eke', 'mke']
    varnames2d = [var + '2d' for var in varnames] 
    varnames1d = [var + '1d' for var in varnames] 
    dimnames = ['s_rho', 'eta_rho', 'xi_rho']
    dimnames2d = ['eta_rho', 'xi_rho']
    dimnames1d = ['eta_rho']
    print('output variables: ', varnames)
    print('output dimensions: ', dimnames)
    #Create output file
    nco = Dataset(outfile, 'w')
    #Create dimensions 
    for dim in dimnames:
        nco.createDimension(dim,len(nch.dimensions[dim]))
    #Create variables 
    for var, var2d, var1d in zip(varnames, varnames2d, varnames1d):
        nco.createVariable(var,np.dtype('float32').char,tuple(dimnames))
        nco.createVariable(var2d,np.dtype('float32').char,tuple(dimnames2d))
        nco.createVariable(var1d,np.dtype('float32').char,tuple(dimnames1d))
    #Copy global attributes from history file 
    for name in nch.ncattrs():
        nco.setncattr(name, nch.getncattr(name))
    # First calculate the means 
    (z_r,z_w) = toolsF.zlevs(gdc['h'], gdc['h'] * 0, nch.hc, nch.Cs_r, nch.Cs_w)
    #Input output variable dictionaries
    ind = {}
    od = {}
    varinlist = ['u', 'v', 'w', 'temp', 'salt']
    ifunclist = [u2rho, v2rho, ident, ident, ident]
    varoutlist = ['uu', 'uv', 'uw', 'vv', 'vw', 'wb', 'u', 'v', 'w', 'b']
    #Temp function to compute output products from input components
    ####################################################################################
    def prod(ind, outkey):
        out = ind[outkey[0]]  
        outstr = outkey[0]
        for I, c in enumerate(outkey):
            if I==0:
                continue
            out = out*ind[c]
            outstr += ('*' + c)
        print('Done with ' + outstr)
        return out   
    ####################################################################################
    #Temp function to compute reynolds stress
    def tau(od, vstr):
        assert(len(vstr) == 2)
        print(vstr+'-'+vstr[0]+'*'+vstr[1])
        return (od[vstr] - od[vstr[0]]*od[vstr[1]])

    ####################################################################################
    
    count = 0
    for filenum in range(start, end,  step):
        hisfile = dr + hisfileStr + '.{0:04}'.format(filenum)+'.nc'
        nch = Dataset(hisfile, 'r')
        for itime in range(step):
            #zeta=Forder(nch.variables['zeta'][itime,:,:])
            #Read input variables
            try:
              for func, var in zip(ifunclist, varinlist):
                ind[var] = ncload3d(nch, var, itime, func)
                print(var  + ' loaded.')
            except:
                ind['salt'] = ind['temp'] * 0
                ind['salt'][:,:,:] = 34.5
            ind['b'] = toolsF.get_buoy(ind['temp'], ind['salt'], z_r, z_w, nch.rho0)
            #Compute output variables
            if count == 0:
                for var in varoutlist:
                    od[var] = prod(ind, var)        
            else:
                for var in varoutlist:
                    od[var] = od[var] + prod(ind, var)         
            count = count + 1
            print(count)
    for var in varoutlist:
        od[var] /= float(count)         
    ux, uy, uz, vx, vy, vz = shear(rho2u(od['u']), rho2v(od['v']), z_r, z_w, gdc['pm'], gdc['pn'], simple = False, mask = None)
    print(uy.shape, vx.shape, ux.shape, vy.shape, vz.shape, uz.shape)
    
    hrs = tau(od, 'uv') * (uy + vx) + tau(od, 'uu') * ux + tau(od, 'vv') * vy
    vrs = tau(od, 'uw') * uz + tau(od, 'vw') * vz
    wb = tau(od, 'wb')
    eke = 0.5 * (tau(od, 'vv') +  tau(od, 'uu'))
    mke = 0.5 * (od['v']*od['v'] + od['u']*od['u'])
    
    for var in varnames:
        nco.variables[var][:,:,:] = revForder(locals()[var])  
        nco.variables[var+'2d'][:,:] = revForder(vertIntSeamount(locals()[var], z_w, z_r, gdc['h'], 970))  
        nco.variables[var+'1d'][:] = revForder(int_xz(locals()[var], z_r, z_w, gdc))  

####################################################################################
def wb(dr, gridfile, hisfileStr, start, end, step):
    outfile = dr + 'ekeTimeMeanFluxes.nc'
    print('output file: ' + outfile)
    gdc = gridDict(dr,gridfile)

    hisfile = dr + hisfileStr + '.{0:04}'.format(start)+'.nc'
    nch = Dataset(hisfile, 'r')
    varnames = ['wb']
    varnames2d = [var + '2d' for var in varnames] 
    varnames1d = [var + '1d' for var in varnames] 
    dimnames = ['s_rho', 'eta_rho', 'xi_rho']
    dimnames2d = ['eta_rho', 'xi_rho']
    dimnames1d = ['eta_rho']
    print('output variables: ', varnames)
    print('output dimensions: ', dimnames)
    #Create output file
    nco = Dataset(outfile, 'w')
    #Create dimensions 
    for dim in dimnames:
        nco.createDimension(dim,len(nch.dimensions[dim]))
    #Create variables 
    for var, var2d, var1d in zip(varnames, varnames2d, varnames1d):
        nco.createVariable(var,np.dtype('float32').char,tuple(dimnames))
        nco.createVariable(var2d,np.dtype('float32').char,tuple(dimnames2d))
        nco.createVariable(var1d,np.dtype('float32').char,tuple(dimnames1d))
    #Copy global attributes from history file 
    for name in nch.ncattrs():
        nco.setncattr(name, nch.getncattr(name))
    # First calculate the means 
    (z_r,z_w) = zlevs(gdc, nch) 
    #Input output variable dictionaries
    ind = {}
    od = {}
    ####################################################################################
    #trivial identity function
    def ident(var):
      return var
    ####################################################################################
    varinlist = ['w', 'temp', 'salt']
    ifunclist = [ident, ident, ident]
    varoutlist = ['wb', 'w', 'b']
    #Temp function to compute output products from input components
    ####################################################################################
    def prod(ind, outkey):
        out = ind[outkey[0]]  
        outstr = outkey[0]
        for I, c in enumerate(outkey):
            if I==0:
                continue
            out = out*ind[c]
            outstr += ('*' + c)
        print('Done with ' + outstr)
        return out   
    ####################################################################################
    #Temp function to compute reynolds stress
    def tau(od, vstr):
        assert(len(vstr) == 2)
        print(vstr+'-'+vstr[0]+'*'+vstr[1])
        return (od[vstr] - od[vstr[0]]*od[vstr[1]])

    ####################################################################################
    
    count = 0
    for filenum in range(start, end,  step):
        hisfile = dr + hisfileStr + '.{0:04}'.format(filenum)+'.nc'
        nch = Dataset(hisfile, 'r')
        for itime in range(step):
            #zeta=Forder(nch.variables['zeta'][itime,:,:])
            #Read input variables
            try:
              for func, var in zip(ifunclist, varinlist):
                ind[var] = ncload3d(nch, var, itime, func)
                print(var  + ' loaded.')
            except:
                ind['salt'] = ind['temp'] * 0
                ind['salt'][:,:,:] = 34.5
            ind['b'] = toolsF.get_buoy(ind['temp'], ind['salt'], z_r, z_w, nch.rho0)
            #Compute output variables
            if count == 0:
                for var in varoutlist:
                    od[var] = prod(ind, var)        
            else:
                for var in varoutlist:
                    od[var] = od[var] + prod(ind, var)         
            count = count + 1
            print(count)
    for var in varoutlist:
        od[var] /= float(count)         
    wb = tau(od, 'wb')
    
    for var in varnames:
        nco.variables[var][:,:,:] = revForder(locals()[var])  
        nco.variables[var+'2d'][:,:] = revForder(vertIntSeamount(locals()[var], z_w, z_r, gdc['h'], 970))  
        nco.variables[var+'1d'][:] = revForder(int_xz(locals()[var], z_r, z_w, gdc))  
####################################################################################
def ekeFlux_z(dr, gridfile, hisfileStr, start, end, step, D=1000):
    outfile = dr + 'ekeTimeMeanFluxes_z.nc'
    print('output file: ' + outfile)
    gdc = gridDict(dr,gridfile)

    hisfile = dr + hisfileStr + '.{0:04}'.format(start)+'.nc'
    nch = Dataset(hisfile, 'r')
    depth = nch.variables['depth'][:]
    dz = abs(depth[1] - depth[0])
    varnames = ['hrs', 'vrs', 'wb', 'eke', 'mke']
    varnames2d = [var + '2d' for var in varnames] 
    varnames1d = [var + '1d' for var in varnames] 
    dimnames = ['depth', 'eta_rho', 'xi_rho']
    dimnames2d = ['eta_rho', 'xi_rho']
    dimnames1d = ['eta_rho']
    print('output variables: ', varnames)
    print('output dimensions: ', dimnames)
    #Create output file
    nco = Dataset(outfile, 'w')
    #Create dimensions 
    for dim in dimnames:
        nco.createDimension(dim,len(nch.dimensions[dim]))
    #Create variables 
    for var, var2d, var1d in zip(varnames, varnames2d, varnames1d):
        nco.createVariable(var,np.dtype('float32').char,tuple(dimnames))
        nco.createVariable(var2d,np.dtype('float32').char,tuple(dimnames2d))
        nco.createVariable(var1d,np.dtype('float32').char,tuple(dimnames1d))
    #Copy global attributes from history file 
    for name in nch.ncattrs():
        nco.setncattr(name, nch.getncattr(name))
    # First calculate the means 
    #Input output variable dictionaries
    ind = {}
    od = {}
    ####################################################################################
    #trivial identity function
    def ident(var):
      return var
    ####################################################################################
    varinlist = ['u', 'v', 'w', 'temp', 'salt']
    ifunclist = [u2rho, v2rho, ident, ident, ident]
    varoutlist = ['uu', 'uv', 'uw', 'vv', 'vw', 'wb', 'u', 'v', 'w', 'b']
    #Temp function to compute output products from input components
    ####################################################################################
    def prod(ind, outkey):
        out = ind[outkey[0]]  
        outstr = outkey[0]
        for I, c in enumerate(outkey):
            if I==0:
                continue
            out *= ind[c]
            outstr += ('*' + c)
        print('Done with ' + outstr)
        return out   
    ####################################################################################
    #Temp function to compute reynolds stress
    def tau(od, vstr):
        assert(len(vstr) == 2)
        print(vstr+'-'+vstr[0]+'*'+vstr[1])
        return (od[vstr] - od[vstr[0]]*od[vstr[1]])

    ####################################################################################
    
    count = 0
    for filenum in range(start, end,  step):
        hisfile = dr + hisfileStr + '.{0:04}'.format(filenum)+'.nc'
        nch = Dataset(hisfile, 'r')
        for itime in range(step):
            #zeta=Forder(nch.variables['zeta'][itime,:,:])
            #Read input variables
            try:
              for func, var in zip(ifunclist, varinlist):
                ind[var] = ncload3d(nch, var, itime, func)
                ind[var][np.abs(ind[var])>10000] = np.nan
                ind[var][ind[var] == 0] = np.nan
                print(var  + ' loaded.')
            except:
                ind['salt'] = ind['temp'] * 0
                ind['salt'][:,:,:] = 34.5
            ind['b'] = toolsF.get_buoy_depth(ind['temp'], ind['salt'], depth, gdc['rho0'])
            #Compute output variables
            if count == 0:
                for var in varoutlist:
                    od[var] = prod(ind, var)        
            else:
                for var in varoutlist:
                    od[var] = od[var] + prod(ind, var)         
            count = count + 1
            print(count)
    for var in varoutlist:
        od[var] /= float(count)         
    ux, uy, uz, vx, vy, vz = shear_z(rho2u(od['u']), rho2v(od['v']), depth, gdc['pm'], gdc['pn'])
    print(uy.shape, vx.shape, ux.shape, vy.shape, vz.shape, uz.shape)
    
    hrs = tau(od, 'uv') * (uy + vx) + tau(od, 'uu') * ux + tau(od, 'vv') * vy
    vrs = tau(od, 'uw') * uz + tau(od, 'vw') * vz
    wb = tau(od, 'wb')
    eke = 0.5 * (tau(od, 'vv') +  tau(od, 'uu'))
    mke = 0.5 * (od['v']*od['v'] + od['u']*od['u'])
    
    for var in varnames:
        nco.variables[var][:,:,:] = revForder(locals()[var])  
        nco.variables[var+'2d'][:,:] = revForder(np.nansum(locals()[var], axis = -1) * dz)  
        nco.variables[var+'1d'][:] = revForder(np.nansum(np.nansum(locals()[var], axis = -1), axis = 0) * gdc['dx'] * dz)  

#returns complement
####################################################################################
def comp(mask):
    return (1 - mask)

#returns masks for cyclones, anticylones 
####################################################################################
def masks_vrt(gradb, vrt, f = None, gradthresh = None, vrtthresh = 0.1):
    maskgb = gradb * 0
    if np.max(vrt)<0.001:
        vrt = vrt/f
    if gradthresh is None:
        gradthresh = np.nanmean(gradb)
        print(gradthresh)
    maskgb[gradb > 1.5*gradthresh] = 1
    maskvrtcy = vrt * 0
    maskvrtacy = vrt * 0
    maskvrtcy[vrt > vrtthresh] = 1
    maskvrtacy[vrt < -vrtthresh] = 1
    #print np.sum(maskgb), np.sum(maskvrtcy), np.sum(maskvrtacy)
    return  comp(maskgb)*maskvrtcy, comp(maskgb)*maskvrtacy
#returns masks for cyclones, anticylones, fronts, strained 
####################################################################################
def masks(S, gradb, vrt, div = None, f = None, gradthresh = None, sthresh = 0.1414, vrtthresh = 0.1, divthresh = 0.05):
    maskgb = gradb * 0
    maskS = S * 0
    if np.max(vrt)<0.001:
        vrt = vrt/f
        S = S/f/f
        if div is not None:
            div = div/f
    if gradthresh is None:
        gradthresh = np.nanmean(gradb)
        print(gradthresh)
    maskgb[gradb > 1.5*gradthresh] = 1
    if sthresh is None:
        sthresh = np.nanmean(S)**0.5
    maskS[S > sthresh**2] = 1
            
    if div is not None:
        maskdiv = div * 0
        maskdiv[np.abs(div) > divthresh] = 1
    maskvrtcy = vrt * 0
    maskvrtacy = vrt * 0
    maskvrtcy[vrt > vrtthresh] = 1
    maskvrtacy[vrt < -vrtthresh] = 1
    #print np.sum(maskgb), np.sum(maskvrtcy), np.sum(maskvrtacy)
    if div is not None:
        return  maskgb, maskS, comp(maskgb)*maskvrtcy, comp(maskgb)*maskvrtacy, maskdiv
    else:
        return  maskgb, maskS, comp(maskgb)*maskvrtcy, comp(maskgb)*maskvrtacy

#conditionally averages var over the mask. Var here is 2+1 d
####################################################################################
def condAv(var, mask = None):
    if mask is None:
        return np.mean(var)
    if len(np.shape(var))==3:
        return (np.sum(var * mask[:,:,None], axis = (0, 1))/np.sum(mask))
    else :
        return (np.sum(var * mask[:,:], axis = (0, 1))/np.sum(mask))
####################################################################################
import pickle

def wrtpkl(fname, var):
    with open(fname+'.pkl','wb') as f:
       pickle.dump(var, f)

####################################################################################
def readpkl(fname):
   with open(fname, 'rb') as f:
       var = pickle.load(f) 
   return var    
####################################################################################
#Create an arbitrary netcdf output file and add dimensions and variables
###################################################################################
#if depth is a vector, it is the depth else the size of s_rho
def ncCreateDims(nco, gd, depth = None, time = True):
    # Create dimensions
    if time:
        keys = ['time']
        values = [None]
    if depth is not None:
        if np.isscalar(depth):
            keys.append(['s_rho', 's_w'])
            values.append([depth, depth+1])
        else:
            keys.append('depth')
            values.append(len(depth))
    keys += ['xi_rho', 'eta_rho', 'xi_u', 'eta_v']
    values += [gd['Nx'], gd['Ny'], gd['Nx']-1, gd['Ny']-1]
    print(keys,values)
    print('Creating dimensions..')
    
    for k, v in zip(keys, values):
        #print(k,': ', v)
        nco.createDimension(k, v)

###################################################################################
#if depth is a vector, it is the depth else the size of s_rho
#gtype is two characters, Second is horizontal: u,v,r(default) and First is vertical :r(default), w
def ncCreateVars(nco, var, gd, depth = None, gtype = 'rr', time = True):
    # Add variable
    x = {'r':'xi_rho', 'u': 'xi_u', 'v': 'xi_rho'}
    y = {'r':'eta_rho', 'u': 'eta_rho', 'v': 'eta_v'}
    z = {'r':'s_rho', 'w':'s_w'}
    dims = [] 
    if time is True:
        dims.append('time')
        try:
            nco.createVariable('ocean_time', np.dtype('float32').char, ('time'))
        except:
            pass
            #print('ocean_time exists..')
        
    if np.isscalar(depth):
        dims.append([z[gtype[0]]])
    elif depth is not None:
        dims.append('depth')
        try:
            nco.createVariable('depth', np.dtype('float32').char, ('depth'))
            nco.variables['depth'][:] = depth
        except:
            pass
            #print('depth exists..')
    dims = tuple(dims + [y[gtype[1]]] + [x[gtype[1]]])
    #print(var, dims)
    nco.createVariable(var, np.dtype('float32').char, dims)

###################################################################################
def ncCreateList(nco, varlist, gd, depth = None, xtype = None, ztype=None, time = True):
    N = len(varlist)
    if xtype is None:
        xtype = ['r']*N
    elif len(xtype)==1:
        xtype = [xtype]*N
    if ztype is None:
        ztype = ['r']*N
    else:
        ztype = [ztype]*N
    for iv, var in enumerate(varlist):
        print(var, ztype[iv] + xtype[iv])
        ncCreateVars(nco, var, gd, depth, gtype = ztype[iv] + xtype[iv] , time = time)
        
###################################################################################
###################################################################################
