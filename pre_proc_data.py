import numpy as np
import os
from vtk import *
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.numpy_interface import algorithms as algs
import vtki
import warnings

warnings.simplefilter("ignore", FutureWarning)
 
# User inputs
#############
RANS_fileloc  = '.'
#RANS_filename = 'flatplate.vtk'
RANS_filename = 'flow.vtk'

LES_fileloc  = '.'
LES_filename = 'h20.vtu'

out_fileloc  = '.'
out_filename = 'output.vtk'

plot = False

##############
# Read in data
##############
# Read in RANS data
###################
print('\nReading data')
print(  '------------')

filename = os.path.join(RANS_fileloc, RANS_filename)

print('Reading RANS data from vtk file: ', filename)

# Read vtk file with vtki
rans_vtk = vtki.read(filename)  

# Get basic info about mesh
rans_nnode = rans_vtk.number_of_points
rans_ncell = rans_vtk.number_of_cells
print('Number of nodes = ', rans_nnode)
print('Number of cells = ', rans_ncell)

# Sort out scalar names (TODO - automate this somehow, very much for SU2 atm)
rans_vtk.rename_scalar('Momentum','roU') #TODO - auto check if velocities or Momentum in vtk file
rans_vtk.rename_scalar('Pressure' ,'p') #TODO - Check if Energy or pressure etc
rans_vtk.rename_scalar('Density' ,'ro')
rans_vtk.rename_scalar('TKE' ,'k')
rans_vtk.rename_scalar('Omega' ,'w') #TODO - Read in eps if that is saved instead
rans_vtk.rename_scalar('Laminar_Viscosity','mu_l') #TODO - if this doesn't exist auto calc from sutherlands
rans_vtk.rename_scalar('Eddy_Viscosity','mu_t') #TODO - if this doesn't exist calc from k and w
rans_vtk.rename_scalar('Wall_Distance','d') #TODO - if this doesn't exist calc 
rans_vtk.point_arrays['U'] = rans_vtk.point_arrays['roU']/np.array([rans_vtk.point_arrays['ro'],]*3).transpose()

print('List of RANS data fields:\n', rans_vtk.scalar_names)

# Remove viscous wall d=0 points to prevent division by zero in feature construction
print('Removing viscous wall (d=0) nodes from mesh')
rans_vtk = rans_vtk.threshold([1e-12,1e99],scalars='d')
print('Number of nodes extracted = ', rans_nnode - rans_vtk.number_of_points)
rans_nnode = rans_vtk.number_of_points
rans_ncell = rans_vtk.number_of_cells
print('New number of nodes = ', rans_nnode)
print('New number of cells = ', rans_ncell)

# Read in LES data
##################
filename = os.path.join(LES_fileloc, LES_filename)

print('\nReading LES data from vtk file: ', filename)

# Read vtk file with vtki
les_vtk = vtki.read(filename)  

# Get basic info about mesh
les_nnode = les_vtk.number_of_points
les_ncell = les_vtk.number_of_cells
print('Number of nodes = ', les_nnode)
print('Number of cells = ', les_ncell)

# Sort out scalar names (TODO - automate this somehow)
les_vtk.rename_scalar('avgUx','U')
les_vtk.rename_scalar('avgUy','V')
les_vtk.point_arrays['W'] = np.ones(les_nnode)*0.0
les_vtk.rename_scalar('avgP' ,'p')
les_vtk.point_arrays['ro']  = np.ones(les_nnode) 
les_vtk.rename_scalar('UU','uu')
les_vtk.rename_scalar('VV','vv')
les_vtk.rename_scalar('WW','ww')
les_vtk.rename_scalar('UV','uv')

print('List of LES data fields:\n', les_vtk.scalar_names)


#####################################
# Interpolate LES data onto RANS mesh
#####################################
print('\nInterpolating LES data onto RANS mesh')
print(  '-------------------------------------')

# Wrap vtki objects in dsa wrapper
# NOTE - NO VTK OPERATIONS SUCH AS INTERP, SLICING ETC BEYOND THIS POINT
rans_dsa = dsa.WrapDataObject(rans_vtk)
les_dsa  = dsa.WrapDataObject(les_vtk)

##############################################
# Process RANS data to generate feature vector
##############################################
print('\nProcessing RANS data into features')
print(  '----------------------------------')
nfeat = 12
q = np.empty([rans_nnode,nfeat])

# Feature 1: non-dim Q-criterion
################################
# Velocity vector
U = rans_dsa.PointData['U'] # NOTE - Getting variables from dsa obj not vtk obj as want to use algs etc later

# Velocity gradient tensor and its transpose
J  = algs.gradient(U)        # J[:,j-1,i-1] is dUidxj
Jt = algs.apply_dfunc(np.transpose,J,(0,2,1))

# Strain and vorticity tensors
Sij = 0.5*(J+Jt)
Oij = 0.5*(J-Jt)

# Frob. norm of Sij and Oij  (Snorm and Onorm are actually S^2 and O^2, sqrt needed to get norms)
Snorm = algs.sum(2.0*Sij**2,axis=1) # sum i axis
Snorm = algs.sum(     Snorm,axis=1) # sum previous summations i.e. along j axis
Onorm = algs.sum(2.0*Oij**2,axis=1) # sum i axis
Onorm = algs.sum(    Onorm, axis=1) # sum previous summations i.e. along j axis

# Store q1
q[:,0] = (Onorm - Snorm)/(Onorm + Snorm)

# clean up 
Snorm = algs.sqrt(Snorm) #revert to revert to real Snorm for use later
Onorm = algs.sqrt(Onorm) #revert to revert to real Onorm for use later

# Feature 2: Turbulence intensity
#################################
k = rans_dsa.PointData['k'] 
UiUi = algs.mag(U)**2.0
q[:,1] = k/(0.5*UiUi+k)

# Feature 3: Turbulence Reynolds number
#######################################
nu = rans_dsa.PointData['mu_l']/rans_dsa.PointData['ro']
q3 = (algs.sqrt(k)*rans_dsa.PointData['d'])/(50.0*nu)
q[:,2] = algs.apply_dfunc(np.minimum, q3, 2.0)

# Feature 4: Pressure gradient along streamline
###############################################
A = np.zeros(rans_nnode)
B = np.zeros(rans_nnode)

dpdx  = algs.gradient(rans_dsa.PointData['p'])

for k in range(0,3):
    A += U[:,k]*dpdx[:,k] 

for i in range(0,3):
    for j in range(0,3):
        B += U[:,i]*U[:,i]*dpdx[:,j]*dpdx[:,j]

q[:,3] = A/(algs.sqrt(B)+algs.abs(A))


# Feature 5: Ratio of turb time scale to mean strain time scale
###############################################################
eps = k*rans_dsa.PointData['w']
#bstar = 0.09 #TODO - which version for SU2. TODO - Option to select omega definition for user
#eps = bstar*k*rans_dsa.PointData['w']
q[:,4] = Snorm*k/(Snorm*k + eps)

# Feature 6: Viscosity ratio
############################
nu_t = rans_dsa.PointData['mu_t']/rans_dsa.PointData['ro']
q[:,5] = nu_t/(100.0*nu + nu_t)

# Feature 7: Ratio of pressure normal stresses to normal shear stresses
#######################################################################
A = np.zeros(rans_nnode)
B = np.zeros(rans_nnode)

for i in range(0,3):
    A += dpdx[:,i]*dpdx[:,i]

for k in range(0,3):
    B += 0.5*rans_dsa.PointData['ro']*J[:,k,k]*J[:,k,k]

q[:,6] = algs.sqrt(A)/(algs.sqrt(A)+B)

# Feature 8: Vortex stretching
##############################
A = np.zeros(rans_nnode)
B = np.zeros(rans_nnode)

vortvec = algs.vorticity(U)

for j in range(0,3):
    for i in range(0,3):
        for k in range(0,3):
            A += vortvec[:,j]*J[:,j,i]*vortvec[:,k]*J[:,k,i]

B = Snorm

q[:,7] = algs.sqrt(A)/(algs.sqrt(A)+B)
q[:,0] = vortvec[:,0]
q[:,1] = vortvec[:,1]
q[:,2] = vortvec[:,2]

# Feature 9: Marker of Gorle et al. (deviation from parallel shear flow)
########################################################################
A = np.zeros(rans_nnode)
B = np.zeros(rans_nnode)

for i in range(0,3):
    for j in range(0,3):
        A += U[:,i]*U[:,j]*J[:,j,i]/(UiUi)

for n in range(0,3):
    for i in range(0,3):
        for j in range(0,3):
            for m in range(0,3):
                B += U[:,n]*U[:,n]*U[:,i]*J[:,j,i]*U[:,m]*J[:,j,m]
q[:,8] = algs.abs(A)/(algs.sqrt(B)+algs.abs(A))


# Feature 10: Ratio of convection to production of k
####################################################
# TODO - Get Reynolds stresses from LEVM and CEVM

# Feature 11: Ratio of total Reynolds stresses to normal Reynolds stresses
##########################################################################

# Feature 12: Cubic eddy viscosity comparision
##############################################

# Store features in vtk obj
###########################
rans_vtk.point_arrays['q'] = q

###################
# Write to vtk file
###################
filename = os.path.join(out_fileloc, out_filename)
rans_vtk.save(filename)

#####################
# Plot stuff to check
#####################
if (plot):
    plotter = vtki.Plotter()
    sargs = dict(interactive=True,height=0.25,title_font_size=12, label_font_size=11,shadow=True, n_labels=5, italic=True, fmt='%.1f',font_family='arial',vertical=False)
    rans_vtk.point_arrays['plot'] = q3
    clims = [np.min(q3), np.max(q3)]
    print(clims)
    plotter.add_mesh(rans_vtk,scalars='plot',rng=clims,scalar_bar_args=sargs)
    plotter.view_xy()
    #plotter.add_mesh(data2,scalars=J2[:,1,0],show_scalar_bar=True,scalar_bar_args=sargs,rng=[-100,100]) #can plot np array directly but colour bar doesn't work...
    plotter.show()
    
