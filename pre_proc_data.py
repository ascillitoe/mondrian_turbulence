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

# Constants
###########
Cmu = 0.09

# Craft's CEVM constants
C1 = -0.1
C2 = 0.1
C3 = 0.26
C4 = -10.0*Cmu**2.0
C5 = 0.0
C6 = -5.0*Cmu**2.0
C7 = 5.0*Cmu**2.0

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
# J[:,i-1,j-1] is dUidxj
# Jt[:,i-1,j-1] is dUjdxi
Jt = algs.gradient(U)        # Jt is this one as algs uses j,i ordering
J  = algs.apply_dfunc(np.transpose,Jt,(0,2,1))

# Strain and vorticity tensors
Sij = 0.5*(J+Jt)
Oij = 0.5*(J-Jt)

# Frob. norm of Sij and Oij  (Snorm and Onorm are actually S^2 and O^2, sqrt needed to get norms)
Snorm = algs.sum(2.0*Sij**2,axis=1) # sum i axis
Snorm = algs.sum(     Snorm,axis=1) # sum previous summations i.e. along j axis
Onorm = algs.sum(2.0*Oij**2,axis=1) # sum i axis
Onorm = algs.sum(     Onorm,axis=1) # sum previous summations i.e. along j axis

# Store q1
q[:,0] = (Onorm - Snorm)/(Onorm + Snorm)

# clean up 
Snorm = algs.sqrt(Snorm) #revert to revert to real Snorm for use later
Onorm = algs.sqrt(Onorm) #revert to revert to real Onorm for use later

# Feature 2: Turbulence intensity
#################################
tke = rans_dsa.PointData['k'] 
UiUi = algs.mag(U)**2.0
q[:,1] = tke/(0.5*UiUi+tke)

# Feature 3: Turbulence Reynolds number
#######################################
nu = rans_dsa.PointData['mu_l']/rans_dsa.PointData['ro']
q3 = (algs.sqrt(tke)*rans_dsa.PointData['d'])/(50.0*nu)
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
A = 1.0/rans_dsa.PointData['w']  #Turbulent time scale (eps = k*w therefore also A = k/eps)
B = 1.0/Snorm
q[:,4] = A/(A+B)

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

#TODO - revisit this. Units don't line up with Ling's q7 definition.
#       Look at balance between shear stress and pressure grad in poisuille flow? 
#       Derive ratio from there? Surely must include viscosity in equation, and maybe d2u/dx2?
#du2dx = algs.gradient(U**2.0) 
#for k in range(0,3):
#    B += 0.5*rans_dsa.PointData['ro']*du2dx[:,k,k]*du2dx[:,k,k]

q[:,6] = algs.sqrt(A)/(algs.sqrt(A)+B)
#q[:,6] = algs.sqrt(A)/(algs.sqrt(A)+algs.abs(B))

# Feature 8: Vortex stretching
##############################
A = np.zeros(rans_nnode)
B = np.zeros(rans_nnode)

vortvec = algs.vorticity(U)

for j in range(0,3):
    for i in range(0,3):
        for k in range(0,3):
            A += vortvec[:,j]*J[:,i,j]*vortvec[:,k]*J[:,i,k]

B = Snorm

q[:,7] = algs.sqrt(A)/(algs.sqrt(A)+B)

# Feature 9: Marker of Gorle et al. (deviation from parallel shear flow)
########################################################################
A = np.zeros(rans_nnode)
B = np.zeros(rans_nnode)

for i in range(0,3):
    for j in range(0,3):
        A += U[:,i]*U[:,j]*J[:,i,j]

for n in range(0,3):
    for i in range(0,3):
        for j in range(0,3):
            for m in range(0,3):
                B += U[:,n]*U[:,n]*U[:,i]*J[:,i,j]*U[:,m]*J[:,m,j]
q[:,8] = algs.abs(A)/(algs.sqrt(B)+algs.abs(A))

# Feature 10: Ratio of convection to production of k
####################################################
delij = np.zeros_like(Sij)
for i in range(0,3): 
    delij[:,i,i] = 1.0
uiuj = (2.0/3.0)*tke*delij - 2.0*nu_t*Sij

dkdx  = algs.gradient(tke)

A = np.zeros(rans_nnode)
B = np.zeros(rans_nnode)

for i in range(0,3):
    A += U[:,i]*dkdx[:,i] 

for j in range(0,3):
    for l in range(0,3):
        B += uiuj[:,j,l]*Sij[:,j,l]

q[:,9] = A/(algs.abs(B)+algs.abs(A))

# Feature 11: Ratio of total Reynolds stresses to normal Reynolds stresses
##########################################################################
# Frob. norm of uiuj
A = algs.sum(uiuj**2,axis=1) # sum i axis
A = algs.sum(      A,axis=1) # sum previous summations i.e. along j axis
A = algs.sqrt(A)

B = tke

q[:,10] = A/(B + A)

# Feature 12: Cubic eddy viscosity comparision
##############################################
# Craft's CEVM uiuj
# C1 term
temp1  = np.zeros_like(uiuj)
temp2  = np.zeros_like(uiuj)
C1term = np.zeros_like(uiuj)
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            temp1[:,i,j] += Sij[:,i,j]*Sij[:,i,k]*Sij[:,j,k]
            for l in range(0,3):
                temp2[:,i,j] += Sij[:,i,j]*delij[:,i,j]*Sij[:,k,l]*Sij[:,k,l]
C1term = 4.0*C1*nu_t**2.0*(temp1 - temp2/3.0)/(Cmu*tke)

# C2 term
temp1  = np.zeros_like(uiuj)
temp2  = np.zeros_like(uiuj)
C2term = np.zeros_like(uiuj)
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            temp1[:,i,j] += Sij[:,i,j]*Oij[:,i,k]*Sij[:,k,j]
            temp2[:,i,j] += Sij[:,i,j]*Oij[:,j,k]*Sij[:,k,i]
C2term = 4.0*C2*nu_t**2.0*(temp1 + temp2)/(Cmu*tke)

# C3 term
temp1  = np.zeros_like(uiuj)
temp2  = np.zeros_like(uiuj)
C3term = np.zeros_like(uiuj)
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            temp1[:,i,j] += Sij[:,i,j]*Oij[:,i,k]*Oij[:,j,k]
            for l in range(0,3):
                temp2[:,i,j] += Oij[:,l,k]*Oij[:,l,k]*Sij[:,i,j]*delij[:,i,j]
C3term = 4.0*C3*nu_t**2.0*(temp1 - temp2/3.0)/(Cmu*tke)

# C4 term
temp1  = np.zeros_like(uiuj)
temp2  = np.zeros_like(uiuj)
C4term = np.zeros_like(uiuj)
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            for l in range(0,3):
                temp1[:,i,j] += Sij[:,i,j]*Sij[:,k,l]*Sij[:,k,i]*Oij[:,l,j]
                temp2[:,i,j] += Sij[:,i,j]*Sij[:,k,j]*Oij[:,l,i]*Sij[:,k,l]
C4term = 8.0*C4*nu_t**3.0*(temp1 + temp2)/(Cmu**2.0*tke**2.0)

# C5 term 
# No need as C5=0

# C6 term
temp1  = np.zeros_like(uiuj)
C6term = np.zeros_like(uiuj)
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            for l in range(0,3):
                temp1[:,i,j] += Sij[:,i,j]*Sij[:,i,j]*Sij[:,k,l]*Sij[:,k,l]
C6term = 8.0*C6*nu_t**3.0*temp1/(Cmu**2.0*tke**2.0)

# C7 term
temp1  = np.zeros_like(uiuj)
C7term = np.zeros_like(uiuj)
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            for l in range(0,3):
                temp1[:,i,j] += Sij[:,i,j]*Sij[:,i,j]*Oij[:,k,l]*Oij[:,k,l]
C7term = 8.0*C7*nu_t**3.0*temp1/(Cmu**2.0*tke**2.0)

# Add cubic terms to linear evm
cevm = C1term + C2term + C3term + C4term + C6term + C7term 
uiujcevmSij = uiuj*Sij + cevm

A = np.zeros(rans_nnode)
B = np.zeros(rans_nnode)

for i in range(0,3):
    for j in range(0,3):
        A += uiujcevmSij[:,i,j] - uiuj[:,i,j]*Sij[:,i,j]
        B += uiuj[:,i,j]*Sij[:,i,j]

q[:,11] = A / (algs.abs(A) + algs.abs(B))

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
    
