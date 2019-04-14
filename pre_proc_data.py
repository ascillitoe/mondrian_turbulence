import os

import numpy as np
import pandas as pd

from vtk.numpy_interface import dataset_adapter as dsa
from vtk.numpy_interface import algorithms as algs
import vtki

from preproc_funcs import *

import warnings
warnings.simplefilter("ignore", FutureWarning)

# User inputs
#############
RANS_fileloc  = '.'
RANS_filename = 'rans_h20.vtk'

LES_fileloc  = '.'
LES_filename = 'les_h20.vtu'

debug_fileloc  = '.'
data_fileloc   = '.'
casename = 'test'

plot = False
solver = 'incomp'
#solver = 'comp'

xclip_min = [-0.1,0,-99]
xclip_max = [0.506,0.142082,99]

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
if (solver=='comp'):
    rans_vtk.rename_scalar('Momentum','roU') #TODO - auto check if velocities or Momentum in vtk file
    rans_vtk.rename_scalar('Pressure' ,'p') #TODO - Check if Energy or pressure etc
    rans_vtk.rename_scalar('Density' ,'ro')
    rans_vtk.rename_scalar('TKE' ,'k')
    rans_vtk.rename_scalar('Omega' ,'w') #TODO - Read in eps if that is saved instead
    rans_vtk.rename_scalar('Laminar_Viscosity','mu_l') #TODO - if this doesn't exist auto calc from sutherlands
    rans_vtk.rename_scalar('Eddy_Viscosity','mu_t') #TODO - if this doesn't exist calc from k and w
    rans_vtk.rename_scalar('Wall_Distance','d') #TODO - if this doesn't exist calc 
    rans_vtk.point_arrays['U'] = rans_vtk.point_arrays['roU']/np.array([rans_vtk.point_arrays['ro'],]*3).transpose()
elif (solver=='incomp'):
    rans_vtk.rename_scalar('Velocity','U')
    rans_vtk.rename_scalar('Pressure' ,'p') #TODO - Check if Energy or pressure etc
    rans_vtk.rename_scalar('TKE' ,'k')
    rans_vtk.rename_scalar('Omega' ,'w') #TODO - Read in eps if that is saved instead
    rans_vtk.rename_scalar('Laminar_Viscosity','mu_l') #TODO - if this doesn't exist auto calc from sutherlands
    rans_vtk.rename_scalar('Eddy_Viscosity','mu_t') #TODO - if this doesn't exist calc from k and w
    rans_vtk.rename_scalar('Wall_Distance','d') #TODO - if this doesn't exist calc 
    rans_vtk.point_arrays['ro'] = np.ones(rans_nnode)*1.0 
else:
    sys.exit('Error: solver should equal comp or incomp')

print('List of RANS data fields:\n', rans_vtk.scalar_names)

# Remove viscous wall d=0 points to prevent division by zero in feature construction
print('Removing viscous wall (d=0) nodes from mesh')
rans_vtk = rans_vtk.threshold([1e-12,1e99],scalars='d')
print('Number of nodes extracted = ', rans_nnode - rans_vtk.number_of_points)
rans_nnode = rans_vtk.number_of_points

# Clip mesh to given ranges
print('Clipping mesh to range: ', xclip_min, ' to ', xclip_max)
rans_vtk = rans_vtk.clip(normal='x', origin=xclip_min,invert=False)
rans_vtk = rans_vtk.clip(normal='x', origin=xclip_max,invert=True )
rans_vtk = rans_vtk.clip(normal='y', origin=xclip_min,invert=False)
rans_vtk = rans_vtk.clip(normal='y', origin=xclip_max,invert=True )
rans_vtk = rans_vtk.clip(normal='z', origin=xclip_min,invert=False)
rans_vtk = rans_vtk.clip(normal='z', origin=xclip_max,invert=True )
print('Number of nodes clipped = ', rans_nnode - rans_vtk.number_of_points)

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
print('Interpolating LES data onto RANS mesh')
les_vtk = rans_vtk.interpolate(les_vtk)
les_nnode = les_vtk.number_of_points
les_ncell = les_vtk.number_of_cells

# Check nnode and nncell for LES and RANS now match
if (rans_nnode!=les_nnode): quit('Warning: rans_nnode != les_nnode... Interpolation failed')
if (rans_ncell!=les_ncell): quit('Warning: rans_ncell != les_ncell... Interpolation failed')

# Wrap vtki objects in dsa wrapper
# NOTE - NO VTK OPERATIONS SUCH AS INTERP, SLICING ETC BEYOND THIS POINT
print('Wrapping vtk objects in dsa wrappers')
rans_dsa = dsa.WrapDataObject(rans_vtk)
les_dsa  = dsa.WrapDataObject(les_vtk)

##############################################
# Process RANS data to generate feature vector
##############################################
print('\nProcessing RANS data into features')
print(  '----------------------------------')
print('Feature:')
nfeat = 12
feat = 0
q = np.empty([rans_nnode,nfeat])
feature_labels = np.empty(nfeat, dtype='object')

# Feature 1: non-dim Q-criterion
################################
print('1: non-dim Q-criterion...')
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
q[:,feat] = (Onorm - Snorm)/(Onorm + Snorm)
feature_labels[feat] = 'Q-Criterion'
feat += 1

# clean up 
Snorm = algs.sqrt(Snorm) #revert to revert to real Snorm for use later
Onorm = algs.sqrt(Onorm) #revert to revert to real Onorm for use later

# Feature 2: Turbulence intensity
#################################
print('2: Turbulence intensity')
tke = rans_dsa.PointData['k'] 
UiUi = algs.mag(U)**2.0
q[:,feat] = tke/(0.5*UiUi+tke)
feature_labels[feat] = 'Turbulence intensity'
feat += 1

# Feature 3: Turbulence Reynolds number
#######################################
print('3: Turbulence Reynolds number')
nu = rans_dsa.PointData['mu_l']/rans_dsa.PointData['ro']
q3 = (algs.sqrt(tke)*rans_dsa.PointData['d'])/(50.0*nu)
q[:,feat] = algs.apply_dfunc(np.minimum, q3, 2.0)
feature_labels[feat] = 'Turbulence Re'
feat += 1

# Feature 4: Pressure gradient along streamline
###############################################
print('4: Pressure gradient along streamline')
A = np.zeros(rans_nnode)
B = np.zeros(rans_nnode)

dpdx  = algs.gradient(rans_dsa.PointData['p'])

for k in range(0,3):
    A += U[:,k]*dpdx[:,k] 

for i in range(0,3):
    for j in range(0,3):
        B += U[:,i]*U[:,i]*dpdx[:,j]*dpdx[:,j]

q[:,feat] = A/(algs.sqrt(B)+algs.abs(A))
feature_labels[feat] = 'Pgrad along streamline'
feat += 1

# Feature 5: Ratio of turb time scale to mean strain time scale
###############################################################
print('5: Ratio of turb time scale to mean strain time scale')
A = 1.0/rans_dsa.PointData['w']  #Turbulent time scale (eps = k*w therefore also A = k/eps)
B = 1.0/Snorm
q[:,feat] = A/(A+B)
feature_labels[feat] = 'turb/strain time-scale'
feat += 1

# Feature 6: Viscosity ratio
############################
print('6: Viscosity ratio')
nu_t = rans_dsa.PointData['mu_t']/rans_dsa.PointData['ro']
q[:,feat] = nu_t/(100.0*nu + nu_t)
feature_labels[feat] = 'Viscosity ratio'
feat += 1

# Feature 7: Ratio of pressure normal stresses to normal shear stresses
#######################################################################
print('7: Ratio of pressure normal stresses to normal shear stresses')
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

q[:,feat] = algs.sqrt(A)/(algs.sqrt(A)+B)
#q[:,6] = algs.sqrt(A)/(algs.sqrt(A)+algs.abs(B))
feature_labels[feat] = 'Pressure/shear stresses'
feat += 1

# Feature 8: Vortex stretching
##############################
print('8: Vortex stretching')
A = np.zeros(rans_nnode)
B = np.zeros(rans_nnode)

vortvec = algs.vorticity(U)

for j in range(0,3):
    for i in range(0,3):
        for k in range(0,3):
            A += vortvec[:,j]*J[:,i,j]*vortvec[:,k]*J[:,i,k]

B = Snorm

q[:,feat] = algs.sqrt(A)/(algs.sqrt(A)+B)
feature_labels[feat] = 'Vortex stretching'
feat += 1

# Feature 9: Marker of Gorle et al. (deviation from parallel shear flow)
########################################################################
print('9: Marker of Gorle et al. (deviation from parallel shear flow)')
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
q[:,feat] = algs.abs(A)/(algs.sqrt(B)+algs.abs(A))
feature_labels[feat] = 'Deviation from parallel shear'
feat += 1

# Feature 10: Ratio of convection to production of k
####################################################
print('10: Ratio of convection to production of k')
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

q[:,feat] = A/(algs.abs(B)+algs.abs(A))
feature_labels[feat] = 'Convection/production of k'
feat += 1

# Feature 11: Ratio of total Reynolds stresses to normal Reynolds stresses
##########################################################################
print('11: Ratio of total Reynolds stresses to normal Reynolds stresses')
# Frob. norm of uiuj
A = algs.sum(uiuj**2,axis=1) # sum i axis
A = algs.sum(      A,axis=1) # sum previous summations i.e. along j axis
A = algs.sqrt(A)

B = tke

q[:,feat] = A/(B + A)
feature_labels[feat] = 'total/normal stresses'
feat += 1

# Feature 12: Cubic eddy viscosity comparision
##############################################
print('12: Cubic eddy viscosity comparision')

# Add quadratic and cubic terms to linear evm
cevm_2nd, cevm_3rd = build_cevm(Sij,Oij)

uiujSij = np.zeros(rans_nnode)
for i in range(0,3):
    for j in range(0,3):
        uiujSij += uiuj[:,i,j]*Sij[:,i,j]

uiujcevmSij = uiujSij + (cevm_2nd/tke)*nu_t**2.0 + (cevm_3rd/tke**2.0)*nu_t**3.0

q[:,feat] = (uiujcevmSij-uiujSij) / (uiujcevmSij+uiujSij)
feature_labels[feat] = 'CEV comparison'
feat += 1

# Store features in vtk obj
###########################
rans_vtk.point_arrays['q'] = q

############################
# Generate LES error metrics
############################
print('\nProcessing LES data into error metrics')
print(  '--------------------------------------')
print('Error metric:')
nerr = 3
err = 0
e_raw  = np.zeros([rans_nnode,nerr])
e_bool = np.zeros([rans_nnode,nerr],dtype=int)
error_labels = np.empty(nerr, dtype='object')

# Copy Reynolds stresses to tensor
uiuj = np.zeros([les_nnode,3,3])
uiuj[:,0,0] = les_dsa.PointData['uu']
uiuj[:,1,1] = les_dsa.PointData['vv']
uiuj[:,2,2] = les_dsa.PointData['ww']
uiuj[:,0,1] = les_dsa.PointData['uv']
#uiuj[:,0,2] = 
#uiuj[:,1,2] = 
uiuj[:,1,0] = uiuj[:,0,1]
uiuj[:,2,0] = uiuj[:,0,2]
uiuj[:,2,1] = uiuj[:,1,2]

# resolved TKE
tke = 0.5*(uiuj[:,0,0]+uiuj[:,1,1]+uiuj[:,2,2])

# Velocity vector
U = algs.make_vector(les_dsa.PointData['U'],les_dsa.PointData['V'],les_dsa.PointData['W'])

# Velocity gradient tensor and its transpose
# J[:,i-1,j-1] is dUidxj
# Jt[:,i-1,j-1] is dUjdxi
Jt = algs.gradient(U)        # Jt is this one as algs uses j,i ordering
J  = algs.apply_dfunc(np.transpose,Jt,(0,2,1))

# Strain and vorticity tensors
Sij = 0.5*(J+Jt)
Oij = 0.5*(J-Jt)

# Error metric 1: Negative eddy viscosity
#########################################
print('1: Negative eddy viscosity')
A = np.zeros(les_nnode)
B = np.zeros(les_nnode)

for i in range(0,3):
    for j in range(0,3):
        A += -uiuj[:,i,j]*Sij[:,i,j] + (2.0/3.0)*tke*delij[:,i,j]*Sij[:,i,j]
        B += 2.0*Sij[:,i,j]*Sij[:,i,j]

nu_t = A/(B+1e-12)
e_raw[:,err] = nu_t

index = algs.where(nu_t<0.0)
e_bool[index,err] = 1
error_labels[err] = 'Negative eddy viscosity'
err += 1

# Error metric 2: Reynolds stress aniostropy
############################################
print('2: Reynolds stress anisotropy')
aij  = np.zeros([les_nnode,3,3])
inv2 = np.zeros(les_nnode)
inv3 = np.zeros(les_nnode)

for i in range(0,3):
    for j in range(0,3):
        aij[:,i,j] = uiuj[:,i,j]/(2.0*tke+1e-12) - delij[:,i,j]/3.0

for i in range(0,3):
    for j in range(0,3):
        inv2 += aij[:,i,j]*aij[:,j,i]
        for n in range(0,3):
            inv3 += aij[:,i,j]*aij[:,i,n]*aij[:,j,n]

#e_raw[:,1] = inv2
e_raw[:,err] = inv3

#index = algs.where(inv2>1.0/6.0)   #TODO - study what is best to use here. inv2, inv3, c1c etc... 
index = algs.where(algs.abs(inv3)>0.01)
e_bool[index,err] = 1
error_labels[err] = 'Stress anisotropy'
err += 1

# Error metric 3: Non-linearity
###############################
print('3: Non-linearity')

# Build cevm equation in form A*nut**3 + B*nut**2 + C*nut + D = 0
B, A = build_cevm(Sij,Oij)
B = B/(tke      +1e-12)
A = A/(tke**2.0 +1e-12)

C = np.zeros_like(A)
D = np.zeros_like(A)
for i in range(0,3):
    for j in range(0,3):
        C += -2.0*Sij[:,i,j]*Sij[:,i,j]
        D += (2.0/3.0)*tke*Sij[:,i,j]*delij[:,i,j] - uiuj[:,i,j]*Sij[:,i,j]

nu_t_cevm = np.empty_like(nu_t)
for i in range(0,les_nnode):
    # Find the roots of the cubic equation (i.e. potential values for nu_t_cevm)
    roots = np.roots([A[i],B[i],C[i],D[i]])
    roots_orig = roots

    # Remove complex solutions (with imaginary part > a small number, to allow for numerical error)
    #roots = roots.real[abs(roots.imag)<1e-5]  #NOTE - Matches nu_t much better without this?!

    # Out of remaining solutions(s), pick one that is closest to linear nu_t
    nu_t_cevm[i] = roots.real[np.argmin( np.abs(roots - np.full(roots.size,nu_t[i])) )]

normdiff = algs.abs(nu_t_cevm - nu_t) / (algs.abs(nu_t_cevm) + algs.abs(nu_t) + 1e-12)
e_raw[:,err] = nu_t_cevm

index = algs.where(normdiff>0.15)
e_bool[index,err] = 1
error_labels[err] = 'Non-linearity'
err += 1

# Store errors in vtk obj
###########################
les_vtk.point_arrays['raw'] = e_raw
les_vtk.point_arrays['boolean'] = e_bool

# Write to vtk files
####################
print('\nWriting data to vtk files')
print(  '-------------------------')
filename = casename + '_feat.vtk'
filename = os.path.join(debug_fileloc, filename)
print('Writing to features to ', filename)
rans_vtk.save(filename)

filename = casename + '_err.vtk'
filename = os.path.join(debug_fileloc, filename)
print('Writing errors to ', filename)
les_vtk.save(filename)

# Store features and targets (error metrics) in pandas dataframe, then save to disk #TODO - option to keep in memory when this script used as a function
print('\nConvert data to pandas dataframes')
print(  '---------------------------------')

# Put features in pandas dataframe and report stats
dataset = pd.DataFrame(q, columns=feature_labels)
feat_descript = dataset.describe()
print('\nStatistics of features:' )
print(feat_descript)
filename = casename + '_feature_stats.csv'
filename = os.path.join(debug_fileloc, filename)
print('Writing feature stats to ', filename)
feat_descript.to_csv(filename)

# Put errors in pandas dataframe and report stats
temp_dataset = pd.DataFrame(e_bool, columns=error_labels)
err_descript = temp_dataset.describe()
print('\nStatistics of errors:')
print(err_descript)
filename = casename + '_error_stats.csv' 
filename = os.path.join(debug_fileloc, filename)
print('Writing error stats to ', filename)
err_descript.to_csv(filename)

# Combining dataframes into one and writing to file
dataset = pd.concat([dataset, temp_dataset], axis=1)
filename = casename + '_dataset.csv'
filename = os.path.join(data_fileloc, filename)
print('\nWriting final pandas data to ', filename)
dataset.to_csv(filename,index=False)

print('\n-----------------------')
print('Finished pre-processing')
print('-----------------------')

#####################
# Plot stuff to check
#####################
if (plot):
    print('\n Plotting...')
    plotter = vtki.Plotter()
    sargs = dict(interactive=True,height=0.25,title_font_size=12, label_font_size=11,shadow=True, n_labels=5, italic=True, fmt='%.1f',font_family='arial',vertical=False)
    rans_vtk.point_arrays['plot'] = q3
    clims = [np.min(q3), np.max(q3)]
    print(clims)
    plotter.add_mesh(rans_vtk,scalars='plot',rng=clims,scalar_bar_args=sargs)
    plotter.view_xy()
    #plotter.add_mesh(data2,scalars=J2[:,1,0],show_scalar_bar=True,scalar_bar_args=sargs,rng=[-100,100]) #can plot np array directly but colour bar doesn't work...
    plotter.show()
    
