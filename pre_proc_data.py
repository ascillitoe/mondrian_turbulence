import numpy as np
import os
from vtk import *
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.numpy_interface import algorithms as algs
import vtki
import warnings
from internal_funcs import *

warnings.simplefilter("ignore", FutureWarning)

# User inputs
#############
RANS_fileloc  = '.'
RANS_filename = 'rans_h20.vtk'

LES_fileloc  = '.'
LES_filename = 'les_h20.vtu'

out_fileloc  = '.'
feat_filename = 'feat2.vtk'
err_filename  = 'err.vtk'

plot = False
solver = 'incomp'
#solver = 'comp'

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
q = np.empty([rans_nnode,nfeat])

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
q[:,0] = (Onorm - Snorm)/(Onorm + Snorm)

# clean up 
Snorm = algs.sqrt(Snorm) #revert to revert to real Snorm for use later
Onorm = algs.sqrt(Onorm) #revert to revert to real Onorm for use later

# Feature 2: Turbulence intensity
#################################
print('2: Turbulence intensity')
tke = rans_dsa.PointData['k'] 
UiUi = algs.mag(U)**2.0
q[:,1] = tke/(0.5*UiUi+tke)

# Feature 3: Turbulence Reynolds number
#######################################
print('3: Turbulence Reynolds number')
nu = rans_dsa.PointData['mu_l']/rans_dsa.PointData['ro']
q3 = (algs.sqrt(tke)*rans_dsa.PointData['d'])/(50.0*nu)
q[:,2] = algs.apply_dfunc(np.minimum, q3, 2.0)

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

q[:,3] = A/(algs.sqrt(B)+algs.abs(A))


# Feature 5: Ratio of turb time scale to mean strain time scale
###############################################################
print('5: Ratio of turb time scale to mean strain time scale')
A = 1.0/rans_dsa.PointData['w']  #Turbulent time scale (eps = k*w therefore also A = k/eps)
B = 1.0/Snorm
q[:,4] = A/(A+B)

# Feature 6: Viscosity ratio
############################
print('6: Viscosity ratio')
nu_t = rans_dsa.PointData['mu_t']/rans_dsa.PointData['ro']
q[:,5] = nu_t/(100.0*nu + nu_t)

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

q[:,6] = algs.sqrt(A)/(algs.sqrt(A)+B)
#q[:,6] = algs.sqrt(A)/(algs.sqrt(A)+algs.abs(B))

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

q[:,7] = algs.sqrt(A)/(algs.sqrt(A)+B)

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
q[:,8] = algs.abs(A)/(algs.sqrt(B)+algs.abs(A))

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

q[:,9] = A/(algs.abs(B)+algs.abs(A))

# Feature 11: Ratio of total Reynolds stresses to normal Reynolds stresses
##########################################################################
print('11: Ratio of total Reynolds stresses to normal Reynolds stresses')
# Frob. norm of uiuj
A = algs.sum(uiuj**2,axis=1) # sum i axis
A = algs.sum(      A,axis=1) # sum previous summations i.e. along j axis
A = algs.sqrt(A)

B = tke

q[:,10] = A/(B + A)

# Feature 12: Cubic eddy viscosity comparision
##############################################
print('12: Cubic eddy viscosity comparision')

# Add quadratic and cubic terms to linear evm
cevm_2nd, cevm_3rd = build_cevm(Sij,Oij)
uiujcevmSij = uiuj*Sij + (cevm_2nd/tke)*nu_t**2.0 + (cevm_3rd/tke**2.0)*nu_t**3.0

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

# Write to vtk file
###################
filename = os.path.join(out_fileloc, feat_filename)
rans_vtk.save(filename)

############################
# Generate LES error metrics
############################
print('\nProcessing LES data into error metrics')
print(  '--------------------------------------')
print('Error metric:')
nerror = 3
e_raw  = np.zeros([rans_nnode,nerror])
e_bool = np.zeros([rans_nnode,nerror],dtype=int)

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

nut = A/(B+1e-12)
e_raw[:,0] = nut

index = algs.where(nut<0.0)
e_bool[index,0] = 1

print('\tMean = ', algs.mean(e_bool[:,0]))

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
e_raw[:,1] = inv3

#index = algs.where(inv2>1.0/6.0)   #TODO - study what is best to use here. inv2, inv3, c1c etc... 
index = algs.where(algs.abs(inv3)>0.01)
e_bool[index,1] = 1

print('\tMean = ', algs.mean(e_bool[:,1]))

# Error metric 3: Non-linearity
###############################
print('3: Non-linearity')

# Build cevm equation in form A*nut**3 + B*nut**2 + C*nut + D = 0
B, A = build_cevm(Sij,Oij)
B = B/tke
A = A/tke**2.0

C = np.zeros_like(A)
D = np.zeros_like(A)
for i in range(0,3):
    for j in range(0,3):
        C += -2.0*Sij[:,i,j]*Sij[:,i,j]
        D += (2.0/3.0)*tke*Sij[:,i,j]*delij[:,i,j] - uiuj[:,i,j]*Sij[:,i,j]


print('\tMean = ', algs.mean(e_bool[:,2]))


# Store errors in vtk obj
###########################
les_vtk.point_arrays['raw'] = e_raw
les_vtk.point_arrays['boolean'] = e_bool

# Write to vtk file
###################
filename = os.path.join(out_fileloc, err_filename)
les_vtk.save(filename)

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
    
