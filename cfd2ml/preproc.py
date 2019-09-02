import numpy as np
import pandas as pd

import os

import vista
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa

from cfd2ml.base import CaseData

def preproc_RANS_based(json,type):
    from cfd2ml.preproc import preproc_RANS_and_HiFi
    from cfd2ml.utilities import convert_rans_fields, convert_hifi_fields

    print('\n-----------------------')
    print('Started pre-processing')
    if (type==1):
        print('Type 1')
    elif (type==2):
        print('Type 2')
    print('-----------------------')

    # Create output dir if needed
    outdir = json['Output directory']
    os.makedirs(outdir, exist_ok=True)

   # Loop through cases, perform preprocessing of CFD data
    for case in json['Cases']:
        id = case['Case ID']
        name = case['Name']

        print('\n**********************************************')
        print('Case %s: %s' %(id,name) )
        print('**********************************************')

        RANSfile = case['RANS file']
        HiFifile = case['HiFi file']
        arraynames = case['HiFi array names']
        
        if (("options" in case)==True):
            options = case['options']

        # Read data
        X_data = CaseData(name)
        Y_data = CaseData(name)

        print('Reading X data from vtk file: ', RANSfile)
        X_data.ReadVTK(RANSfile)
        print('Reading Y data from vtk file: ', HiFifile)
        Y_data.ReadVTK(HiFifile)

        # Convert RANS and HiFi array names
        X_data.vtk = convert_rans_fields(X_data.vtk)
        Y_data.vtk = convert_hifi_fields(Y_data.vtk,arraynames)

        # Run preproc
        X_data, Y_data = preproc_RANS_and_HiFi(X_data, Y_data, type, **options)

        # Write data
        X_data.Write(os.path.join(outdir, id + '_X')) 
        Y_data.Write(os.path.join(outdir, id + '_Y'))

    print('\n-----------------------')
    print('Finished pre-processing')
    print('-----------------------')

def preproc_RANS_and_HiFi(q_data, e_data, type, clip=None,comp=False):
   
    #################################
    # Initial processing of RANS data
    #################################
    print('\nInitial processing of RANS data')
    print(  '------------')
    
    rans_vtk = q_data.vtk
    
    # Get basic info about mesh
    rans_nnode = rans_vtk.number_of_points
    rans_ncell = rans_vtk.number_of_cells
    print('Number of nodes = ', rans_nnode)
    print('Number of cells = ', rans_ncell)
   
    # Remove viscous wall d=0 points to prevent division by zero in feature construction
    print('Removing viscous wall (d=0) nodes from mesh')
    rans_vtk = rans_vtk.threshold([1e-12,1e99],scalars='d')
    print('Number of nodes extracted = ', rans_nnode - rans_vtk.number_of_points)
    rans_nnode = rans_vtk.number_of_points
    
    # Clip mesh to given ranges   # TODO - move this to after gradient calcs!
    if (clip is not None):
        xclip_min = clip[0:3]
        xclip_max = clip[3:6]
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
    
    # Initial processing of HiFi data
    ################################
    print('\nInitial processing of HiFi data')
    
    les_vtk = e_data.vtk
    
    # Get basic info about mesh
    les_nnode = les_vtk.number_of_points
    les_ncell = les_vtk.number_of_cells
    print('Number of nodes = ', les_nnode)
    print('Number of cells = ', les_ncell)
        
    ##############################################
    # Process RANS data to generate feature vector
    ##############################################
    print('\nProcessing RANS data into features')
    print(  '----------------------------------')

    if (type==1):
        q, feature_labels = make_features(rans_vtk)
    elif (type==2):
        q, feature_labels = make_features_inv(rans_vtk)

    # Store features in vtk obj
    ###########################
    rans_vtk.point_arrays['q'] = q
    
    ############################
    # Generate HiFi error metrics
    ############################
    print('\nProcessing HiFi data into error metrics')
    print(  '--------------------------------------')

    e_raw, e_bool, error_labels = make_errors(les_vtk)
 
    # Store errors in vtk obj
    les_vtk.point_arrays['raw'] = e_raw
    les_vtk.point_arrays['boolean'] = e_bool
     
    #####################################
    # Interpolate HiFi data onto RANS mesh (do after metrics are generated so Sij etc from fine mesh)
    #####################################
    print('Interpolating HiFi data onto RANS mesh')
    les_vtk = rans_vtk.sample(les_vtk,pass_point_arrays=False)
    les_nnode = les_vtk.number_of_points
    les_ncell = les_vtk.number_of_cells
    
    # Check nnode and nncell for HiFi and RANS now match
    if (rans_nnode!=les_nnode): quit('********** Warning: rans_nnode != les_nnode... Interpolation failed **********')
    if (rans_ncell!=les_ncell): quit('********** Warning: rans_ncell != les_ncell... Interpolation failed **********')

    e_bool = les_vtk.point_arrays['boolean']

    #############################
    # Final checks and store data
    #############################
    # pandas dataframes
    q_data.pd = pd.DataFrame(q, columns=feature_labels)
    e_data.pd = pd.DataFrame(e_bool, columns=error_labels)

    # Check for NaN's and flag them (by saving to q_nan in vtk with value=99)
    q_nansum =q_data.pd.isnull().sum().sum() 
    if (q_nansum > 0):
        print('********** Warning %d NaNs in X data **********' %(q_nansum))
        q_nans = q_data.pd.fillna(99)
        rans_vtk.point_arrays['q_nan'] = q_nans

    df = pd.DataFrame(e_raw)
    e_nansum = df.isnull().sum().sum() 
    if (e_nansum > 0):
        print('********** Warning %d NaNs in Y data **********' %(e_nansum))
        e_nans = e_data.pd.fillna(99)
        les_vtk.point_arrays['e_raw_nan'] = e_nans

    # Put new vtk data back into vtk objects
    q_data.vtk = rans_vtk
    e_data.vtk = les_vtk
   

    return q_data, e_data


def preproc_RANS(q_data, e_data, clip=None, comp=False):

    ###################
    # Read in RANS data
    ###################
    print('\nInitial processing of RANS data')
    print(  '------------')
    
    rans_vtk = q_data.vtk

    # Get basic info about mesh
    rans_nnode = rans_vtk.number_of_points
    rans_ncell = rans_vtk.number_of_cells
    print('Number of nodes = ', rans_nnode)
    print('Number of cells = ', rans_ncell)
   
    # Remove viscous wall d=0 points to prevent division by zero in feature construction
    print('Removing viscous wall (d=0) nodes from mesh')
    rans_vtk = rans_vtk.threshold([1e-12,1e99],scalars='d')
    print('Number of nodes extracted = ', rans_nnode - rans_vtk.number_of_points)
    rans_nnode = rans_vtk.number_of_points
    
    # Clip mesh to given ranges
    if (clip is not None):
        xclip_min = clip[0:3]
        xclip_max = clip[3:6]
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
    
    # Process RANS data to generate feature vector
    ##############################################
    print('\nProcessing RANS data into features')
    print(  '----------------------------------')

    q, feature_labels = make_features(rans_vtk)
    
    # Store features in vtk obj
    ###########################
    rans_vtk.point_arrays['q'] = q
 
    # Store features in pandas dataframe
    print('\nSaving data in pandas dataframe')
    q_data.pd = pd.DataFrame(q, columns=feature_labels)

    # Put new vtk data back into vtk object
    q_data.vtk = rans_vtk

    print('\n-----------------------')
    print('Finished pre-processing')
    print('-----------------------')

    return q_data


def make_features(rans_vtk):
    from cfd2ml.utilities import build_cevm

    small = np.finfo(float).tiny

    rans_nnode = rans_vtk.number_of_points

    delij = np.zeros([rans_nnode,3,3])
    for i in range(0,3): 
        delij[:,i,i] = 1.0

    # Wrap vista object in dsa wrapper
    rans_dsa = dsa.WrapDataObject(rans_vtk)

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
    q[:,feat] = (Onorm - Snorm)/(Onorm + Snorm + small)
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
    q[:,feat] = tke/(0.5*UiUi+tke+small)
    feature_labels[feat] = 'Turbulence intensity'
    feat += 1
    
    # Feature 3: Turbulence Reynolds number
    #######################################
    print('3: Turbulence Reynolds number')
    nu = rans_dsa.PointData['mu_l']/rans_dsa.PointData['ro']
    Red = (algs.sqrt(tke)*rans_dsa.PointData['d'])/(50.0*nu)
    q[:,feat] = algs.apply_dfunc(np.minimum, Red, 2.0)
    #Red = 0.09**0.25*algs.sqrt(tke)*rans_dsa.PointData['d']/nu
    #q[:,feat] = algs.apply_dfunc(np.minimum, Red, 100.0)
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
    
    q[:,feat] = A/(algs.sqrt(B)+algs.abs(A)+small)
    feature_labels[feat] = 'Pgrad along streamline'
    feat += 1
    
    # Feature 5: Ratio of turb time scale to mean strain time scale
    ###############################################################
    print('5: Ratio of turb time scale to mean strain time scale')
    A = 1.0/rans_dsa.PointData['w']  #Turbulent time scale (eps = k*w therefore also A = k/eps)
    B = 1.0/Snorm
    q[:,feat] = A/(A+B+small)
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
    
    q[:,feat] = algs.sqrt(A)/(algs.sqrt(A)+B+small)
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
    
    q[:,feat] = algs.sqrt(A)/(algs.sqrt(A)+B+small)
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
    q[:,feat] = algs.abs(A)/(algs.sqrt(B)+algs.abs(A)+small)
    feature_labels[feat] = 'Deviation from parallel shear'
    feat += 1
    
    # Feature 10: Ratio of convection to production of k
    ####################################################
    print('10: Ratio of convection to production of k')
    uiuj = (2.0/3.0)*tke*delij - 2.0*nu_t*Sij
    
    dkdx  = algs.gradient(tke)
    
    A = np.zeros(rans_nnode)
    B = np.zeros(rans_nnode)
    
    for i in range(0,3):
        A += U[:,i]*dkdx[:,i] 
    
    for j in range(0,3):
        for l in range(0,3):
            B += uiuj[:,j,l]*Sij[:,j,l]
    
    q[:,feat] = A/(algs.abs(B)+algs.abs(A)+small)
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
    
    q[:,feat] = A/(B + A + small)
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
    
    q[:,feat] = (uiujcevmSij-uiujSij) / (uiujcevmSij+uiujSij + small)
    feature_labels[feat] = 'CEV comparison'
    feat += 1

    return q, feature_labels

def make_errors(les_vtk):
    from tqdm import tqdm

    small = np.finfo(float).tiny

    les_nnode = les_vtk.number_of_points

    delij = np.zeros([les_nnode,3,3])
    for i in range(0,3): 
        delij[:,i,i] = 1.0

    # Wrap vista object in dsa wrapper
    les_dsa = dsa.WrapDataObject(les_vtk)

    print('Error metric:')
    nerr = 10
    err = 0
    e_raw  = np.zeros([les_nnode,nerr])
    e_bool = np.zeros([les_nnode,nerr],dtype=int)
    error_labels = np.empty(nerr, dtype='object')
    
    # Copy Reynolds stresses to tensor
    uiuj = np.zeros([les_nnode,3,3])
    uiuj[:,0,0] = les_dsa.PointData['uu']
    uiuj[:,1,1] = les_dsa.PointData['vv']
    uiuj[:,2,2] = les_dsa.PointData['ww']
    uiuj[:,0,1] = les_dsa.PointData['uv']
    uiuj[:,0,2] = les_dsa.PointData['uw']
    uiuj[:,1,2] = les_dsa.PointData['vw']
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
    
    Str = algs.sqrt(B) # magnitude of Sij strain tensor (used later)
    nu_t = A/(B+small)
    e_raw[:,err] = nu_t
    
    index = algs.where(nu_t<0.0)
    e_bool[index,err] = 1
    error_labels[err] = 'Negative eddy viscosity'
    err += 1
    
    # Error metric 2/3: 2nd and 3rd invariants of aij
    #################################################
    print('2/3: Reynolds stress anisotropy, inv2 and inv3')
    aij  = np.zeros([les_nnode,3,3])
    inv2 = np.zeros(les_nnode)
    inv3 = np.zeros(les_nnode)
    
    for i in range(0,3):
        for j in range(0,3):
            aij[:,i,j] = uiuj[:,i,j]/(2.0*tke+small) - delij[:,i,j]/3.0
    
    for i in range(0,3):
        for j in range(0,3):
            inv2 += aij[:,i,j]*aij[:,j,i]
            for n in range(0,3):
                inv3 += aij[:,i,j]*aij[:,i,n]*aij[:,j,n]
    
    e_raw[:,err] = inv2
    index = algs.where(inv2>1.0/6.0)
    e_bool[index,err] = 1
    error_labels[err] = 'Stress anisotropy 2nd inv'
    err += 1

    e_raw[:,err] = inv3
    index = algs.where(inv3>0.005)
    e_bool[index,err] = 1
    error_labels[err] = 'Stress anisotropy 3rd inv'
    err += 1

    # Error metric 4: Negative Pk
    ############################################
    print('4: Negative Pk')
    A = np.zeros(les_nnode)
    for i in range(0,3):
        for j in range(0,3):
            A[:] += (-uiuj[:,i,j] * J[:,i,j])

    e_raw[:,err] = A
    index = algs.where(A<0.0)

    e_bool[index,err] = 1
    error_labels[err] = 'Negative Pk'
    err += 1

    # Error metric 5: 2-eqn Cmu constant
    ############################################
    print('5: 2-equation Cmu constant')
    A = np.zeros(les_nnode)
    for i in range(0,3):
        for j in range(0,3):
            A[:] += aij[:,i,j]*Sij[:,i,j]

    Cmu = nu_t**2.0*(Str/(tke+small))**2.0
    e_raw[:,err] = Cmu
    allow_err = 0.25 #i.e. 10% err
    Cmu_dist = algs.abs(Cmu - 0.09)
#    index = algs.where(Cmu_dist>allow_err*0.09)
    index = algs.where(Cmu>1.1*0.09)
    e_bool[index,err] = 1
    error_labels[err] = 'Cmu != 0.09'
    err += 1

    ab = ((uiuj[:,1,1]-uiuj[:,0,0])*U[:,0]*U[:,1] + uiuj[:,0,1]*(U[:,0]**2-U[:,1]**2))/(U[:,0]**2+U[:,1]**2)
    e_raw[:,err] = ab

#    # Error metric 3: Non-linearity
#    ###############################
#    print('3: Non-linearity')
#    
#    # Build cevm equation in form A*nut**3 + B*nut**2 + C*nut + D = 0
#    B, A = build_cevm(Sij,Oij)
#    B = B/(tke      +1e-12)
#    A = A/(tke**2.0 +1e-12)
#    
#    C = np.zeros_like(A)
#    D = np.zeros_like(A)
#    for i in range(0,3):
#        for j in range(0,3):
#            C += -2.0*Sij[:,i,j]*Sij[:,i,j]
#            D += (2.0/3.0)*tke*Sij[:,i,j]*delij[:,i,j] - uiuj[:,i,j]*Sij[:,i,j]
#    
#    nu_t_cevm = np.empty_like(nu_t)
#    for i in tqdm(range(0,les_nnode)):
#        # Find the roots of the cubic equation (i.e. potential values for nu_t_cevm)
#        roots = np.roots([A[i],B[i],C[i],D[i]])
#        roots_orig = roots
#    
#        # Remove complex solutions (with imaginary part > a small number, to allow for numerical error)
#        #roots = roots.real[abs(roots.imag)<1e-5]  #NOTE - Matches nu_t much better without this?!
#    
#        # Out of remaining solutions(s), pick one that is closest to linear nu_t
#        if(roots.size==0):
#            nu_t_cevm[i] = nu_t[i]
#        else:
#            nu_t_cevm[i] = roots.real[np.argmin( np.abs(roots - np.full(roots.size,nu_t[i])) )]
#    
#    normdiff = algs.abs(nu_t_cevm - nu_t) / (algs.abs(nu_t_cevm) + algs.abs(nu_t) + 1e-12)
#    e_raw[:,err] = nu_t_cevm
#    
#    index = algs.where(normdiff>0.15)
#    e_bool[index,err] = 1
#    error_labels[err] = 'Non-linearity'
#    err += 1

    return e_raw, e_bool, error_labels


def make_features_inv(rans_vtk):
    from cfd2ml.utilities import eijk

    small = np.finfo(float).tiny

    rans_nnode = rans_vtk.number_of_points

    # Wrap vista object in dsa wrapper
    rans_dsa = dsa.WrapDataObject(rans_vtk)

    print('Feature:')
    nfeat = 50
    q = np.empty([rans_nnode,nfeat])
    feature_labels = np.empty(nfeat, dtype='object')
    
    # Non-dim strain and vorticity
    ##############################
    print('Constructing non-dim strain and vorticity tensor')
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
    Snorm = algs.sqrt(Snorm) 
    Onorm = algs.sqrt(Onorm) 

    # Non-dim Sij by eps/k
    tke = rans_dsa.PointData['k']
    eps = rans_dsa.PointData['w']*tke
    Sij_h = Sij / (eps/tke)

    # Non-dim Oij by eps/k
    Oij_h = Oij / (eps/tke)

    # Non-dim pressure gradient
    ###########################
    print('Constructing non-dim pressure gradient')
    dpdx  = algs.gradient(rans_dsa.PointData['p'])
    ro = rans_dsa.PointData['ro']
    DUDt = U[:,0]*J[:,:,0] + U[:,1]*J[:,:,1] + U[:,2]*J[:,:,2]
    dpdx_h = dpdx / ro*algs.mag(DUDt)

    # Non-dim tke gradient
    ######################
    print('Constructing non-dim tke gradient')
    dkdx  = algs.gradient(tke)
    dkdx_h = dkdx / (eps/algs.sqrt(tke)) 

    # Transform dpdx into ani-symmetric tensor Ap=-I x dpdx
    Ap = np.zeros([rans_nnode,3,3])
    I = np.eye(3)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    Ap[:,a,b] -= eijk(b,c,d)*I[a,c]*dpdx_h[:,d]


    # Transform dkdx into ani-symmetric tensor Ak=-I x dkdx
    Ak = np.zeros([rans_nnode,3,3])
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    Ak[:,a,b] -= eijk(b,c,d)*I[a,c]*dkdx_h[:,d]

    # Construct all invariant bases
    ###############################
    # Use numpy matmul to construct S^2, S^3 etc as we use these alot 
    # (matmul can be used as for arrays of dim>2 as "it is treated as a stack of matrices residing in the last two indexes and is broadcast accordingly")
    S = Sij_h
    O = Oij_h
    S2 = np.matmul(S,S)
    S3 = np.matmul(S2,S)
    O2 = np.matmul(O,O)
    Ap2 = np.matmul(Ap,Ap)
    Ak2 = np.matmul(Ak,Ak)

    # 1-2
    q[:,0] = algs.trace(S2)
    feature_labels[0] = 'S2'
    q[:,1] = algs.trace(S3)
    feature_labels[1] = 'S3'
    # 3-5
    q[:,2] = algs.trace(O2)
    feature_labels[2] = 'O2'
    q[:,3] = algs.trace(Ap2)
    feature_labels[3] = 'Ap2'
    q[:,4] = algs.trace(Ak2)
    feature_labels[4] = 'Ak2'
    # 6-14
    q[:,5] = algs.trace(np.matmul(O2,S))
    feature_labels[5] = 'O2*S'
    q[:,6] = algs.trace(np.matmul(O2,S2))
    feature_labels[6] = 'O2*S2'
    q[:,7] = algs.trace(np.matmul(np.matmul(O2,S),np.matmul(O,S2)))
    feature_labels[7] = 'O2*S*O*S2'
    q[:,8] = algs.trace(np.matmul(Ap2,S))
    feature_labels[8] = 'Ap2*S'
    q[:,9] = algs.trace(np.matmul(Ap2,S2))
    feature_labels[9] = 'Ap2*S2'
    q[:,10] = algs.trace(np.matmul(np.matmul(Ap2,S),np.matmul(Ap,S2)))
    feature_labels[10] = 'Ap2*S*Ap*S2'
    q[:,11] = algs.trace(np.matmul(Ak2,S))
    feature_labels[11] = 'Ak2*S'
    q[:,12] = algs.trace(np.matmul(Ak2,S2))
    feature_labels[12] = 'Ak2*S2'
    q[:,13] = algs.trace(np.matmul(np.matmul(Ak2,S),np.matmul(Ak,S2)))
    feature_labels[13] = 'Ak2*S*Ak*S2'

    # 15-17
    q[:,14] = algs.trace(np.matmul(O,Ap))
    feature_labels[14] = 'O*Ap'
    q[:,15] = algs.trace(np.matmul(Ap,Ak))
    feature_labels[15] = 'Ap*Ak'
    q[:,16] = algs.trace(np.matmul(O,Ak))
    feature_labels[16] = 'O*Ak'

    # 18-41
    q[:,17] = algs.trace(np.matmul(O,np.matmul(Ap,S)))
    feature_labels[17] = 'O*Ap*S'
    q[:,18] = algs.trace(np.matmul(O,np.matmul(Ap,S2)))
    feature_labels[18] = 'O*Ap*S2'
    q[:,19] = algs.trace(np.matmul(O2,np.matmul(Ap,S)))
    feature_labels[19] = 'O2*Ap*S'
    q[:,20] = algs.trace(np.matmul(Ap2,np.matmul(O,S)))
    feature_labels[20] = 'Ap2*O*S'
    q[:,21] = algs.trace(np.matmul(O2,np.matmul(Ap,S2)))
    feature_labels[21] = 'O2*Ap*S2'
    q[:,22] = algs.trace(np.matmul(Ap2,np.matmul(O,S2)))
    feature_labels[22] = 'Ap2*O*S2'
    q[:,23] = algs.trace(np.matmul(np.matmul(O2,S),np.matmul(Ap,S2)))
    feature_labels[23] = 'O2*S*Ap*S2'
    q[:,24] = algs.trace(np.matmul(np.matmul(Ap2,S),np.matmul(O,S2)))
    feature_labels[24] = 'Ap2*S*O*S2'

    q[:,25] = algs.trace(np.matmul(O,np.matmul(Ak,S)))
    feature_labels[25] = 'O*Ak*S'
    q[:,26] = algs.trace(np.matmul(O,np.matmul(Ak,S2)))
    feature_labels[26] = 'O*Ak*S2'
    q[:,27] = algs.trace(np.matmul(O2,np.matmul(Ak,S)))
    feature_labels[27] = 'O2*Ak*S'
    q[:,28] = algs.trace(np.matmul(Ak2,np.matmul(O,S)))
    feature_labels[28] = 'Ak2*O*S'
    q[:,29] = algs.trace(np.matmul(O2,np.matmul(Ak,S2)))
    feature_labels[29] = 'O2*Ak*S2'
    q[:,30] = algs.trace(np.matmul(Ak2,np.matmul(O,S2)))
    feature_labels[30] = 'Ak2*O*S2'
    q[:,31] = algs.trace(np.matmul(np.matmul(O2,S),np.matmul(Ak,S2)))
    feature_labels[31] = 'O2*S*Ak*S2'
    q[:,32] = algs.trace(np.matmul(np.matmul(Ak2,S),np.matmul(O,S2)))
    feature_labels[32] = 'Ak2*S*O*S2'

    q[:,33] = algs.trace(np.matmul(Ap,np.matmul(Ak,S)))
    feature_labels[33] = 'Ap*Ak*S'
    q[:,34] = algs.trace(np.matmul(Ap,np.matmul(Ak,S2)))
    feature_labels[34] = 'Ap*Ak*S2'
    q[:,35] = algs.trace(np.matmul(Ap2,np.matmul(Ak,S)))
    feature_labels[35] = 'Ap2*Ak*S'
    q[:,36] = algs.trace(np.matmul(Ak2,np.matmul(Ap,S)))
    feature_labels[36] = 'Ak2*Ap*S'
    q[:,37] = algs.trace(np.matmul(Ap2,np.matmul(Ak,S2)))
    feature_labels[37] = 'Ap2*Ak*S2'
    q[:,38] = algs.trace(np.matmul(Ak2,np.matmul(Ap,S2)))
    feature_labels[38] = 'Ak2*Ap*S2'
    q[:,39] = algs.trace(np.matmul(np.matmul(Ap2,S),np.matmul(Ak,S2)))
    feature_labels[39] = 'Ap2*S*Ak*S2'
    q[:,40] = algs.trace(np.matmul(np.matmul(Ak2,S),np.matmul(Ap,S2)))
    feature_labels[40] = 'Ak2*S*Ap*S2'

#    # 42
    q[:,41] = algs.trace(np.matmul(O,np.matmul(Ap,Ak)))
    feature_labels[41] = 'O*Ap*Ak'

#    # 43-47
    q[:,42] = algs.trace(np.matmul(np.matmul(O,Ap),np.matmul(Ak,S)))
    feature_labels[42] = 'O*Ap*Ak*S'
    q[:,43] = algs.trace(np.matmul(np.matmul(O,Ak),np.matmul(Ap,S)))
    feature_labels[43] = 'O*Ak*Ap*S'
    q[:,44] = algs.trace(np.matmul(np.matmul(O,Ap),np.matmul(Ak,S2)))
    feature_labels[44] = 'O*Ap*Ak*S2'
    q[:,45] = algs.trace(np.matmul(np.matmul(O,Ak),np.matmul(Ap,S2)))
    feature_labels[45] = 'O*Ak*Ap*S2'
    q[:,46] = algs.trace(np.matmul(np.matmul(np.matmul(O,Ap),np.matmul(S,Ak)),S2))
    feature_labels[46] = 'O*Ap*S*Ak*S2'

    # Supplementary features
    ########################
    feat = 47
    print('Calculating supplementary features: ')

    # Wall distanced based Re
    print('Turbulence Reynolds number')
    nu = rans_dsa.PointData['mu_l']/rans_dsa.PointData['ro']
    Red = (algs.sqrt(tke)*rans_dsa.PointData['d'])/(50.0*nu)
    q[:,feat] = algs.apply_dfunc(np.minimum, Red, 2.0)
    feature_labels[feat] = 'Turbulence Re'
    feat += 1

    # Turbulence intensity
    print('Turbulence intensity')
    UiUi = algs.mag(U)**2.0
    q[:,feat] = tke/(0.5*UiUi+tke+small)
    feature_labels[feat] = 'Turbulence intensity'
    feat += 1

    # Ratio of turb time scale to mean strain time scale
    print('Ratio of turb time scale to mean strain time scale')
    A = 1.0/rans_dsa.PointData['w']  #Turbulent time scale (eps = k*w therefore also A = k/eps)
    B = 1.0/Snorm
    q[:,feat] = A/(A+B+small)
    feature_labels[feat] = 'turb/strain time-scale'
    feat += 1

    return q, feature_labels
