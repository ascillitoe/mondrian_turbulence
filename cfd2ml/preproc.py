import numpy as np
import pandas as pd

import os
import copy
import vista
from vtk.numpy_interface import algorithms as algs
from vtk.numpy_interface import dataset_adapter as dsa

from cfd2ml.base import CaseData

def preproc_RANS_based(json):
    from cfd2ml.preproc import preproc_RANS_and_HiFi
    from cfd2ml.utilities import convert_rans_fields, convert_hifi_fields

    print('\n-----------------------')
    print('Started pre-processing')
    x_type = json['x_type'] 
    y_type = json['y_type'] 

    if (x_type==1):
        print('Feature set 1: Interpretable features')
    elif (x_type==2):
        print('Feature set 2: Invarient features')

    nondim = 'local' #default
    if (("non-dim" in json)==True):
        nondim = json['non-dim']
        if nondim=='local':
            print('Local feature non-dimensionalistion')
        elif nondim=='global':
            print('Global feature non-dimensionalistion')
        else: 
            quit('Invalid non-dim option')
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
        X_data, Y_data = preproc_RANS_and_HiFi(X_data, Y_data, x_type, y_type, nondim, **options)

        # Write data
        X_data.Write(os.path.join(outdir, id + '_X')) 
        Y_data.Write(os.path.join(outdir, id + '_Y'))

    print('\n-----------------------')
    print('Finished pre-processing')
    print('-----------------------')

def preproc_RANS_and_HiFi(q_data, y_data, x_type, y_type, nondim, clip=None,comp=False,Ls=1,Us=1,ros=1):
   
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
    rans_ncell = rans_vtk.number_of_cells
    
    # Initial processing of HiFi data
    ################################
    print('\nInitial processing of HiFi data')
    
    les_vtk = y_data.vtk
    
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

    if (x_type==1):
        q, feature_labels = make_features(rans_vtk,Ls=Ls,Us=Us,ros=ros, nondim=nondim)
    elif (x_type==2):
        q, feature_labels = make_features_inv(rans_vtk,Ls=Ls,Us=Us,ros=ros,nondim=nondim)

    # Store features in vtk obj
    ###########################
    rans_vtk.point_arrays['q'] = q
    
    ############################
    # Generate HiFi error metrics
    ############################
    print('\nProcessing HiFi data into error metrics')
    print(  '--------------------------------------')

    y_raw, y_targ, target_labels = make_targets(les_vtk,y_type,Ls=Ls,Us=Us,ros=ros)
    
    # Store errors in vtk obj
    les_vtk.point_arrays['raw'] = y_raw
    les_vtk.point_arrays['target'] = y_targ
     
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

    #############################################################
    # Add Rij error between RANS and LES as an extra error metric (can only do after interp)
    #############################################################
    rans_les_uiuj_err(rans_vtk,les_vtk, 0.05,Ls=Ls,Us=Us)
    target_labels = np.append(target_labels,'Rij error')

    ###########################
    # Clip mesh to given ranges (also do after features and metrics generated to prevent probs with gradients at new boundaries)
    ###########################
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
        les_vtk = les_vtk.clip(normal='x', origin=xclip_min,invert=False)
        les_vtk = les_vtk.clip(normal='x', origin=xclip_max,invert=True )
        les_vtk = les_vtk.clip(normal='y', origin=xclip_min,invert=False)
        les_vtk = les_vtk.clip(normal='y', origin=xclip_max,invert=True )
        les_vtk = les_vtk.clip(normal='z', origin=xclip_min,invert=False)
        les_vtk = les_vtk.clip(normal='z', origin=xclip_max,invert=True )
        print('Number of nodes clipped = ', rans_nnode - rans_vtk.number_of_points)
    rans_nnode = rans_vtk.number_of_points
    rans_ncell = rans_vtk.number_of_cells
    print('New number of nodes = ', rans_nnode)
    print('New number of cells = ', rans_ncell)
 
    # Get X and Y data back out of vtk objs after interp and clipping
    q = rans_vtk.point_arrays['q']
    y_targ = les_vtk.point_arrays['target']

    #############################
    # Final checks and store data
    #############################
    # pandas dataframes
    q_data.pd = pd.DataFrame(q, columns=feature_labels)
    y_data.pd = pd.DataFrame(y_targ, columns=target_labels)

    # Check for NaN's and flag them (by saving to q_nan in vtk with value=99)
    q_nansum =q_data.pd.isnull().sum().sum() 
    if (q_nansum > 0):
        print('********** Warning %d NaNs in X data **********' %(q_nansum))
        q_nans = q_data.pd.fillna(99)
        rans_vtk.point_arrays['q_nan'] = q_nans

    df = pd.DataFrame(y_raw)
    y_nansum = df.isnull().sum().sum() 
    if (y_nansum > 0):
        print('********** Warning %d NaNs in Y data **********' %(y_nansum))
        y_nans = y_data.pd.fillna(99)
        les_vtk.point_arrays['y_raw_nan'] = y_nans

    # Put new vtk data back into vtk objects
    q_data.vtk = rans_vtk
    y_data.vtk = les_vtk
   

    return q_data, y_data


def preproc_RANS(q_data, y_data, clip=None, comp=False):

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

def make_features(rans_vtk,Ls=1,Us=1,ros=1,nondim='local'):
    from cfd2ml.utilities import build_cevm

    small = np.cbrt(np.finfo(float).tiny)
    Ps = 0.5*ros*Us**2

    rans_nnode = rans_vtk.number_of_points

    delij = np.zeros([rans_nnode,3,3])
    for i in range(0,3): 
        delij[:,i,i] = 1.0

    # Wrap vista object in dsa wrapper
    rans_dsa = dsa.WrapDataObject(rans_vtk)

    print('Feature:')
    nfeat = 15
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
    q[:,feat] = (Onorm - 0.5*Snorm)/(Onorm + 0.5*Snorm + small)
    feature_labels[feat] = 'Normalised strain'
    feat += 1
    
    # clean up 
    Snorm = algs.sqrt(Snorm) #revert to revert to real Snorm for use later
    Onorm = algs.sqrt(Onorm) #revert to revert to real Onorm for use later
    
    # Feature 2: Turbulence intensity
    #################################
    print('2: Turbulence intensity')
    tke = rans_dsa.PointData['k'] 
    UiUi = algs.mag(U)**2.0
#    q[:,feat] = tke/(0.5*UiUi+tke+small)
    q[:,feat] = tke/(0.5*UiUi+small)
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
    print('4: Stream-wise pressure gradient')
    A = np.zeros(rans_nnode)
    B = np.zeros(rans_nnode)
    
    dpdx  = algs.gradient(rans_dsa.PointData['p'])
    ro = rans_dsa.PointData['ro']
    Umag = algs.mag(U)

    for k in range(0,3):
        A += U[:,k]*dpdx[:,k] 

    if nondim=='global':
        A = A/Umag
        q[:,feat] = A*Ls/Ps
    elif nondim=='local':
        for i in range(0,3):
            for j in range(0,3):
                B += U[:,i]*U[:,i]*dpdx[:,j]*dpdx[:,j]
        q[:,feat] = A/(algs.sqrt(B)+algs.abs(A)+small)

    feature_labels[feat] = 'Stream-wise Pgrad'
    feat += 1
    
    # Feature 5: Ratio of turb time scale to mean strain time scale
    ###############################################################
    print('5: Ratio of turb time scale to mean strain time scale')
#    A = 1.0/rans_dsa.PointData['w']  #Turbulent time scale (eps = k*w therefore also A = k/eps)
#    B = 1.0/Snorm
#    q[:,feat] = A/(A+B+small)
    q[:,feat] = Snorm/(rans_dsa.PointData['w']+small)
    feature_labels[feat] = 'turb/strain time-scale'
    feat += 1
    
    # Feature 6: Viscosity ratio
    ############################
    print('6: Viscosity ratio')
    nu_t = rans_dsa.PointData['mu_t']/ro
#    q[:,feat] = nu_t/(100.0*nu + nu_t)
    q[:,feat] = nu_t/(nu+small)
    feature_labels[feat] = 'Viscosity ratio'
    feat += 1
    
    # Feature 7: Vortex stretching
    ##############################
    print('7: Vortex stretching')
    A = np.zeros(rans_nnode)
    B = np.zeros(rans_nnode)
    
    vortvec = algs.vorticity(U)
    
    for j in range(0,3):
        for i in range(0,3):
            for k in range(0,3):
                A += vortvec[:,j]*J[:,i,j]*vortvec[:,k]*J[:,i,k]
    
    B = Snorm
    
#    q[:,feat] = algs.sqrt(A)/(algs.sqrt(A)+B+small)
    q[:,feat] = algs.sqrt(A)#/(algs.sqrt(A)+B+small)
    feature_labels[feat] = 'Vortex stretching'
    feat += 1
    
    # Feature 8: Marker of Gorle et al. (deviation from parallel shear flow)
    ########################################################################
    print('8: Marker of Gorle et al. (deviation from parallel shear flow)')
    if nondim=='global':
        g = np.zeros([rans_nnode,3])
        m = np.zeros(rans_nnode)
        s = U/Umag
        for j in range(3):
            for i in range(3):
                g[:,j] += s[:,i]*J[:,i,j]
            m += g[:,j]*s[:,j]
        m = np.abs(m)
        q[:,feat] = m*Ls/Us 
    elif nondim=='local':
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
    
    # Feature 9: Ratio of convection to production of k
    ####################################################
    print('9: Ratio of convection to production of k')
    uiuj = (2.0/3.0)*tke*delij - 2.0*nu_t*Sij
    dkdx  = algs.gradient(tke)
    A = np.zeros(rans_nnode)
    B = np.zeros(rans_nnode)
    for i in range(0,3):
        A += U[:,i]*dkdx[:,i] 
    for j in range(0,3):
        for l in range(0,3):
            B += uiuj[:,j,l]*Sij[:,j,l]
    q[:,feat] = A/(algs.abs(B)+small)
    feature_labels[feat] = 'Convection/production of k'
    feat += 1
    
    # Feature 10: Ratio of total Reynolds stresses to normal Reynolds stresses
    ##########################################################################
    print('10: Ratio of total Reynolds stresses to normal Reynolds stresses')
    # Frob. norm of uiuj
    A = algs.sum(uiuj**2,axis=1) # sum i axis
    A = algs.sum(      A,axis=1) # sum previous summations i.e. along j axis
    A = algs.sqrt(A)
    B = tke
    q[:,feat] = A/(B + small)
    feature_labels[feat] = 'total/normal stresses'
    feat += 1
    
    # Feature 11: Cubic eddy viscosity comparision
    ##############################################
    print('11: Cubic eddy viscosity comparision')
    
    # Add quadratic and cubic terms to linear evm
    cevm_2nd, cevm_3rd = build_cevm(Sij,Oij)
    uiujSij = np.zeros(rans_nnode)
    for i in range(0,3):
        for j in range(0,3):
            uiujSij += uiuj[:,i,j]*Sij[:,i,j]
    uiujcevmSij = uiujSij + (cevm_2nd/tke)*nu_t**2.0 + (cevm_3rd/tke**2.0)*nu_t**3.0
    q[:,feat] = (uiujcevmSij-uiujSij) / (0.5*(np.abs(uiujcevmSij)+np.abs(uiujSij)) + small)
    feature_labels[feat] = 'CEV comparison'
    feat += 1

    # Feature 12: Streamline normal pressure gradient
    #################################################
    print('12: Stream-normal pressure gradient')
    A = algs.cross(U,dpdx)
    A = np.sqrt(A[:,0]**2 + A[:,1]**2 + A[:,2]**2)

    if nondim=='global':
        A = A/Umag
        q[:,feat] = A*Ls/Ps
    elif nondim=='local':
        B = np.zeros(rans_nnode)
        for i in range(0,3):
            for j in range(0,3):
                B += U[:,i]*U[:,i]*dpdx[:,j]*dpdx[:,j]
        q[:,feat] = A/(A + algs.sqrt(B) + small) 

    feature_labels[feat] = 'Stream-normal Pgrad'
    feat += 1

    # Feature 13: Streamline curvature
    ##################################
    print('13: Streamline curvature')
#    A = np.zeros([rans_nnode,3])
#   
#    # Gradient of Gamma
#    Gamma = U#/algs.mag(U)
#    dGammadx = algs.gradient(Gamma)
#
#    for i in range(0,3):
#        for j in range(0,3):
#            A[:,i] += U[:,j]*dGammadx[:,j,i]
#    A = algs.mag(A/algs.mag(U)*algs.mag(U))
#
#    q[:,feat] = A
#    feature_labels[feat] = 'Streamline curvature'
#    feat += 1
    D2 = 0.5*(Snorm**2 + Onorm**2)
#    cr1 = 1.0
#    cr2 = 12.0
#    cr3 = 1.0
    cr2 = 12
    cr3 = 1/np.pi
    rstar = Snorm/(Onorm+small)
    
    dSijdx1 = algs.gradient(Sij[:,:,0])
    dSijdx2 = algs.gradient(Sij[:,:,1])
    dSijdx3 = algs.gradient(Sij[:,:,2])

    DSijDt = np.zeros([rans_nnode,3,3])
    for i in range(3):
        for j in range(3):
            DSijDt[:,i,j] = U[:,0]*dSijdx1[:,j,i] + U[:,1]*dSijdx2[:,j,i] + U[:,2]*dSijdx3[:,j,i]

    rhat = np.zeros(rans_nnode)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                rhat += (2*Oij[:,i,k]*Sij[:,j,k]/D2**2)*DSijDt[:,i,j]

#    fr1 = -((2*rstar)/(1+rstar))*(cr3*algs.arctan(cr2*rhat)) 
#    fr1 = ( (1+cr1)*((2*rstar)/(1+rstar))*(1-cr3*algs.arctan(cr2*rhat)) ) - cr1
    rhathat = algs.arctan(0.25*rhat)*2/np.pi
    q[:,feat] = rhathat #fr1
    feature_labels[feat] = 'Streamline curvature'
    feat += 1


    # Feature 14: Anisotropy of pressure hessian
    ############################################
    print('14: Anisotropy of pressure hessian')
    # Calculate pressure hessian
    Hij = algs.gradient(dpdx)
    Hij = algs.apply_dfunc(np.transpose,Hij,(0,2,1))
    aniso = np.zeros(rans_nnode)
    iso   = np.zeros(rans_nnode)
    # Frob. norm of Hij
    for i in range(3):
        for j in range(3):
            aniso += (Hij[:,i,j]-Hij[:,i,j]*delij[:,i,j])**2
        iso   += Hij[:,i,i]**2

    aniso = np.sqrt(aniso)
    iso   = np.sqrt(iso)
    q[:,feat] = (aniso)/(iso + small)

    feature_labels[feat] = 'Anisotropy of pressure hessian'
    feat += 1

    # Feature 15: White noise
    #########################
    print('15: White noise')
    q[:,feat] = np.random.uniform(low=-1.0, high=1.0, size=rans_nnode)
    feature_labels[feat] = 'White noise'
    feat += 1

    return q, feature_labels

def make_targets(les_vtk,y_type,Ls=1,Us=1,ros=1):
    from tqdm import tqdm

    small = np.cbrt(np.finfo(float).tiny)
    Ps = 0.5*ros*Us**2

    les_nnode = les_vtk.number_of_points

    delij = np.zeros([les_nnode,3,3])
    for i in range(0,3): 
        delij[:,i,i] = 1.0

    # Wrap vista object in dsa wrapper
    les_dsa = dsa.WrapDataObject(les_vtk)

    if (y_type=='classification'):
        ntarg = 5
        y_targ = np.zeros([les_nnode,ntarg],dtype=int)
        print('Classifier targets:')
    elif (y_type=='regression'):
        ntarg = 2
        y_targ = np.zeros([les_nnode,ntarg],dtype=float)
        print('regressor targets:')

    y_raw  = np.zeros([les_nnode,ntarg])
    target_labels = np.empty(ntarg, dtype='object')
    targ = 0
  
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
  
    # Anisotropy tensor and eigenvalues
    aij  = copy.deepcopy(Sij)*0.0
    inv2 = np.zeros(les_nnode)
    inv3 = np.zeros(les_nnode)
    
    for i in range(0,3):
        for j in range(0,3):
            aij[:,i,j] = uiuj[:,i,j]/(2.0*tke+small) - delij[:,i,j]/3.0
    
    # Get eigenvalues of aij
    eig = algs.eigenvalue(aij)
    eig1 = eig[:,0]
    eig2 = eig[:,1]
    eig3 = eig[:,2]
    
    # Get coords on barycentric triangle from eigenvalues
    xc = [1.0, 0.0, 0.5] #x,y coords of corner of triangle
    yc = [0.0, 0.0, np.cos(np.pi/6.0)]
    C1c = eig1 - eig2
    C2c = 2*(eig2-eig3)
    C3c = 3*eig3 + 1
    x0 = C1c*xc[0] + C2c*xc[1] + C3c*xc[2]
    y0 = C1c*yc[0] + C2c*yc[1] + C3c*yc[2]

    if (y_type=='Classification'):
        # Target 1: Negative eddy viscosity
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
        nu_t = nu_t/(Us*Ls)
        y_raw[:,targ] = nu_t
        
        index = algs.where(nu_t<0.0)
        y_targ[index,targ] = 1
        target_labels[targ] = 'Negative eddy viscosity'
        targ += 1
        
        # Target 2: Deviation from plane shear
        #################################################
        print('2: Deviation from plane shear turbulence')
        # Get distance from plane shear line
        p1 = (1/3,0)
        p2 = (0.5,np.sqrt(3)/2)
        dist = abs( (p2[1]-p1[1])*x0 - (p2[0]-p1[0])*y0 + p2[0]*p1[1] - p2[1]*p1[0] ) /  np.sqrt( (p2[1]-p1[1])**2 + (p2[0]-p1[0])**2 )  
        y_raw[:,targ] = dist
        index = algs.where(dist>0.25)
    
        y_targ[index,targ] = 1
        target_labels[targ] = 'Deviation from plane shar turbulence'
        targ += 1
    
        # Target 3: Anisotropy of turbulence
        ##########################################
        print('3: Anisotropy of turbulence')
        Caniso = 1.0 - C3c
        y_raw[:,targ] = Caniso
        index = algs.where(Caniso>0.5)
        y_targ[index,targ] = 1
        target_labels[targ] = 'Stress anisotropy'
        targ += 1
    
        # Target 4: Negative Pk
        ############################################
        print('4: Negative Pk')
        A = np.zeros(les_nnode)
        for i in range(0,3):
            for j in range(0,3):
                A[:] += (-uiuj[:,i,j] * J[:,i,j])
    
        A = A*Ls/Us**3 
        y_raw[:,targ] = A
        index = algs.where(A<-0.0005)
    
        y_targ[index,targ] = 1
        target_labels[targ] = 'Negative Pk'
        targ += 1
    
        # Target 5: 2-eqn Cmu constant
        ############################################
        print('5: 2-equation Cmu constant')
        A = np.zeros(les_nnode)
        for i in range(0,3):
            for j in range(0,3):
                A[:] += aij[:,i,j]*Sij[:,i,j]
    
        Cmu = nu_t**2.0*(Str/(tke+small))**2.0
    
        y_raw[:,targ] = Cmu
        allow_err = 0.25 #i.e. 10% err
        Cmu_dist = algs.abs(Cmu - 0.09)
    #    index = algs.where(Cmu_dist>allow_err*0.09)
        index = algs.where(Cmu>1.1*0.09)
        y_targ[index,targ] = 1
        target_labels[targ] = 'Cmu != 0.09'
        targ += 1
    
    #    ab = ((uiuj[:,1,1]-uiuj[:,0,0])*U[:,0]*U[:,1] + uiuj[:,0,1]*(U[:,0]**2-U[:,1]**2))/(U[:,0]**2+U[:,1]**2)
    #    y_raw[:,err] = ab
    
    #    # Target 3: Non-linearity
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
    #    y_raw[:,err] = nu_t_cevm
    #    
    #    index = algs.where(normdiff>0.15)
    #    y_targ[index,err] = 1
    #    error_labels[err] = 'Non-linearity'
    #    err += 1

    elif (y_type=='regression'):
        # Target 3: Anisotropy of turbulence
        ##########################################
        print('1: Anisotropy of turbulence')
        Caniso = 1.0 - C3c
        y_raw[:,targ] = Caniso
        y_targ[:,targ] = Caniso
        target_labels[targ] = 'Stress anisotropy'
        targ += 1

    return y_raw, y_targ, target_labels

def rans_les_uiuj_err(rans_vtk,les_vtk, thrsh,Ls=1,Us=1):

    small = np.cbrt(np.finfo(float).tiny)

    nnode = les_vtk.number_of_points

    delij = np.zeros([nnode,3,3])
    for i in range(0,3): 
        delij[:,i,i] = 1.0

    ###############
    # Get RANS uiuj
    ###############
    rans_dsa = dsa.WrapDataObject(rans_vtk)
    U = rans_dsa.PointData['U'] # NOTE - Getting variables from dsa obj not vtk obj as want to use algs etc later
    # Velocity gradient tensor and its transpose
    # J[:,i-1,j-1] is dUidxj
    # Jt[:,i-1,j-1] is dUjdxi
    Jt = algs.gradient(U)        # Jt is this one as algs uses j,i ordering
    J  = algs.apply_dfunc(np.transpose,Jt,(0,2,1))
    
    # Strain and vorticity tensors
    Sij = 0.5*(J+Jt)
    nu_t = rans_dsa.PointData['mu_t']/rans_dsa.PointData['ro']
    tke = rans_dsa.PointData['k'] 
    uiuj_rans = (2.0/3.0)*tke*delij - 2.0*nu_t*Sij

    ###############
    # Get LES uiuj
    ###############
    # Wrap vista object in dsa wrapper
    les_dsa = dsa.WrapDataObject(les_vtk)

    # Copy Reynolds stresses to tensor
#    uiuj_les = np.zeros([nnode,3,3])
    uiuj_les = copy.deepcopy(uiuj_rans)*0.0
    uiuj_les[:,0,0] = les_dsa.PointData['uu']
    uiuj_les[:,1,1] = les_dsa.PointData['vv']
    uiuj_les[:,2,2] = les_dsa.PointData['ww']
    uiuj_les[:,0,1] = les_dsa.PointData['uv']
    uiuj_les[:,0,2] = les_dsa.PointData['uw']
    uiuj_les[:,1,2] = les_dsa.PointData['vw']
    uiuj_les[:,1,0] = uiuj_les[:,0,1]
    uiuj_les[:,2,0] = uiuj_les[:,0,2]
    uiuj_les[:,2,1] = uiuj_les[:,1,2]
    tke_les = 0.5*(uiuj_les[:,0,0]+uiuj_les[:,1,1]+uiuj_les[:,2,2])

    # Calc divergence of DNS and RANS uiuj
    div_RANS = np.zeros([nnode,3])
    div_LES  = np.zeros([nnode,3])

    for i in range(3):
        for j in range(3):
            div_RANS[:,i] += algs.gradient(uiuj_rans[:,i,j])[:,i]
            div_LES[:,i] += algs.gradient(uiuj_les[:,i,j])[:,i]

    div_err = 1.0*div_LES - rans_dsa.PointData['ro']*div_RANS

    err  = np.sqrt(div_err[:,0]**2 + div_err[:,1]**2 + div_err[:,2]**2)

#    err = err/(tke_les+small)
    err = err*Ls/Us**2

#    # Calc Frobenius norm distance between RANS and LES
#    err  = np.zeros(nnode)
#    for i in range(3):
#        for j in range(3):
#            err[:]  += ( (uiuj_rans[:,i,j]-uiuj_les[:,i,j]) )**2.
#    err = np.sqrt(err)
#    err = err/(tke_les+small)

    index = algs.where(err>thrsh)
    err_bool = np.zeros(nnode,dtype=int)
    err_bool[index] = 1 
    y_raw = les_vtk.point_arrays['raw'] 
    y_targ = les_vtk.point_arrays['target'] 

    les_vtk.point_arrays['raw'] = np.append(y_raw,err.reshape(-1,1),axis=1)
    les_vtk.point_arrays['target'] = np.append(y_targ,err_bool.reshape(-1,1),axis=1)

def make_features_inv(rans_vtk,Ls=1,Us=1,ros=1,nondim='local'):
    from cfd2ml.utilities import eijk

    small = np.finfo(float).tiny

    rans_nnode = rans_vtk.number_of_points

    # Wrap vista object in dsa wrapper
    rans_dsa = dsa.WrapDataObject(rans_vtk)

    print('Feature:')
#    nfeat = 21
    nfeat = 50
    q = np.empty([rans_nnode,nfeat])
    feature_labels = np.empty(nfeat, dtype='object')
    
    # strain and vorticity
    ##############################
    print('Constructing strain and vorticity tensor')
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

    ########################################
    # Calculating pressure and tke gradients
    ########################################
    print('Calculating pressure and tke gradients')
    tke = rans_dsa.PointData['k']
    dkdx  = algs.gradient(tke)
    dpdx  = algs.gradient(rans_dsa.PointData['p'])

    #################################################
    # Non-dim everything here. Either local or global
    #################################################
    if nondim=='local':
        # Non-dim Sij by eps/k
        eps = rans_dsa.PointData['w']*tke
        Sij_h = Sij / (eps/tke)
    
        # Non-dim Oij by eps/k
        Oij_h = Oij / (eps/tke)
    
        # Non-dim pressure gradient
        ro = rans_dsa.PointData['ro']
        DUDt = U[:,0]*J[:,:,0] + U[:,1]*J[:,:,1] + U[:,2]*J[:,:,2]
        dpdx_h = dpdx / ro*algs.mag(DUDt)
    
        # Non-dim tke gradient
        dkdx_h = dkdx / (eps/algs.sqrt(tke)) 

    elif nondim=='global':
        Ps = 0.5*ros*Us**2
        Sij_h = Sij/(Us/Ls)
        Oij_h = Oij/(Us/Ls)
        dpdx_h = dpdx/(Ps/Ls)
        dkdx_h = dkdx/(Us**2/Ls)

#    q[:,0]  = Sij_h[:,0,0]
#    q[:,1]  = Sij_h[:,1,1]
#    q[:,2]  = Sij_h[:,2,2]
#    q[:,3]  = Sij_h[:,0,1]
#    q[:,4]  = Sij_h[:,0,2]
#    q[:,5]  = Sij_h[:,1,2]
#    q[:,6]  = Oij_h[:,0,0]
#    q[:,7]  = Oij_h[:,1,1]
#    q[:,8]  = Oij_h[:,2,2]
#    q[:,9]  = Oij_h[:,0,1]
#    q[:,10] = Oij_h[:,0,2]
#    q[:,11] = Oij_h[:,1,2]
#    q[:,12] = dpdx_h[:,0]
#    q[:,13] = dpdx_h[:,1]
#    q[:,14] = dpdx_h[:,2]
#    q[:,15] = dkdx_h[:,0]
#    q[:,16] = dkdx_h[:,1]
#    q[:,17] = dkdx_h[:,2]
#    feature_labels[0]  = 'S11'
#    feature_labels[1]  = 'S22'
#    feature_labels[2]  = 'S33'
#    feature_labels[3]  = 'S12'
#    feature_labels[4]  = 'S13'
#    feature_labels[5]  = 'S23'
#    feature_labels[6]  = 'O11'
#    feature_labels[7]  = 'O22'
#    feature_labels[8]  = 'O33'
#    feature_labels[9]  = 'O12'
#    feature_labels[10] = 'O13'
#    feature_labels[11] = 'O23'
#    feature_labels[12] = 'dpdx'
#    feature_labels[13] = 'dpdy'
#    feature_labels[14] = 'dpdz'
#    feature_labels[15] = 'dkdx'
#    feature_labels[16] = 'dkdy'
#    feature_labels[17] = 'dkdz'
#    feat = 18

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
    feat = 47

    # Supplementary features
    ########################
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
