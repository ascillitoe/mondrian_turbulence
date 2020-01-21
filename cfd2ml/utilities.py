import numpy as np
import pandas as pd

def build_cevm(Sij,Oij):
    nnode = Sij.shape[0]

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

    delij = np.zeros_like(Sij)
    for i in range(0,3): 
        delij[:,i,i] = 1.0

    # C1 term
    temp1  = np.zeros(nnode)
    temp2  = np.zeros(nnode)
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                temp1 += Sij[:,i,j]*Sij[:,i,k]*Sij[:,j,k]
                for l in range(0,3):
                    temp2 += Sij[:,i,j]*delij[:,i,j]*Sij[:,k,l]*Sij[:,k,l]
    C1term = 4.0*C1*(temp1 - temp2/3.0)/Cmu
    
    # C2 term
    temp1  = np.zeros(nnode)
    temp2  = np.zeros(nnode)
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                temp1 += Sij[:,i,j]*Oij[:,i,k]*Sij[:,k,j]
                temp2 += Sij[:,i,j]*Oij[:,j,k]*Sij[:,k,i]
    C2term = 4.0*C2*(temp1 + temp2)/Cmu
    
    # C3 term
    temp1  = np.zeros(nnode)
    temp2  = np.zeros(nnode)
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                temp1 += Sij[:,i,j]*Oij[:,i,k]*Oij[:,j,k]
                for l in range(0,3):
                    temp2 += Oij[:,l,k]*Oij[:,l,k]*Sij[:,i,j]*delij[:,i,j]
    C3term = 4.0*C3*(temp1 - temp2/3.0)/Cmu

    # C4 term
    temp1  = np.zeros(nnode)
    temp2  = np.zeros(nnode)
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    temp1 += Sij[:,i,j]*Sij[:,k,l]*Sij[:,k,i]*Oij[:,l,j]
                    temp2 += Sij[:,i,j]*Sij[:,k,j]*Oij[:,l,i]*Sij[:,k,l]
    C4term = 8.0*C4*(temp1 + temp2)/Cmu**2.0
    
    # C5 term
    # No need as C5=0
    
    # C6 term
    temp1  = np.zeros(nnode)
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    temp1 += Sij[:,i,j]*Sij[:,i,j]*Sij[:,k,l]*Sij[:,k,l]
    C6term = 8.0*C6*temp1/Cmu**2.0
    
    # C7 term
    temp1  = np.zeros(nnode)
    for i in range(0,3):
        for j in range(0,3):
            for k in range(0,3):
                for l in range(0,3):
                    temp1 += Sij[:,i,j]*Sij[:,i,j]*Oij[:,k,l]*Oij[:,k,l]
    C7term = 8.0*C7*temp1/Cmu**2.0

    cevm_2nd = C1term + C2term + C3term
    cevm_3rd = C4term + C6term + C7term

    return cevm_2nd, cevm_3rd

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("         Predicted")
    print("         Off   On ")
    print("True Off          ")
    print("     On         \n")
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    if(np.shape(cm)==(1,1)):
        print('No confusion!')
    else:
        for i, label1 in enumerate(labels):
            print("    %{0}s".format(columnwidth) % label1, end=" ")
            for j in range(len(labels)):
                cell = "%{0}.1d".format(columnwidth) % cm[i, j]
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                print(cell, end=" ")
            print()

def netcdf_to_vtk(cdffile,cdfscalars,vtkscalars):
    import pyvista as vista
    from scipy.io.netcdf import netcdf_file as Dataset

    # check scalar lists same length
    if (len(cdfscalars) != len(vtkscalars)):
        quit('WARNING: cdfscalars and vtkscalars in netcdf_to_vtk not same length')

    # Read cdffile
    print('\nReading NetCDF object from ', cdffile)
    ncfile = Dataset(cdffile,'r')
    
    nx = ncfile.dimensions['resolution_x']
    ny = ncfile.dimensions['resolution_y']
    nz = ncfile.dimensions['resolution_z']

    print('nx = ', nx)
    print('ny = ', ny)
    if (nz is not None): print('nz = ', nz)

    # Extract grid data
    x = np.array(ncfile.variables['grid_x'][0:nx],dtype='float64')
    x = np.tile(x,(ny,1)).transpose()
    y = np.array(ncfile.variables['grid_yx'][0:ny,0:nx],dtype='float64').transpose()
    z = np.zeros_like(x)
    
    vtk_obj = vista.StructuredGrid(x, y, z)

    # Extract scalars in list and save in vtk object
    for i in range(len(cdfscalars)):
        print('cdfscalar "%s" to vtkscalar "%s"' %(cdfscalars[i],vtkscalars[i]))
        var = np.array(ncfile.variables[cdfscalars[i]][0:ny,0:nx],dtype='float64').transpose()
        vtk_obj.point_arrays[vtkscalars[i]] = var.flatten(order='F')        

    return vtk_obj

def convert_hifi_fields(vtk,arraynames):

    nnode = vtk.number_of_points

    # Rename arrays according ot arraynames dict
    for array in arraynames:

        # If arraynames[array] is a string, rename
        if isinstance(arraynames[array],str): 
            vtk.rename_scalar(arraynames[array],array)

# Otherwise initialise scalar array to the value stored in arraynames[array]
        else:
            vtk.point_arrays[array] = np.ones(nnode)*arraynames[array]

    return vtk

def convert_rans_fields(vtk,comp=False):
    # required arrays: ro,U,p,k,w,mu_l,mu_t,d
    mydict = {}

    mydict['p']    = ['Pressure']
    mydict['k']    = ['TKE']
    mydict['w']    = ['Omega']
    mydict['mu_l'] = ['Laminar_Viscosity']
    mydict['mu_t'] = ['Eddy_Viscosity']
    mydict['d'] = ['Wall_Distance']

    if(comp==True):
        mydict['roU'] = ['Momentum']
        mydict['ro']  = ['Density']
    else:
        mydict['U']  = ['Velocity']
        vtk.point_arrays['ro'] = np.ones(vtk.number_of_points)

    scalars = vtk.scalar_names
    #print('\nSearching in vtk object to find desired scalar fields')
    #print('Current scalars: ' + str(scalars))
    #print('Searching for: ' + str(list(mydict.keys())))

    for want in mydict.keys():
        if(want not in scalars): 
            possible = mydict[want]
            scalar = find_scalar(want,scalars,possible)
            vtk.rename_scalar(scalar,want)

    if (comp==True): 
        vtk.point_arrays['roU'] = rans_vtk.point_arrays['roU']/np.array([rans_vtk.point_arrays['ro'],]*3).transpose()
        vtk.rename_scalar('roU','U')

    return vtk



def find_scalar(want,list,possible):
    found = [i for i in possible if i in list]
    if not found:
        quitstr = want + ' not found in vtk object. Possible scalar names: ' + str(possible)
        quit(quitstr)
    elif(len(found)>1):
        quitstr = 'Found more than one option for scalar called ' + want + '. Option found = ' + str(found)
        quit(quitstr)
    else:
        found = found[0]
        #print('Found scalar field ' + want + ', called ' + str(found) + '. Renaming...')
    return found


def convert_param_dist(json_dist):
    from scipy.stats import randint as sp_randint
    import numpy as np

    param_dist = {}
    Nparams = []

    for param in json_dist:
        value = json_dist[param]

        # If param item a list then this contains low and high values for sp_randint
        if isinstance(value,list):
            low  = value[0]
            high = value[1]
            param_dist[param] = sp_randint(low,high)        

            Nparams.append(high-low)  #assuming only integer params for now!

        # Else set param as a single value
        else:
            param_dist[param] = [value,]

            Nparams.append(1)


        Nmax = np.prod(Nparams)

    return param_dist, Nmax

def plot_precision_recall_threshold(p, r, thresholds, t=0.5, ax=None, c=None,case=None):
    """
    plots the precision recall curve and shows the current value for each
    by identifying the classifier's threshold (t).
    """
    from sklearn.metrics import auc
    import matplotlib.pyplot as plt

    if (ax is None):
        fig, ax = plt.subplots() 
    ax.set_ylim([0.5, 1.01])
    ax.set_xlim([0.5, 1.01])
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')

    # plot the curve
    if (case is not None):
        lab = 'Case %s: AUC=%.4f' % (case,auc(r, p))
    else:
        lab = 'AUC=%.4f' % (auc(r, p))

    ax.step(r, p, alpha=0.2,
             where='post',label=lab,lw=2,c=c)
    ax.fill_between(r, p, step='post', alpha=0.2,color=c)

    # plot the current threshold on the line
    close_default_clf = np.argmin(np.abs(thresholds - t))
    ax.plot(r[close_default_clf], p[close_default_clf], 'x', c='k',
            markersize=10,mew=2)

def plot_precision_recall_vs_threshold(p, r, thresholds, ax=None,c=None,t=None,case=None):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    if (ax is None):
        fig, ax = plt.subplots() 

    if (case is not None): 
        labp='Case ' + case + ': Precision'
        labr='Case ' + case + ': Recall'
    else:
        labp='Precision'
        labr='Recall'
    ax.plot(thresholds, p[:-1], "-", c=c, label=labp)
    ax.plot(thresholds, r[:-1], "--", c=c, label=labr)

    # Plot the current threshold
    if (t is not None):
        close_default_clf = np.argmin(np.abs(thresholds - t))
        ax.plot(thresholds[close_default_clf], p[close_default_clf], '+', c=c,
                markersize=10,mew=2,label='_nolegend_')
        ax.plot(thresholds[close_default_clf], r[close_default_clf], 'x', c=c,
                markersize=10,mew=2,label='_nolegend_')

    ax.set_ylabel("Score")
    ax.set_xlabel("Decision Threshold")

def eijk(i,j,k):
    """ Levi-Civita symbol
    """
    try:
        e = ((j-i)*(k-i)*(k-j))/(abs(j-i)*abs(k-i)*abs(k-j))
    except ZeroDivisionError:
        e = 0

    return e

def RFE_perm(model,X,y,feats,cv=5,scoring='neg_mean_absolute_error',timing=False):
#def RFE_perm(model,X,y,min_features=1,step=1,cv=5,scoring='neg_mean_absolute_error',timing=False):
    from eli5.sklearn import PermutationImportance
    from types import GeneratorType
    import time

    # if pandas data then convert to numpy arrays
    if isinstance(X, pd.DataFrame): X = X.to_numpy()
    if isinstance(y, pd.Series): y = y.to_numpy()

    # if cv is a generator convert to list so it doesn't disappear after first iter
    if isinstance(cv,GeneratorType): cv = list(cv)

    nfeat = np.shape(X)[1]
    index = np.arange(nfeat)
    bestscore = -99
    niter = len(feats)
#    niter = int(np.floor((nfeat - min_features)/step)+1)
    scores = np.empty(niter)
    nfeats = np.empty(niter)
    traintime = np.empty(niter)
    predtime  = np.empty(niter)
    featsets = np.zeros([niter,nfeat])
#    for i, n in enumerate(range(nfeat,min_features-1,-step)):
    for i, n in enumerate(feats):
        if n==nfeat:  # first iter
            newfeat = index
            Xcut = X
        else:
            newfeat    = sortimport[:n]  # take n most important features from previous iter    
            Xcut = Xcut[:,newfeat]
        index = index[newfeat]

        # Get train time and prediction time
        if timing:
            start = time.time()
            model.fit(Xcut,y)
            end = time.time()
            traintime[i] = end - start
            start = time.time()
            model.predict(Xcut)
            end = time.time()
            predtime[i] = end - start

        perm = PermutationImportance(model, random_state=42,scoring=scoring,cv=cv)
        perm.fit(Xcut,y)
        featimport = perm.feature_importances_
        sortimport = np.argsort(featimport)[::-1]

        score = np.mean(perm.scores_)
        print('Number of features: %i, score: %.2f %%' %(n,100*np.abs(score)))
    
        scores[i] = score
        nfeats[i] = n
        featsets[i,index] = 1 
    
        if(score >= bestscore): #> or = because if equal with smaller nfeat, then better!
            bestscore = score
            bestfeat  = index

    if timing:
        return [nfeats,scores,traintime,predtime], bestscore, bestfeat, featsets
    else:
        return [nfeats,scores], bestscore, bestfeat, featsets


def mahalanobis(x=None, data=None, cov=None,norm=True):
    from scipy.linalg import pinv

    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = pinv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    dist =  mahal.diagonal()
    if norm:
        npts = np.shape(x)[0]
        distnorm = np.empty(npts)
        for i in range(npts):
            count = (dist>dist[i]).sum()
            gamma = float(count)/float(npts)
            distnorm[i] = 1-gamma
        dist = distnorm
    return dist
