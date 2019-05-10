import numpy as np

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
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
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
    import vista
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

def search_les_fields(vtk,hasw=False,hasro=False):
    # required arrays: ro,U,V,W,p,uu,vv,ww,uv,uw,vw
    mydict = {}
    mydict['U']  = ['mean_u_xyz','avgUx']
    mydict['V']  = ['mean_v_xyz','avgUy']
    mydict['p']  = ['mean_p_xyz','avgP']
    mydict['uu'] = ['reynolds_stress_uu_xyz','UU']
    mydict['vv'] = ['reynolds_stress_vv_xyz','VV']
    mydict['ww'] = ['reynolds_stress_ww_xyz','WW']
    mydict['uv'] = ['reynolds_stress_uv_xyz','UV']
    mydict['uw'] = ['reynolds_stress_uw_xyz','UW']
    mydict['vw'] = ['reynolds_stress_vw_xyz','VW']

    if(hasw==True):
        mydict['W']  = ['mean_w_xyz','avgUz']
    else:
        vtk.point_arrays['W'] = np.zeros(vtk.number_of_points)

    if(hasro==True):
        mydict['ro']  = ['mean_ro_xyz','Density']
    else:
        vtk.point_arrays['ro'] = np.ones(vtk.number_of_points)

    scalars = vtk.scalar_names
    print('\nSearching in vtk object to find desired scalar fields')
    print('Current scalars: ' + str(scalars))
    print('Searching for: ' + str(list(mydict.keys())))


    for want in mydict.keys():
        if(want not in scalars): 
            possible = mydict[want]
            scalar = find_scalar(want,scalars,possible)
            vtk.rename_scalar(scalar,want)

    return vtk

def search_rans_fields(vtk,comp=False):
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
    print('\nSearching in vtk object to find desired scalar fields')
    print('Current scalars: ' + str(scalars))
    print('Searching for: ' + str(list(mydict.keys())))

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
        print('Found scalar field ' + want + ', called ' + str(found) + '. Renaming...')
    return found

