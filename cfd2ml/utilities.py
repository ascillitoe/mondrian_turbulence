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

def netcdf_to_vtk(cdffile,vtkfile,cdfscalars,vtkscalars):
    import vtki
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
    
    vtk_obj = vtki.StructuredGrid(x, y, z)

    # Extract scalars in list and save in vtk object
    for i in range(len(cdfscalars)):
        print('cdfscalar "%s" to vtkscalar "%s"' %(cdfscalars[i],vtkscalars[i]))
        var = np.array(ncfile.variables[cdfscalars[i]][0:ny,0:nx],dtype='float64').transpose()
        vtk_obj.point_arrays[vtkscalars[i]] = var.flatten(order='F')        

    # Write vtk file
    print('\nWriting vtk object to ', vtkfile)
    vtk_obj.save(vtkfile)


    return vtk_obj


