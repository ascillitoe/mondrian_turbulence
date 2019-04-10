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
