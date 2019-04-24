import os

import numpy as np
import pandas as pd

# Load scikit's random forest classifier library and related tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, multilabel_confusion_matrix, precision_recall_curve

from classify_funcs import *

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter("ignore", FutureWarning)

#############
# User inputs
#############
casename = 'test'
modelname = 'test1'
data_fileloc = '.'
debug_fileloc = '.'

nX = 12
nY = 3

##########################
# Read in pandas dataframe
##########################
filename = casename + '_q_dataset.csv'
filename = os.path.join(data_fileloc, filename)
print('\nReading pandas features dataset from ', filename)
q_dataset = pd.read_csv(filename)
filename = casename + '_e_dataset.csv'
filename = os.path.join(data_fileloc, filename)
print('\nReading pandas targets dataset from ', filename)
e_dataset = pd.read_csv(filename)

#################
# Load classifier
#################
from joblib import load
filename = modelname + '.joblib'
filename = os.path.join(data_fileloc, filename)
print('\nLoading classifer from ', filename)
clf = load(filename) 

##############
# Prepare data
##############
#TODO - check for missing values and NaN's

# Find feature and target headers
X_headers = q_dataset.columns
Y_headers = e_dataset.columns

nX = X_headers.size
nY = Y_headers.size

#Y_headers = [Y_headers[1]]
#nY = 1 #TODO - temp

print('\nFeatures:')
for i in range(0,nX):
    print('%d/%d: %s' %(i+1,nX,X_headers[i]) )
print('\nTargets:')
for i in range(0,nY):
    print('%d/%d: %s' %(i+1,nY,Y_headers[i]) )

X_pred = q_dataset[X_headers]
Y_true = e_dataset[Y_headers]


######################################################
# Predict targets using loaded classifier and features
######################################################
print('\nPredicting targets using classifier')
Y_pred = clf.predict(X_pred)
Y_pred = pd.DataFrame(Y_pred,columns=Y_headers)

##################
# Accuracy metrics
##################
# Classifier accuracy
print('\nSubset accuracy of classifier =  %.2f %%' %(accuracy_score(Y_true,Y_pred)*100) )
print('\nClassifier accuracy for each target:')
for label in Y_headers:
    print('%s: %.2f %%' %(label, accuracy_score(Y_true[label],Y_pred[label])*100) )

# Confusion matrix
confuse_mat = multilabel_confusion_matrix(Y_true, Y_pred)
print('\nConfusion matrices:')
i = 0
for label in Y_headers:
    print('\nTarget %d: %s' %(i+1,label))
    print_cm(confuse_mat[i,:,:], ['Off','On'])
    i += 1

####################################
# Save predicted targets in vtk file 
####################################
import vtki

# Read vtk file with vtki
filename = casename + '_feat.vtk'
filename = os.path.join(debug_fileloc, filename)
print('Reading vtk file: ', filename)

rans_vtk = vtki.read(filename)  

# Get basic info about mesh
rans_nnode = rans_vtk.number_of_points
rans_ncell = rans_vtk.number_of_cells
print('Number of nodes = ', rans_nnode)
print('Number of cells = ', rans_ncell)

rans_vtk.point_arrays['Y_pred'] = Y_pred.to_numpy() 
#confuse_labels = # How to convert confuse_mat to [1,2,3,4] for True positive, True negative, False positive, false negative? #TODO
#rans_vtk.point_arrays['Y_confuse'] = confuse_labels

filename = casename + '_pred.vtk'
filename = os.path.join(debug_fileloc, filename)
print('Writing vtk file: ', filename)
rans_vtk.save(filename)
