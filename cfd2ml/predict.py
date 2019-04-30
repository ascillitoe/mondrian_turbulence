import numpy as np
import pandas as pd
import os
from joblib import dump, load
import matplotlib.pyplot as plt
import vtki
from cfd2ml.base import CaseData

def predict(q_data,modelname,loadloc='.',clf=None):

    ##################################
    # Load classifier if not passed in
    ##################################
    if clf is None:
        filename = modelname + '.joblib'
        filename = os.path.join(loadloc, filename)
        print('\nLoading classifer from ', filename)
        clf = load(filename) 

    ####################
    # Check feature data
    ####################
    #TODO - check for missing values and NaN's
    
    # Find feature and target headers
    X = q_data.pd
    X_headers = X.columns
    nX = X_headers.size

    ###################################
    # Predict Y using loaded classifier
    ###################################
    print('\nUse trained classifier to predict targets based on test features')
    Y_pred = clf.predict(X)
    Y_pred = Y_pred.toarray()
    nY = Y_pred.shape[1]

    # Save prediction to new CaseData object
    new_data = CaseData('model_' + modelname + '_qdat_' + q_data.name) 
    new_data.pd  = pd.DataFrame(Y_pred)  #Add to pandas dataframe

    new_data.vtk = vtki.UnstructuredGrid(q_data.vtk.offset,q_data.vtk.cells,q_data.vtk.celltypes,q_data.vtk.points) #init new vtk grid with same cells and points as q_data.vtk
    new_data.vtk.point_arrays['Y_pred'] = Y_pred # Add Y_pred into new vtk grid

    return new_data


def predict_and_compare(q_data,e_data,modelname,loadloc='.',clf=None,accuracy=False):
    from sklearn.metrics import accuracy_score
    if(accuracy==True): 
        from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_curve
        from cfd2ml.utilities import print_cm

    ##################################
    # Load classifier if not passed in
    ##################################
    if clf is None:
        filename = modelname + '.joblib'
        filename = os.path.join(loadloc, filename)
        print('\nLoading classifer from ', filename)
        clf = load(filename) 

    ####################
    # Check feature data
    ####################
    #TODO - check for missing values and NaN's
    
    # Find feature and target headers
    X = q_data.pd
    X_headers = X.columns
    nX = X_headers.size

    Y_true = e_data.pd
    Y_headers = Y_true.columns
    nY = Y_headers.size

    ###################################
    # Predict Y using loaded classifier
    ###################################
    print('\nUse trained classifier to predict targets based on test features')
    Y_pred = clf.predict(X)
    Y_pred = Y_pred.toarray()

    # Save prediction to new CaseData object
    new_data = CaseData('model_' + modelname + '_qdat_' + q_data.name) 

    new_data.vtk = vtki.UnstructuredGrid(q_data.vtk.offset,q_data.vtk.cells,q_data.vtk.celltypes,q_data.vtk.points) #init new vtk grid with same cells and points as q_data.vtk
    new_data.vtk.point_arrays['Y_pred'] = Y_pred # Add Y_pred into new vtk grid

    Y_pred = pd.DataFrame(Y_pred,columns=Y_headers)
    new_data.pd  = pd.DataFrame(Y_pred)  #Add to pandas dataframe

    ###################################################
    # Compare to actual test data to determine accuracy
    ###################################################
    new_data.vtk.point_arrays['Y_true'] = Y_true.values # Add Y_true into the new vtk grid
   
    # Classifier accuracy
    print('\nSubset accuracy of classifier =  %.2f %%' %(accuracy_score(Y_true,Y_pred)*100) )
    print('\nClassifier accuracy for each target:')
    for label in Y_headers:
        print('%s: %.2f %%' %(label, accuracy_score(Y_true[label],Y_pred[label])*100) )

    if (accuracy==True):
        # Classification report (f1-score, recall etc)
        print('\nClassification report for each target:')
        i = 0
        for label in Y_headers:
            print('\nTarget %d: %s' %(i+1,label))
            print(classification_report(Y_true[label], Y_pred[label],target_names=['Off','On']) )
        
        # Confusion matrix
        confuse_mat = multilabel_confusion_matrix(Y_true, Y_pred)
        print('\nConfusion matrices:')
        i = 0
        for label in Y_headers:
            print('\nTarget %d: %s' %(i+1,label))
            print_cm(confuse_mat[i,:,:], ['Off','On'])
            i += 1
        
        # Plot precision-recall curve
        print('\nPlotting precision-recall score')
        Y_score = clf.predict_proba(X)
        Y_score = pd.DataFrame(Y_score.toarray(),columns=Y_headers)

        for label in Y_headers:
            precision, recall, thresholds = precision_recall_curve(Y_true[label], Y_score[label])
            plt.step(recall, precision, alpha=0.5,where='post',label=label)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curves')
        plt.legend()
        plt.show()


    return new_data
