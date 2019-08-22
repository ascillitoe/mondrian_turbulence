import numpy as np
import pandas as pd
import os
from joblib import dump, load
import matplotlib.pyplot as plt
import vista
from cfd2ml.base import CaseData
from sklearn.metrics import precision_recall_curve, auc
plt.rcParams.update({'font.size': 18})

def predict(json):

    print('\n-----------------------')
    print('Started prediction')
    print('-----------------------')

    model  = json['model']
    datloc = json['data_location']
    cases  = json['data_cases']
    savloc = json['save_location']
    os.makedirs(savloc, exist_ok=True)

    compare_predict = False
    if (("prediction_accuracy" in json)==True): 
        compare_predict = True
        target = json['prediction_accuracy']['target']
        type   = json['prediction_accuracy']['type']

    # Read in ML model
    filename = model + '.joblib'
    print('\nReading model from ', filename)
    model = load(filename) 

    # Read in each X_data, predict Y, write predicted Y
    caseno = 1
    for case in cases:
        # Read in RANS (X) data
        filename = os.path.join(datloc,case+'_X.pkl')
        X_case = CaseData(filename)

        print('\n***********************')
        print(' Case: ', caseno)
        print('***********************')

        # Predict HiFi (Y) data and store add to vtk
        Y_pred = CaseData(case + '_pred') 
        Y_pred.pd = pd.Series(model.predict(X_case.pd)) # only need as numpy ndarray but convert to pd series for consistency 
        Y_pred.vtk = vista.UnstructuredGrid(X_case.vtk.offset,X_case.vtk.cells,X_case.vtk.celltypes,X_case.vtk.points)
        Y_pred.vtk.point_arrays['Y_pred'] = Y_pred.pd.to_numpy()

        Y_prob = pd.Series(model.predict_proba(X_case.pd)[:,1]) # only need as numpy ndarray but convert to pd series for consistency 
        Y_pred.vtk.point_arrays['Y_prob'] = Y_prob.to_numpy()

        # Read in true HiFi (Y) data and compare to predict
        if (compare_predict==True):
            filename = os.path.join(datloc,case+'_Y.pkl')
            Y_true = CaseData(filename)

            # Write Y_true to vtk for analysis
            index = Y_true.pd.columns.get_loc(target)
            Y_pred.vtk.point_arrays['Y_true'] = Y_true.pd.to_numpy()[:,index]

            # accuracy metrics
            if (type=='classification'):
                predict_classifier_accuracy(Y_pred.pd,Y_true.pd[target])
            elif(type=='regression'):
                quit('Regression not implemented yet')

            # Write TP, TN, FP, FN to vtk
            if (type=='classification'):
                Y_pred.vtk.point_arrays['confuse'] = confusion_labels(Y_pred.pd, Y_true.pd[target])
            elif(type=='regression'):
                quit('Regression not implemented yet')

            precision, recall, _ = precision_recall_curve(Y_true.pd[target], Y_prob)
            lab = 'Case %s AUC=%.4f' % (case, auc(recall, precision))
            plt.step(recall, precision, label=lab)    

        filename = os.path.join(savloc,Y_pred.name + '.vtk')
        Y_pred.WriteVTK(filename)

        caseno += 1

    plt.legend()
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.show()

    print('\n-----------------------')
    print('Finished prediction')
    print('-----------------------')
    

def predict_classifier_accuracy(Y_pred,Y_true):
    # Y_pred and Y_true are pandas dataframes
 
    from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
    from cfd2ml.utilities import print_cm

    # F1 scores
    f1score = f1_score(Y_true , Y_pred)
    # Accuracy scores
    Ascore = accuracy_score(Y_true , Y_pred)
    # Balanced accuracy scores
    BAscore = balanced_accuracy_score(Y_true , Y_pred)

    # Print validation scores (training scores are stored to print mean later, but not printed for each fold)
    print('F1 score = %.2f %%' %(f1score*100) )
    print('Total error = %.2f %%' %((1.0-Ascore)*100) )
    print('Per-class error = %.2f %%' %((1.0-BAscore)*100) )

    # Print confusion matrix for this fold
    print('Confusion matrix:')
    confuse_mat = confusion_matrix(Y_true, Y_pred)
    print_cm(confuse_mat, ['Off','On'])


def confusion_labels(Y_pred, Y_true):
    # Y_pred and Y_true are vtk obj's

#    new_data.vtk.point_arrays['Y_true'] = Y_true.values # Add Y_true into the new vtk grid
    # construct array of classes [1,2,3,4] for True +ve, True -ve, False +ve, False -ve
    true = Y_true.to_numpy()#.values
    pred = Y_pred.to_numpy()#.values
    confuse = np.zeros_like(true)
    TP = np.where((true==1) & (pred==1))
    TN = np.where((true==0) & (pred==0))
    FP = np.where((true==1) & (pred==0))
    FN = np.where((true==0) & (pred==1))
    confuse[TP] = 1
    confuse[TN] = 2
    confuse[FP] = 3
    confuse[FN] = 4

    return confuse
