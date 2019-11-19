import numpy as np
import pandas as pd
import os
#from joblib import dump, load
import pickle
import matplotlib.pyplot as plt
import pyvista as vista
from cfd2ml.base import CaseData
from cfd2ml.utilities import plot_precision_recall_threshold, plot_precision_recall_vs_threshold
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble.forest import RandomForestRegressor
from skgarden import MondrianForestRegressor

plt.rcParams.update({'font.size': 18})

def predict(json):

    print('\n-----------------------')
    print('Started prediction')
    print('-----------------------')

    modelname  = json['model']
    datloc = json['data_location']
    cases  = json['data_cases']
    savloc = json['save_location']

    os.makedirs(savloc, exist_ok=True)

    compare_predict = False
    if (("prediction_accuracy" in json)==True): 
        compare_predict = True
        target = json['prediction_accuracy']['target']
        type   = json['prediction_accuracy']['type']

    thresh = 0.5
    if (("prediction_threshold" in json)==True): 
        thresh = json['prediction_threshold']

    if(("features_to_drop" in json)==True):
        features_to_drop = json['features_to_drop']
    else:
        features_to_drop = None

    if(("uq" in json)==True):
        uq = True
        import forestci as fci
    else:
        uq = False

    # Read in ML model
#    filename = modelname + '.joblib'
    filename = modelname + '.p'
    print('\nReading model from ', filename)
#    model = load(filename) 
    model = pickle.load(open(filename, 'rb'))
    # TEMP
    X = np.load('X.npy')
    Y = np.load('Y.npy')
    model.fit(X,Y) 
    # TEMP END
    cmap = plt.get_cmap('tab10')
    # Open a figure axes
    fig1, ax1 = plt.subplots() 
    fig2, ax2 = plt.subplots()

    # Read in each X_data, predict Y, write predicted Y
    for caseno, case in enumerate(cases):
        # Read in RANS (X) data
        filename = os.path.join(datloc,case+'_X.pkl')
        X_case = CaseData(filename)

        print('\n***********************')
        print(' Case %d: %s ' %(caseno+1,case) )
        print('***********************')

        X_pred = X_case.pd
        if (features_to_drop is not None): 
            X_pred = X_pred.drop(columns=features_to_drop)

        # Predict HiFi (Y) data and store add to vtk
        Y_pred = CaseData(case + '_pred') 
        Y_pred.vtk = vista.UnstructuredGrid(X_case.vtk.offset,X_case.vtk.cells,X_case.vtk.celltypes,X_case.vtk.points)
        if (type=='classification'):
            Y_prob = pd.Series(model.predict_proba(X_pred)[:,1]) # only need as numpy ndarray but convert to pd series for consistency 
            Y_pred.pd = pd.Series(predict_with_threshold(Y_prob, thresh))
            Y_pred.vtk.point_arrays['Y_prob'] = Y_prob.to_numpy()
        elif(type=='regression'):
            if isinstance(model,RandomForestRegressor):
                y_pred = model.predict(X_pred)
            elif isinstance(model,MondrianForestRegressor) and uq is False: #if uq true prediction made below
                y_pred = model.predict(X_pred)

            # Uncertainty quantification
            if(uq is True):
                filename = modelname + '_Xdat.csv'
                X_train = pd.read_csv(filename)
                if (features_to_drop is not None): X_train = X_train.drop(columns=features_to_drop)
                if isinstance(model,RandomForestRegressor):
                    print('Calculating jackknife infinitesimal variance')
                    y_var = fci.random_forest_error(model, X_train, X_pred,calibrate=True)
                    y_sd = np.sqrt(np.maximum(y_var,0))
                elif isinstance(model,MondrianForestRegressor):
                    print('Calculating mondrian forest posterior mean and standard deviation')
                    y_pred, y_sd = model.predict(X_pred,return_std=True)

                Y_pred.vtk.point_arrays['Y_std'] = y_sd
                # Print out rms of var
                sd_rms = np.sqrt(np.mean(y_sd**2))
                y_rms  = np.sqrt(np.mean(y_pred**2))
                print('sd_rms/y_rms = ', 100*sd_rms/y_rms, '%')

        Y_pred.pd = pd.Series(y_pred) # only need as numpy ndarray but convert to pd series for consistency 
        Y_pred.vtk.point_arrays['Y_pred'] = y_pred

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

                # Write TP, TN, FP, FN to vtk
                if (type=='classification'):
                    Y_pred.vtk.point_arrays['confuse'] = confusion_labels(Y_pred.pd, Y_true.pd[target])

                # Calc precision, recall and decision thresholds
                precisions, recalls, thresholds = precision_recall_curve(Y_true.pd[target], Y_prob)
                c = cmap(caseno)

                # Plot precision-recall curve with current decision threshold marked
                plot_precision_recall_threshold(precisions, recalls, thresholds, t=thresh, ax=ax1,c=c)

                # Plot precision and recall vs decision threshold
                plot_precision_recall_vs_threshold(precisions, recalls, thresholds,ax=ax2, c=c,t=thresh,case=case)

            elif(type=='regression'):
                predict_regressor_accuracy(Y_pred.pd,Y_true.pd[target])
                Y_pred.vtk.point_arrays['error'] = local_error(Y_pred.pd, Y_true.pd[target])

        filename = os.path.join(savloc,Y_pred.name + '.vtk')
        Y_pred.WriteVTK(filename)

    if (type=='classification'):
        ax1.legend()
        ax2.legend()
        plt.show()

    print('\n-----------------------')
    print('Finished prediction')
    print('-----------------------')

def local_error(Y_pred,Y_true):
    err = Y_pred - Y_true
    return err

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

def predict_regressor_accuracy(Y_pred,Y_true):
    # Y_pred and Y_true are pandas dataframes
 
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # r2 scores
    r2score = r2_score(Y_true , Y_pred)
    # Mean absolute error scores
    MAEscore = mean_absolute_error(Y_true , Y_pred)
    # Mean squared error scores
    MSEscore = mean_squared_error(Y_true , Y_pred)

    # Print validation scores (training scores are stored to print mean later, but not printed for each fold)
    print('r2 score = %.2f %%' %(r2score*100) )
    print('Mean absolute error = %.2f %%' %(MAEscore*100) )
    print('Mean squared error = %.2f %%' %(MSEscore*100) )

def confusion_labels(Y_pred, Y_true):
    # Y_pred and Y_true are vtk obj's

#    new_data.vtk.point_arrays['Y_true'] = Y_true.values # Add Y_true into the new vtk grid
    # construct array of classes [1,2,3,4] for True +ve, True -ve, False +ve, False -ve
    true = Y_true.to_numpy()#.values
    pred = Y_pred.to_numpy()#.values
    confuse = np.zeros_like(true)
    TP = np.where((true==1) & (pred==1))
    TN = np.where((true==0) & (pred==0))
    FP = np.where((true==0) & (pred==1))
    FN = np.where((true==1) & (pred==0))
    confuse[TP] = 1
    confuse[TN] = 2
    confuse[FP] = 3
    confuse[FN] = 4

    return confuse

def predict_with_threshold(y_scores, t):
    """
    This function adjusts class predictions based on the prediction threshold (t).
    Will only work for binary classification problems.

    Based on https://towardsdatascience.com/fine-tuning-a-classifier-in-scikit-learn-66e048c21e65

    :param array-like or sparse matrix of shape = [n_samples]:
        Prediction proba from binary classifier
    :param float:
        The decision threshold
    """
    return [1 if y >= t else 0 for y in y_scores]

