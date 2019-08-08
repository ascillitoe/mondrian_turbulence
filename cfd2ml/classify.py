import numpy as np
import pandas as pd

import os

from joblib import dump, load

import matplotlib.pyplot as plt

from cfd2ml.base import CaseData

def classify(json):
    from cfd2ml.classify import RF_classifier

    print('\n-----------------------')
    print('Started training')
    print('Type: Classification')
    print('-----------------------')

    modelname  = json['save_model']
    datloc     = json['training_data_location']
    cases      = json['training_data_cases']
    target     = json['classifier_target']

    options = None
    sample  = None
    if (("options" in json)==True):
        options    = json['options']
        if(("sample" in options)==True):
            sample = options['sample']

    # Read data
    X_data = pd.DataFrame()
    Y_data = pd.DataFrame()

    # Read in each data set, append into single X and Y df's to pass to classifier. 
    # "group" identifier column is added to X_case to id what case the data came from. 
    # (Used in LeaveOneGroupOut CV later)
    caseno = 1
    for case in cases:
        # Read in RANS (X) data
        filename = os.path.join(datloc,case+'_X.pkl')
        X_case = CaseData(filename)

        # Add "group" id column
        X_case.pd['group'] = caseno
        caseno += 1

        # Read in HiFi (Y) data
        filename = os.path.join(datloc,case+'_Y.pkl')
        Y_case = CaseData(filename)

        # Add X and Y data to df's
        X_data = X_data.append(X_case.pd,ignore_index=True)
        Y_data = Y_data.append(Y_case.pd,ignore_index=True)

    # Randomly sample a % of the data
    nrows = X_data.shape[0]
    print('Original number of rows in dataset: = ', nrows)
    if(sample is not None):
        index = np.random.choice(X_data.index,int(sample*nrows))
        X_data = X_data.iloc[index]
        Y_data = Y_data.iloc[index]
        nrows = len(X_data.index)
        print('Number of rows in dataset after sampling: = ', nrows)

    # Train classifier
    rf_clf =  RF_classifier(X_data,Y_data[target],options=options) 

    # Save classifier
    filename = modelname + '.joblib'
    print('\nSaving classifer to ', filename)
    dump(rf_clf, filename) 

    print('\n-----------------------')
    print('Finished training')
    print('-----------------------')

def RF_classifier(X_data,Y_data,options=None):
    from sklearn.ensemble import RandomForestClassifier

    ####################
    # Parse user options
    ####################
    params = {}
    gridsearch   = False
    GS_settings  = None
    randomsearch = False
    RS_settings  = None
    accuracy = False
    cv_type = 'logo'
    scoring = 'f1'

    if (options is not None):

        if (("RF_parameters" in options)==True):
            params = options['RF_parameters']

        if (("grid_search" in options)==True):
            from sklearn.model_selection import GridSearchCV
            gridsearch = True
            GS_params   = options['grid_search']['parameter_grid']
            if (("settings" in options['grid_search'])==True): GS_settings = options['grid_search']['settings'] 

        if (("random_search" in options)==True):
            from sklearn.model_selection import RandomizedSearchCV
            from cfd2ml.utilities import convert_param_dist
            randomsearch = True
            RS_params, RS_Nmax   = convert_param_dist(options['random_search']['parameter_grid'])
            print('RS_Nmax = ', RS_Nmax)
            if (("settings" in options['random_search'])==True): RS_settings = options['random_search']['settings'] 

        if(randomsearch==True and gridsearch==True): quit('********** Stopping! grid_search and random_search both set *********')

        if (("accuracy" in options)==True):
            accuracy = options['accuracy']
            if (accuracy==True):
                from sklearn.model_selection import cross_validate
                from sklearn.metrics import precision_recall_curve, auc, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
                from cfd2ml.utilities import print_cm

        if (("scoring" in options)==True):
            scoring = options['scoring']

        if (("cv_type" in options)==True):
            cv_type = options['cv_type']

    ##############
    # Prepare data
    ##############
    # Find feature and target headers
    X_headers = X_data.columns
    Y_header  = Y_data.name

    if(cv_type=='logo'): groups = X_data['group']
    X_data = X_data.drop(columns='group')
   
    nX = X_headers.size
    print('\nFeatures:')
    for i in range(0,nX):
        print('%d/%d: %s' %(i+1,nX,X_headers[i]) )
    print('\nTarget: ', Y_header)
  
    ########################
    # Prepare other settings
    ########################
    # Setting cross-validation type (either leave-one-group-out or 5-fold)
    if(cv_type=='logo'):
        from sklearn.model_selection import LeaveOneGroupOut
        logo = LeaveOneGroupOut()
        ngroup = logo.get_n_splits(groups=groups)
        print('\nUsing Leave-One-Group-Out cross validation on ', ngroup, ' groups')
    elif(cv_type=='kfold'):
        from sklearn.model_selection import StratifiedKFold
        print('\nUsing 5-fold cross validation')
        k_fold = StratifiedKFold(n_splits=10, random_state=42,shuffle=True)
        cv = k_fold.split(X_data,Y_data)

    #########################
    # Training the classifier
    #########################
    # TODO TODO TODO - improve accuracy by using balanced or weighted random forest
    # (see https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf)
    if(gridsearch==True):
        # Finding optimal hyperparameters with GridSearchCV
        print('\n Performing GridSearchCV to find optimal hyperparameters for random forest classifier')
        clf = RandomForestClassifier(**params,random_state=42)
        if (cv_type=='logo'): cv = logo.split(X_data,Y_data,groups)
        GS_clf = GridSearchCV(estimator=clf,param_grid=GS_params, cv=cv, scoring=scoring, iid=False, verbose=2, **GS_settings)
        GS_clf.fit(X_data,Y_data)

        # Write out results to file
        scores_df = pd.DataFrame(GS_clf.cv_results_)#.sort_values(by='rank_test_score')
        scores_df.to_csv('GridSearch_results.csv')

        # Pich out best results
        best_params = GS_clf.best_params_
        best_score  = GS_clf.best_score_
        clf = GS_clf.best_estimator_  # (this clf has been fit to all of the X_data,Y_data)

        print('\nBest hyperparameters found:', best_params)
        print('\nScore with these hyperparameters:', best_score)

    elif(randomsearch==True):
        # Finding optimal hyperparameters with RandomSearchCV
        print('\n Performing RandomizedSearchCV to find optimal hyperparameters for random forest classifier')
        clf = RandomForestClassifier(**params,random_state=42)
        if (cv_type=='logo'): cv = logo.split(X_data,Y_data,groups)
        RS_clf = RandomizedSearchCV(estimator=clf,param_distributions=RS_params, cv=cv, scoring=scoring,iid=False, verbose=2, error_score=np.nan, **RS_settings)
        RS_clf.fit(X_data,Y_data)
        
        # Write out results to file
        scores_df = pd.DataFrame(RS_clf.cv_results_)#.sort_values(by='rank_test_score')
        scores_df.to_csv('RandomSearch_results.csv')

        # Pick out best results
        best_params = RS_clf.best_params_
        best_score  = RS_clf.best_score_
        clf = RS_clf.best_estimator_  # (this clf has been fit to all of the X_data,Y_data)

        print('\nBest hyperparameters found:', best_params)
        print('\nScore with these hyperparameters:', best_score)


    else:
        # Train RF classifier with hyperparameters given by user
        print('\nTraining random forest classifer with given hyperparameters')
        clf = RandomForestClassifier(**params)
        clf.fit(X_data,Y_data)

    # Cross validation accuracy metrics
    if(accuracy==True):
        print('\nPerforming cross validation to determine train and test accuracy/error, and precision-recall curves')

        #TODO - capability to decide on probablity threshold, and predict with chosen threshold

        # Get generator object depending on cv strategy
        if (cv_type=='logo'): 
            cv = logo.split(X_data,Y_data,groups)
        elif(cv_type=='kfold'):
            cv = k_fold.split(X_data,Y_data)  # Need to regen "Generator" object

        fig1, ax1 = plt.subplots()

        # Init lists
        y_real   = []
        y_proba  = []
        train_f1 = []
        test_f1  = []
        train_A  = []
        test_A   = []
        train_BA = []
        test_BA  = []

        # Loop through CV folds
        i = 0
        for train_index, test_index in cv:
            X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
            Y_train, Y_test = Y_data.iloc[train_index], Y_data.iloc[test_index]

            # Train classifier
            clf_cv = clf
            clf_cv.fit(X_train, Y_train)

            # Predict Y
            Y_pred_train = clf_cv.predict(X_train)
            Y_pred_test  = clf_cv.predict(X_test )

            # F1 scores
            f1score = f1_score(Y_test , Y_pred_test)
            train_f1.append(f1_score(Y_train, Y_pred_train) )
            test_f1.append(f1score)
            # Accuracy scores
            Ascore = accuracy_score(Y_test , Y_pred_test)
            train_A.append(accuracy_score(Y_train, Y_pred_train) )
            test_A.append(Ascore)
            # Balanced accuracy scores
            BAscore = balanced_accuracy_score(Y_test , Y_pred_test)
            train_BA.append(balanced_accuracy_score(Y_train, Y_pred_train) )
            test_BA.append(BAscore)

            # Print validation scores (training scores are stored to print mean later, but not printed for each fold)
            if(cv_type=='logo'):
                print('\nTest group = ', groups.iloc[test_index[0]])
            elif(cv_type=='kfold'):
                print('\nFold = ', i)
            print('-------------------')
            print('F1 score = %.2f %%' %(f1score*100) )
            print('Total error = %.2f %%' %((1.0-Ascore)*100) )
            print('Per-class error = %.2f %%' %((1.0-BAscore)*100) )

            # Print confusion matrix for this fold
            print('Confusion matrix:')
            confuse_mat = confusion_matrix(Y_test, Y_pred_test)
            print_cm(confuse_mat, ['Off','On'])
            
            # Prediction probability based on X_test (used for precision-recall curves)
            pred_proba = clf_cv.predict_proba(X_test)
            precision, recall, _ = precision_recall_curve(Y_test, pred_proba[:,1])
            lab = 'Fold %d AUC=%.4f' % (i+1, auc(recall, precision))
            ax1.step(recall, precision, label=lab)
            y_real.append(Y_test)
            y_proba.append(pred_proba[:,1])

            i += 1

        # Calculate errors from accuracies
        train_TE = 1.0 -  np.array(train_A)
        test_TE  = 1.0 -  np.array(test_A)
        train_CAE = 1.0 - np.array(train_BA)
        test_CAE  = 1.0 - np.array(test_BA)

        # Print performance scores
        print('\nMean training scores:')
        print('F1 score = %.2f %%' %(np.mean(train_f1)*100) )
        print('Total error = %.2f %%' %(np.mean(train_TE)*100) )
        print('Per-class error = %.2f %%' %(np.mean(train_CAE)*100) )
    
        print('\nMean validation scores:')
        print('F1 score = %.2f %%' %(np.mean(test_f1)*100) )
        print('Total error = %.2f %%' %(np.mean(test_TE)*100) )
        print('Per-class error = %.2f %%' %(np.mean(test_CAE)*100) )

        
        # Average precision-recall over folds, and plot curves
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        precision, recall, _ = precision_recall_curve(y_real, y_proba)
        lab = 'Overall AUC=%.4f' % (auc(recall, precision))
        ax1.step(recall, precision, label=lab, lw=2, color='black')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.legend(loc='lower left', fontsize='small')
        

        plt.show()

    return clf
