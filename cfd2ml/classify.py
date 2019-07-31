import numpy as np
import pandas as pd

import os

from joblib import dump, load

import matplotlib.pyplot as plt

def RF_classifier(X_data,Y_data,options=None):
    from sklearn.ensemble import RandomForestClassifier

    ####################
    # Parse user options
    ####################
    params = None
    gridsearch = False
    accuracy = False
    cv_type = 'logo'
    scoring = 'f1'

    if (options is not None):

        if (("RF_parameters" in options)==True):
            params = options['RF_parameters']

        if (("grid_search" in options)==True):
            from sklearn.model_selection import GridSearchCV
            gridsearch = True
            gridsearch_opts = options['grid_search']
            gridsearch_params = gridsearch_opts['parameter_grid']

        if (("accuracy" in options)==True):
            accuracy = True
            from sklearn.model_selection import cross_validate
            from sklearn.metrics import precision_recall_curve, auc, f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix
            from cfd2ml.utilities import print_cm

        if (("scoring" in options)==True):
            scoring = options['scoring']

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
        from sklearn.model_selection import kfold
        print('\nUsing 5-fold cross validation')
        cv = 5

    #########################
    # Training the classifier
    #########################
    # TODO TODO TODO - improve accuracy by using balanced or weighted random forest
    # (see https://statistics.berkeley.edu/sites/default/files/tech-reports/666.pdf)
    if(gridsearch==True):
        # Optimal hyperparameters found with GridSearchCV
        print('\n Performing GridSearchCV to find optimal hyperparameters for random forest classifier')
        clf = RandomForestClassifier()
        if (cv_type=='logo'): cv = logo.split(X_data,Y_data,groups)
        GS_clf = GridSearchCV(estimator=clf,param_grid=gridsearch_params, cv=cv, scoring=scoring, n_jobs=-1, iid=False, verbose=1)
        GS_clf.fit(X_data,Y_data)
        
        best_params = GS_clf.best_params_
        best_score  = GS_clf.best_score_
        clf = GS_clf.best_estimator_  # (this clf has been fit to all of the X_data,Y_data)

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
            k_fold = KFold(n_splits=cv)
            k_fold.split(X_data)

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
        

# TODO - Training and validation error versus hyper-parameters if gridsearch

        plt.show()

    return clf



#
#
#    else:
#        # Hyperparameters chosen by user
#        # Hold out user specificied amount of data to get training accuracy
#        clf = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, random_state = random_state,verbose=True)
#        clf.fit(X_train, Y_train)
#
#    # Accuracy metrics and plots
#    ############################
#    clf.verbose = False #Turn verbose off after this to tidy prints
#
#    # Predict test set for accuracy evaluation
#    print('\nUse trained classifier to predict targets based on test features')
#    Y_pred = clf.predict(X_test)
#    Y_pred = pd.Series(Y_pred,name=Y_header)
#    
#    # Classifier accuracy
#    print('\nSubset accuracy of classifier =  %.2f %%' %(accuracy_score(Y_test,Y_pred)*100) )
#    print('\nClassifier accuracy:')
#    print('%s: %.2f %%' %(Y_header, accuracy_score(Y_test,Y_pred)*100) )
#
#    if (accuracy==True):
#        # Classification report (f1-score, recall etc)
#        print('\nClassification report for each target:')
#        print(classification_report(Y_test, Y_pred,target_names=['Off','On']) )
#        
#        # Confusion matrix
#        print('\nConfusion matrix:')
#        confuse_mat = confusion_matrix(Y_test, Y_pred)
#        print_cm(confuse_mat, ['Off','On'])
#        
#        # Plot precision-recall curve
#        print('\nPlotting precision-recall score')
#        Y_score = clf.predict_proba(X_test)
#        Y_score = pd.Series(Y_score,name=Y_header)
#
#        precision, recall, thresholds = precision_recall_curve(Y_test, Y_score)
#        plt.step(recall, precision, alpha=0.5,where='post',label=Y_header)
#        
#        plt.xlabel('Recall')
#        plt.ylabel('Precision')
#        plt.ylim([0.0, 1.05])
#        plt.xlim([0.0, 1.0])
#        plt.title('Precision-Recall curve')
#        plt.legend()
#        plt.show()
#
#    clf.verbose = True # reset


def DT_classifier(q_dataset,e_dataset,accuracy=False,train_percentage=0.75,criterion='entropy',random_state=42,max_depth=5):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    if(accuracy==True): 
        from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
        from cfd2ml.utilities import print_cm

    ##############
    # Prepare data
    ##############
    #TODO - check for missing values and NaN's
    
    # Find feature and target headers
    X_headers = q_dataset.columns
    Y_headers = e_dataset.columns
    
    nX = X_headers.size
    nY = Y_headers.size
    
    print('\nFeatures:')
    for i in range(0,nX):
        print('%d/%d: %s' %(i+1,nX,X_headers[i]) )
    print('\nTargets:')
    for i in range(0,nY):
        print('%d/%d: %s' %(i+1,nY,Y_headers[i]) )
    
    # Split dataset into train and test dataset
    print('\nSpliting into test and training data. Training split = ', train_percentage*100, '%')
    X_train, X_test, Y_train, Y_test = train_test_split(q_dataset[X_headers], e_dataset[Y_headers],train_size=train_percentage,random_state=random_state)

    print('Number of observations for training data = ', X_train.shape[0])
    print('Number of observations for test data = ',  X_test.shape[0])
    
    ## Scale features NOTE - is this needed? Or is orig non-dim enough?
    ## Feature Scaling
    #print('\nScaling features')
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test  = scaler.transform(X_test)
    #X_train = pd.DataFrame(X_train,columns=X_headers)
    #X_test  = pd.DataFrame( X_test,columns=X_headers)
    
    # Train decision tree classifier on training data
    print('\nTraining decision tree classifier on training data...')
    clf = DecisionTreeClassifier(criterion = criterion, random_state = random_state,min_samples_leaf=1,max_depth=max_depth)
    clf.fit(X_train, Y_train)

    # Accuracy metrics and plots
    ############################
    clf.verbose = False #Turn verbose off after this to tidy prints

    # Predict test set for accuracy evaluation
    print('\nUse trained classifier to predict targets based on test features')
    Y_pred = clf.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred.toarray(),columns=Y_headers)
    
    # Classifier accuracy
    print('\nSubset accuracy of classifier =  %.2f %%' %(accuracy_score(Y_test,Y_pred)*100) )
    print('\nClassifier accuracy for each target:')
    for label in Y_headers:
        print('%s: %.2f %%' %(label, accuracy_score(Y_test[label],Y_pred[label])*100) )

    if (accuracy==True):
        # Classification report (f1-score, recall etc)
        print('\nClassification report for each target:')
        i = 0
        for label in Y_headers:
            print('\nTarget %d: %s' %(i+1,label))
            print(classification_report(Y_test[label], Y_pred[label],target_names=['Off','On']) )
        
        # Confusion matrix
        print('\nConfusion matrices:')
        i = 0
        for label in Y_headers:
            confuse_mat = confusion_matrix(Y_test[label], Y_pred[label])
            print('\nTarget %d: %s' %(i+1,label))
            print_cm(confuse_mat, ['Off','On'])
            i += 1
        
        # Plot precision-recall curve
        print('\nPlotting precision-recall score')
        Y_score = clf.predict_proba(X_test)
        Y_score = pd.DataFrame(Y_score.toarray(),columns=Y_headers)

        for label in Y_headers:
            precision, recall, thresholds = precision_recall_curve(Y_test[label], Y_score[label])
            plt.step(recall, precision, alpha=0.5,where='post',label=label)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall curves')
        plt.legend()
        plt.show()

    clf.verbose = True # reset

    return clf, X_train.index, X_test.index
