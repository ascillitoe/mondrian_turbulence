import numpy as np
import pandas as pd

import os

from joblib import dump, load

import matplotlib.pyplot as plt
from skmultilearn.problem_transform import BinaryRelevance

def RF_classifier(q_dataset,e_dataset,modelname,saveloc=None,accuracy=False,train_percentage=0.75,n_estimators=10,criterion='entropy',random_state=42):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    if(accuracy==True): 
        from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_curve
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
    
    # Train random forest classifier on training data
    print('\nTraining Random Forest classifier on training data...')
    clf = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, random_state = random_state,verbose=True)
    clf = BinaryRelevance(classifier=clf,require_dense=[False,True])
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
        confuse_mat = multilabel_confusion_matrix(Y_test, Y_pred)
        print('\nConfusion matrices:')
        i = 0
        for label in Y_headers:
            print('\nTarget %d: %s' %(i+1,label))
            print_cm(confuse_mat[i,:,:], ['Off','On'])
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

    # Save classifier
    #################
    if (saveloc is not None):
        filename = modelname + '.joblib'
        filename = os.path.join(saveloc, filename)
        print('\nSaving classifer to ', filename)
        dump(clf, filename) 

    clf.verbose = True # reset

    return clf, X_train.index, X_test.index
