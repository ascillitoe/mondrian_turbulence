def main():
    import sys
    import json

    # Get command-line arguments
    inputfile = sys.argv[1]

    # Read json file
    with open(inputfile) as json_file:
        json_dat = json.load(json_file)
   
    # What is the task? preproc...
    task = json_dat['task']

    # Perform requested task
    ########################
    # Pre-processing task
    if (task=='preproc'):
        type = json_dat['type'] 
        if (type==1):
            preproc1(json_dat)
        elif (type==2):
            preproc2(json_dat)
        elif (type==3):
            preproc3(json_dat)

    elif(task=='train'):
        type = json_dat['type'] 
        if (type=="classification"):
            classify(json_dat)


def preproc1(json):
    import os
    from cfd2ml.base import CaseData
    from cfd2ml.preproc import preproc_RANS_and_HiFi
    from cfd2ml.utilities import convert_rans_fields, convert_hifi_fields

    # Create output dir if needed
    outdir = json['Output directory']
    os.makedirs(outdir, exist_ok=True)

   # Loop through cases, perform preprocessing of CFD data
    for case in json['Cases']:
        id = case['Case ID']
        name = case['Name']

        RANSfile = case['RANS file']
        HiFifile = case['HiFi file']
        arraynames = case['HiFi array names']
        
        if (("options" in case)==True):
            options = case['options']

        # Read data
        X_data = CaseData(name)
        Y_data = CaseData(name)
        print('Reading X data from vtk file: ', RANSfile)
        X_data.ReadVTK(RANSfile)
        print('Reading Y data from vtk file: ', HiFifile)
        Y_data.ReadVTK(HiFifile)

        # Convert RANS and HiFi array names
        X_data.vtk = convert_rans_fields(X_data.vtk)
        Y_data.vtk = convert_hifi_fields(Y_data.vtk,arraynames)

        # Run preproc
        X_data, Y_data = preproc_RANS_and_HiFi(X_data, Y_data, **options)

        # Write data
        X_data.Write(os.path.join(outdir, id + '_X')) 
        Y_data.Write(os.path.join(outdir, id + '_Y'))


def classify(json):
    import os
    from cfd2ml.base import CaseData
    from cfd2ml.classify import RF_classifier
    from joblib import dump
    import pandas as pd

    modelname  = json['save_model']
    datloc     = json['training_data_location']
    cases      = json['training_data_cases']
    target     = json['classifier_target']

    if (("options" in json)==True):
        options    = json['options']
    else: 
        options = None

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
        X_data = X_data.append(X_case.pd)
        Y_data = Y_data.append(Y_case.pd)

    # Train classifier
    rf_clf =  RF_classifier(X_data,Y_data[target],options=options) 

    # Save classifier
    filename = modelname + '.joblib'
    print('\nSaving classifer to ', filename)
    dump(rf_clf, filename) 

if __name__ == '__main__':
    main()


