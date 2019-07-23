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
        preproctype = json_dat['type'] 
        if (preproctype==1):
            preproc1(json_dat)
        elif (preproctype==2):
            preproc2(json_dat)
        elif (preproctype==3):
            preproc3(json_dat)

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
        X_data = CaseData(id + '_X')
        Y_data = CaseData(id + '_Y')
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
        X_data.Write(fileloc=outdir) 
        Y_data.Write(fileloc=outdir) 

if __name__ == '__main__':
    main()


