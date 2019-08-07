def main():
    import sys
    import json
    from cfd2ml.preproc import preproc1
    from cfd2ml.classify import classify
    from cfd2ml.predict import predict

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

    elif(task=='predict'):
        predict(json_dat)


if __name__ == '__main__':
    main()
