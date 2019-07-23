# Doing this as want to store vista and pandas in one class, so can pass around between preproc, classifiy and predict easily
import vista
import pandas as pd
import os
import pickle

# CaseData object to store pandas and vista objects together, with read and write capability                                                                                                                                                                                 
class CaseData:
    def __init__(self, name):

        self.name = None
        self.pd   = None
        self.vtk  = None
        self.vtkfile = None
        self.csvfile = None

        if ('.pkl' in name): #if /.pkl in name then we want to read meta data from pkl file
            self.Read(name)
        else:
            self.name = name #otherwise init an empty CaseData object with self.name = name

    def ReadVTK(self,filename):
        print("Reading vtk data from " + filename + " into CaseData called " + self.name)
        self.vtk     = vista.read(filename)
        self.vtkfile = os.path.abspath(filename)

    def ReadPandas(self,filename):
        print("Reading csv data from " + filename + " into CaseData called " + self.name)
        self.pd = pd.read_csv(filename)
        self.csvfile = os.path.abspath(filename)

    def WriteVTK(self,filename):
        print("Writing vtk data from " + self.name + " CaseData to file called " + filename)
        self.vtk.save(filename)
        self.vtkfile = os.path.abspath(filename)

    def WritePandas(self,filename,index=False):
        print("Writing pandas DataFrame from " + self.name + " CaseData to file called " + filename)
        self.pd.to_csv(filename,index=index)
        self.csvfile = os.path.abspath(filename)

    def WritePandasStats(self,filename):
        print('Writing statistics of pandas data from ' + self.name + ' CaseData to ', filename)
        self.pd.describe().to_csv(filename,index='True')

    def PrintPandasStats(self):
        print('Statistics of pandas data from ' + self.name + ' CaseData:')
        print(self.pd.describe())

    def Write(self,fileloc='.'):
        filename = os.path.join(fileloc,self.name + '.pkl')
        print('Saving metadata for CaseData ' + self.name + ' to file ' + filename)

        mydict = {'name':self.name}

        self.vtkfile = os.path.abspath(os.path.join(fileloc,self.name + '.vtk'))
        self.WriteVTK(self.vtkfile)
        self.csvfile = os.path.abspath(os.path.join(fileloc,self.name + '.csv'))
        self.WritePandas(self.csvfile)

        mydict['vtk file'] = self.vtkfile 
        mydict['csv file'] = self.csvfile
        output = open(filename, 'wb')
        pickle.dump(mydict, output) 
        output.close()

    def Read(self,filename):
        print('Reading metadata for CaseData from ' + filename)

        pkl_file = open(filename, 'rb')
        mydict = pickle.load(pkl_file)
        pkl_file.close()

        self.name    = mydict.get('name')
        self.vtkfile = mydict.get('vtk file')
        self.csvfile = mydict.get('csv file')

        print('Name of CaseData object: ' + self.name)

        self.ReadVTK(self.vtkfile)
        self.ReadPandas(self.csvfile)
