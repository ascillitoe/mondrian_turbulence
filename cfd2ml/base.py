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

        if(self.vtk is not None):
            self.vtkfile = os.path.abspath(os.path.join(fileloc,self.name + '.vtk'))
            self.WriteVTK(self.vtkfile)

        
#        if(self.vtk is not None and self.vtkfile is None): #If vtk data, but no vtkfile, then the data is new so write a vtkfile 
#            self.vtkfile = os.path.abspath(os.path.join(fileloc,self.name + '.vtk'))
#            self.WriteVTK(self.vtkfile)
#        elif(self.vtk is not None and self.vtkfile is not None): #if vtk data and file, still check if the vtk grid has been modified (e.g. clipped etc), or new arrays. If yes write a new file.
# THIS IS ALL TOO COMPLICATED... SIMPLIFY!            
#            oldnnode = vista.read(self.vtkfile).number_of_points
#            newnnode = self.vtk.number_of_points
#            oldnarray = vista.read(self.vtkfile).n_scalars
#            newnarray = self.vtk.n_scalars
#            if (newnnode != oldnnode):
#                file = os.path.split(self.vtkfile)
#                self.vtkfile = os.path.abspath(os.path.join(file[0],'new_' + file[1]))
#                print('\nNumber of nodes in vtk grid has changed, writing a modified vtk file: ' + self.vtkfile)
#                print('Old nnode = ', oldnnode)
#                print('New nnode = ', newnnode)
#                self.WriteVTK(self.vtkfile)
#            elif(newnarray!=oldnarray):
#                file = os.path.split(self.vtkfile)
#                self.vtkfile = os.path.abspath(os.path.join(file[0],'new_' + file[1]))
#                print('\nNumber of arrays in vtk data has changed, writing a modified vtk file: ' + self.vtkfile)
#                print('Old narray = ', oldnarray)
#                print('New narray = ', newnarray)
#                self.WriteVTK(self.vtkfile)

        if(self.pd is not None and self.csvfile is None):
            self.csvfile = os.path.abspath(os.path.join(fileloc,self.name + '.csv'))
            self.WritePandas(self.csvfile)

        mydict['vtk file'] = self.vtkfile 
        mydict['csv file'] = self.csvfile

        print(mydict)
        output = open(filename, 'wb')
        pickle.dump(mydict, output) 
        output.close()

    def Read(self,filename):
        print('Reading metadata for CaseData from ' + filename)

        pkl_file = open(filename, 'rb')
        mydict = pickle.load(pkl_file)
        pkl_file.close()

        print(mydict)

        self.name    = mydict.get('name')
        self.vtkfile = mydict.get('vtk file')
        self.csvfile = mydict.get('csv file')

        print('Name of CaseData object: ' + self.name)

        if(self.vtkfile is not None):
            self.ReadVTK(self.vtkfile)

        if(self.csvfile is not None):
            self.ReadPandas(self.csvfile)

        self.vtkfile = os.path.abspath(self.vtkfile)
        self.csvfile = os.path.abspath(self.csvfile)
