import os
import numpy as np
from scipy.io import wavfile
import pandas as pd
import dipy
from dipy import core
from dipy.core import geometry
import math

def getDatasetNames(datasetsFolder):
    allfilesData = os.listdir(datasetsFolder + "\\labels")
    dataDir = []

    for item in allfilesData:
        filename = item.split('.')
        dataDir.append(filename[0][6:])
        
    return dataDir
	
	
	
class AmbiIeeeSample:
    def __init__(self, filesDirectory, fileName):
        # Rewriting full path to files
        self.labelFile = filesDirectory + '\\labels\\label_' + fileName + '.csv'
        self.audioFile_A = filesDirectory + '\\data\\' + fileName + '_A.wav'
        self.audioFile_B = filesDirectory + '\\data\\' + fileName + '_B.wav'
        self.samplingRate, _ = wavfile.read(self.audioFile_A)
        
    def getMetaData(self):
        # Loading label file
        table = pd.read_csv(self.labelFile, delimiter=',')
        #Loading meta-data
        table = self.completeMetadata(table)
        
        return table
    

    def getAudioData(self):
        table = self.getMetaData()
        
        # Loading ambisonics N-by-4 audio, mic A
        _, dataA = wavfile.read(self.audioFile_A)
        # Loading ambisonics N-by-4 audio, mic B
        _, dataB = wavfile.read(self.audioFile_B)
        
        # Adding audio data to table
        table = self.completeWithMics(table, dataA, dataB)

        return table
    
    
    
    def completeMetadata(self, table):
        # Creating vectors
        nOfRows = len(list(table.iterrows()))
        startSamples = np.zeros((nOfRows), dtype=int)
        endSamples = np.zeros((nOfRows), dtype=int)
        azimuts = np.zeros((nOfRows), dtype=float)
        elevations = np.zeros((nOfRows), dtype=float)
        radii = np.zeros((nOfRows), dtype=float)

        # Populating arrays
        for index, row in table.iterrows():
            startSamples[index] = round(row['Start'] * self.samplingRate, 0)
            endSamples[index] = round(row['End'] * self.samplingRate, 0)    

            radii[index], elevations[index], azimuts[index] = dipy.core.geometry.cart2sphere(row['X'], row['Y'], row['Z'])

        # Converting angles from rad to degree
        radToDeg = 360 / 2 / math.pi
        azimuts *= radToDeg
        elevations *= radToDeg

        # Copying columns into table    
        newTable = table
        newTable.insert(len(table.columns), 'R', radii)
        newTable.insert(len(table.columns), 'Elev', elevations)
        newTable.insert(len(table.columns), 'Azimut', azimuts)
        newTable.insert(len(table.columns), 'StartSample', startSamples)
        newTable.insert(len(table.columns), 'EndSample', endSamples)

        return newTable


    def completeWithMics(self, table, dataA, dataB):
        # Creating vectors
        nOfRows = len(list(table.iterrows()))
        audioArraysA = []
        audioArraysB = []

        # Populating arrays
        for index, row in table.iterrows():
            audioArraysA.append(dataA[table['StartSample'][index] : table['EndSample'][index] + 1, :])
            audioArraysB.append(dataB[table['StartSample'][index] : table['EndSample'][index] + 1, :])

        # Copying columns into table    
        table.insert(len(table.columns), 'MicA', audioArraysA)
        table.insert(len(table.columns), 'MicB', audioArraysB)

        return table