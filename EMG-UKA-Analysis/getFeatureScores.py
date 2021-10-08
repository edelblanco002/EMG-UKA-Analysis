from bar import printProgressBar
import datasetManipulation
import math
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import tables
import telegramNotification
import time

def getFeatureScores(batch):
    # This function trains the sklearn SelectKBest algorithm using different score functions
    # and returns the scores calculated by each one

    features = batch[:,1:]
    labels = batch[:,0]

    clfFClass = SelectKBest(f_classif,k='all').fit(features,labels)
    clfMutual = SelectKBest(mutual_info_classif,k='all').fit(features,labels)

    return clfFClass.scores_, clfMutual.scores_

def main(dirpath = 'C:/Users/Eder/Downloads/EMG-UKA-Trial-Corpus',scriptpath = 'C:/Users/Eder/source/repos/EMG-UKA-Trial-Analysis',uttType='audible',analyzedLabels='All'):
        
    phoneDict = datasetManipulation.getPhoneDict(scriptpath)
    
    tableFile = tables.open_file(f"{dirpath}/{uttType}Table.h5",mode='r')
    table = tableFile.root.data
    
    nExamples = np.shape(table)[0]
    nBatches = 1
    
    nChannels = 6
    nFeatures = 5
    stackingWidth = 15

    rowSize = nChannels*nFeatures*(stackingWidth*2 + 1)

    scoresFClass = np.zeros((nBatches,rowSize))
    scoresMutual = np.zeros((nBatches,rowSize))
    
    totalRemovedNan = 0
    totalRemovedLabels = 0

    mainT0 = time.time()
    print(f"Calculating scores using {nBatches} batches:")
    for n in range(nBatches):
        printProgressBar(n, nBatches, prefix = 'Progress:', suffix = f'{n}/{nBatches} ({n*100/nBatches})', length = 50)
    
        #t0 = time.time()

        if nBatches == 1:
            batch = table[:]
        else:
            # The batch is build taking evenly spaced examples
            batch = table[n::nBatches]
        
        #print("\nGet batch time: ",time.time()-t0," s")
    
        batch, removedNaN = datasetManipulation.removeNaN(batch)
        totalRemovedNan += removedNaN

        removedLabels = 0

        if analyzedLabels == 'Simple':
            batch, removedLabels = datasetManipulation.removeTransitionPhonemes(batch,phoneDict)
        elif analyzedLabels == 'Transitions':
            batch, removedLabels = datasetManipulation.removeSimplePhonemes(batch,phoneDict)

        totalRemovedLabels += removedLabels

        # The step of removing outliers is ommited
        #datasetManipulation.visualizeUnivariateStats(batch)    
        #batch = datasetManipulation.removeOutliers(batch,0.1)
        #datasetManipulation.visualizeUnivariateStats(batch)
    
        scoresFClass[n,:],scoresMutual[n,:] = getFeatureScores(batch)
    
    np.save(f"{dirpath}/scoresFClass{uttType}{analyzedLabels}.npy",scoresFClass)
    np.save(f"{dirpath}/scoresMutual{uttType}{analyzedLabels}.npy",scoresMutual)
    
    executionTime=time.time()-mainT0

    executionTimeSec = executionTime%60
    executionTimeMin = math.floor((executionTime/60)%60)
    executionTimeHours = math.floor(executionTime/3600)
    
    tableFile.close()

    telegramNotification.sendTelegram(f"Execution finished.\n\nOriginal number of examples: {nExamples}\nRemoved NaN: {totalRemovedNan}\nRemoved transition labels and silences: {totalRemovedLabels}\nUtterance type: {uttType}\nAnalyzed labels: {analyzedLabels}\nExecution time: {executionTimeHours} h {executionTimeMin} m {executionTimeSec} s")

    return

if __name__ == '__main__':
    main()