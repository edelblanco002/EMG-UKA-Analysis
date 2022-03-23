from bar import printProgressBar
import datasetManipulation
import gatherDataIntoTable
from globalVars import DIR_PATH, SCRIPT_PATH, N_BATCHES, REMOVE_CONTEXT_PHONEMES, MFCC_ROW_SIZE, ROW_SIZE
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

def main(uttType='audible',analyzedLabels='all',speaker='all',session='all',analyzeMFCCs=False):
    
    gatherDataIntoTable.main(uttType=uttType,subset='both',speaker='all',session='all')

    # If the probes are done with a specific speaker and session, build the name of the file where it is saved.
    basename = ""

    if speaker != 'all':
        basename += f"{speaker}_"

        if session != 'all':
            basename += f"{session}_"

    basename += f"{uttType}"

    tableFile = tables.open_file(f"{DIR_PATH}/{basename}Table.h5",mode='r')
    table = tableFile.root.data
    
    nExamples = np.shape(table)[0]
    
    if analyzeMFCCs:
        row_size = ROW_SIZE
    else:
        row_size = MFCC_ROW_SIZE

    scoresFClass = np.zeros((N_BATCHES,row_size - 1))
    scoresMutual = np.zeros((N_BATCHES,row_size - 1))
    
    totalRemovedNan = 0
    totalRemovedLabels = 0

    mainT0 = time.time()
    print(f"Calculating scores using {N_BATCHES} batches:")
    for n in range(N_BATCHES):
        printProgressBar(n, N_BATCHES, prefix = 'Progress:', suffix = f'{n}/{N_BATCHES} ({n*100/N_BATCHES})', length = 50)
    
        #t0 = time.time()

        if N_BATCHES == 1:
            batch = table[:]
        else:
            # The batch is build taking evenly spaced examples
            batch = table[n::N_BATCHES]
        
        #print("\nGet batch time: ",time.time()-t0," s")
    
        batch, removedNaN = datasetManipulation.removeNaN(batch)
        totalRemovedNan += removedNaN

        removedLabels = 0

        if analyzedLabels == 'simple':
            batch, removedLabels = datasetManipulation.removeTransitionPhonemes(batch)
        elif analyzedLabels == 'transitions':
            batch, removedLabels = datasetManipulation.removeSimplePhonemes(batch)

        totalRemovedLabels += removedLabels

        # If the analyzed corpus is Pilot Study and removeContext is set to True, remove the context phonemes
        if REMOVE_CONTEXT_PHONEMES:
            batch, removedLabels = datasetManipulation.removeContextPhonesPilotStudy(batch)

        totalRemovedLabels += removedLabels

        # The step of removing outliers is ommited
        #datasetManipulation.visualizeUnivariateStats(batch)    
        #batch = datasetManipulation.removeOutliers(batch,0.1)
        #datasetManipulation.visualizeUnivariateStats(batch)
    
        scoresFClass[n,:],scoresMutual[n,:] = getFeatureScores(batch)
    
    np.save(f"{DIR_PATH}/scoresFClass{uttType}{analyzedLabels}.npy",scoresFClass)
    np.save(f"{DIR_PATH}/scoresMutual{uttType}{analyzedLabels}.npy",scoresMutual)
    
    executionTime=time.time()-mainT0

    executionTimeSec = executionTime%60
    executionTimeMin = math.floor((executionTime/60)%60)
    executionTimeHours = math.floor(executionTime/3600)
    
    tableFile.close()
    gatherDataIntoTable.removeTables()

    telegramNotification.sendTelegram(f"Execution finished.\n\nOriginal number of examples: {nExamples}\nRemoved NaN: {totalRemovedNan}\nRemoved transition labels and silences: {totalRemovedLabels}\nUtterance type: {uttType}\nAnalyzed labels: {analyzedLabels}\nExecution time: {executionTimeHours} h {executionTimeMin} m {executionTimeSec} s")

    return

if __name__ == '__main__':
    main()