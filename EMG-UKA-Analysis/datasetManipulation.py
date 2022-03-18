from globalVars import SCRIPT_PATH, REMOVE_SILENCES
import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
import seaborn as sns
import time

def getPhoneDict():
    # This function loads a dictionary that links each label number with its corresponding phoneme.

    phoneDict = {}
    
    # Reads the map that contains the relation between each phoneme and its number
    file = open(f"{SCRIPT_PATH}/phoneMap",'r')
    lines = file.readlines()
    file.close()
    
    # Builds the dictionary using the number as key and the phoneme as value
    for line in lines:
        if line.strip():
            line.replace('\n','')
            columns = line.split(' ')
            phone = columns[0]                     
            key = int(columns[1])
            phoneDict[key] = phone

    return phoneDict

def mergeDataset(uttFeatures,uttLabels):
    # This function takes the features from all the utterances and puts them together into a single ndarray

    features = uttFeatures[0]
    labels = uttLabels[0]

    if len(uttLabels) > 1: # If there is one single utterance (rare case), there are not more utterances to stack
        for i in range(1, len(uttLabels)):
            features = np.concatenate((features, uttFeatures[i]), axis=0)
            labels = np.concatenate((labels, uttLabels[i]), axis=0)

    return features, labels

def removeNaN(batch):
    # This function removes the examples that contains any NaN value in its features

    size0 = np.shape(batch)[0]
    
    nz = np.isnan(batch).sum(1)
    batch = batch[nz == 0,:]

    size1 = np.shape(batch)[0]

    removedExamples = size0 - size1

    return batch, removedExamples

"""def removeNaNHilbert(batch):
    # This function removes the examples that contains any NaN value in its features

    size0 = np.shape(batch)[0]
    
    print(np.shape(np.isnan(batch)))

    nz = np.isnan(batch).sum(2).sum(1)
    batch = batch[nz == 0,:,:]

    size1 = np.shape(batch)[0]

    removedExamples = size0 - size1

    return batch, removedExamples"""

def removeOutliers(batch,nu=0.2):
    # This function uses the OneClassSVM algorithm to detect outliers and remove them

    t0 = time.time()

    features = batch[:,1:]
    labels = batch[:,0]

    outDet = OneClassSVM(nu=nu)
    detectedOutliers = outDet.fit_predict(features)
    
    size0 = np.shape(features)[0]
    
    batch = batch[detectedOutliers == 1,:]
    
    size1 = np.shape(batch)[0]
    removedExamples = size0 - size1
    #print("Outliers removed")
    #print(f"{removedExamples} of {size0} examples removed ({round(removedExamples*100/size0,2)}%)")

    #print("\nremoveOutliers execution time: ",time.time()-t0," s")
    return batch

def removeSimplePhonemes(batch,phoneDict,reduceLabels=True):
    # This function removes the examples labeled with simple phonemes or silences

    t0 = time.time()
    size0 = np.shape(batch)[0]

    # The silence mark 'sp' is located at the end of simple marks
    # The transition marks are located after the silence marks 
    for key in sorted(phoneDict.keys()):
        if ('-' in phoneDict[key]) or ('-' in phoneDict[key]):
            firstKey = key
            break

    nz = batch[:,0] >= firstKey
    batch = batch[nz == True, :]

    # Remove transitions between silences
    if REMOVE_SILENCES:
        labelsToRemove = ['sp+sil','sp-sil','sil+sp','sil-sp']

        for key in sorted(phoneDict.keys()):
            if phoneDict[key] in labelsToRemove:
                nz = batch[:,0] == key
                batch = batch[nz == False, :]

    # If reduceLabels, the labels with '-' are changed for the label of the same phoneme transition but with sign '+' (that shold be the previous label in the map, so it's the previous number)
    # That way, transitions are no considered to have two parts (A+B and A-B), but only one label for the same transition (A+B) 
    if reduceLabels:
        evenCorrection = (firstKey + 2) % 2 # Is the first label with '-' odd (1) or even (0)? (phoneDict[firstKey + 1] shold have '+' sign, so phoneDict[firstKey + 2] has '-')
        for i in range(np.shape(batch)[0]):
            if (batch[i,0] + evenCorrection) % 2 == 0: # Those labels that are odd or even (whatever the first label with '-' is) are changed
                batch[i,0] -= 1                        # for the previous label (the same label with '+' sign)


    size1 = np.shape(batch)[0]
    removedExamples = size0 - size1
    #print(f"Transition phones and silences removed")
    #print(f"{removedExamples} of {size0} examples removed ({round(removedExamples*100/size0,2)}%)")
    #print("\nremoveTransitionPhonemes execution time: ",time.time()-t0," s")
    return batch, removedExamples

def removeTransitionPhonemes(batch,phoneDict):
    # This function removes the examples labeled as transition phonemes or silences

    size0 = np.shape(batch)[0]

    # In the phoneme dictionary, the label 'no_label' and the simple phonemes are at the top of the list
    # The transition phonemes contains a '+' or '-' mark, so the first transition phoneme is detected by looking if it contains any of those symbols
    # The last considered phoneme is the one before the first transition phoneme 
    for key in sorted(phoneDict.keys()):
        if ('+' in phoneDict[key]) or ('-' in phoneDict[key]):
            lastKey = key
            break

    nz = batch[:,0] < lastKey
    batch = batch[nz == True, :]

    # Remove silence phonemes (sp and sil)
    if REMOVE_SILENCES:
        labelsToRemove = ['sil','sp']

        for key in sorted(phoneDict.keys()):
            if phoneDict[key] in labelsToRemove:
                nz = batch[:,0] == key
                batch = batch[nz == False, :]
    # If the silences are not removed, map 'sp' silences to 'sil'
    else:
        spLabel = -2 # -2 label does not exist, it's just a value to initialize the variable
        silLabel = -2

        for key in sorted(phoneDict.keys()):
            # Which is the number key for sp label?
            if phoneDict[key] == 'sp':
                spLabel = key
            # And which is the number key for sil label?
            elif phoneDict[key] == 'sil':
                silLabel = key
            
        # Assign to 'sp' labels the label of 'sil'
        batch[batch[:,0] == spLabel, 0] = silLabel

    size1 = np.shape(batch)[0]
    removedExamples = size0 - size1
    #print(f"Transition phones and silences removed")
    #print(f"{removedExamples} of {size0} examples removed ({round(removedExamples*100/size0,2)}%)")
    #print("\nremoveTransitionPhonemes execution time: ",time.time()-t0," s")
    return batch, removedExamples

"""def removeTransitionPhonemesHilbert(batch,phoneDict):
    print("Transition shape in: ",np.shape(batch))
    # This function removes the examples labeled as transition phonemes or silences

    size0 = np.shape(batch)[0]

    # In the phoneme dictionary, the label 'no_label' and the simple phonemes are at the top of the list
    # The transition phonemes contains a '+' or '-' mark, so the first transition phoneme is detected by looking if it contains any of those symbols
    # The last considered phoneme is the one before the first transition phoneme 
    for key in sorted(phoneDict.keys()):
        if ('+' in phoneDict[key]) or ('-' in phoneDict[key]):
            lastKey = key
            break

    nz = batch[:,0,0] < lastKey
    batch = batch[nz == True, :]

    # Remove silence phonemes (sp and sil)
    if REMOVE_SILENCES:
        labelsToRemove = ['sil','sp']

        for key in sorted(phoneDict.keys()):
            if phoneDict[key] in labelsToRemove:
                nz = batch[:,0,0] == key
                batch = batch[nz == False, :]
    # If the silences are not removed, map 'sp' silences to 'sil'
    else:
        spLabel = -2 # -2 label does not exist, it's just a value to initialize the variable
        silLabel = -2

        for key in sorted(phoneDict.keys()):
            # Which is the number key for sp label?
            if phoneDict[key] == 'sp':
                spLabel = key
            # And which is the number key for sil label?
            elif phoneDict[key] == 'sil':
                silLabel = key
            
        # Assign to 'sp' labels the label of 'sil'
        batch[batch[:,0] == spLabel, 0] = silLabel

    size1 = np.shape(batch)[0]
    removedExamples = size0 - size1
    #print(f"Transition phones and silences removed")
    #print(f"{removedExamples} of {size0} examples removed ({round(removedExamples*100/size0,2)}%)")
    #print("\nremoveTransitionPhonemes execution time: ",time.time()-t0," s")

    print("Transition shape out: ",np.shape(batch))
    return batch, removedExamples"""

def removeContextPhonesPilotStudy(batch,phoneDict):
    # This function its made to work with the PilotStudy
    # The phonemes at the beggining and the end of the utterances are supposed to be marked with numbers
    # The purpose of this function is to remove those phonemes

    size0 = np.shape(batch)[0]

    for key in sorted(phoneDict.keys()):
        if any(chr.isdigit() for chr in phoneDict[key]):   # If key has a number in its name
            nz = batch[:,0] == key              # Remove it from the batch
            batch = batch[nz == False, :]

    size1 = np.shape(batch)[0]
    removedExamples = size0 - size1

    return batch, removedExamples

def visualizeUnivariateStats(batch):
    # This function draws a plot with a subplot for each kind of feature, to show its univariate distribution

    stackingWidth = 15
    nFrames = stackingWidth*2 + 1
    nChannels = 6
    nFeatures = 5
    
    # The features of all frames all stacked into one single column for each kind of feature
    dataframe = batch[:,1:].reshape((nChannels*np.shape(batch)[0]*nFrames,nFeatures))
    
    channelCol = []
    
    for i in range(np.shape(batch)[0]):
        for j in range(nChannels):
            for k in range(nFrames):
                channelCol.append(j+1)
    
    dataframe = np.hstack((np.transpose([channelCol]),dataframe))
    
    colNames = ['Channel','w','Pw','Pr','zr','r']
    
    df = pd.DataFrame(dataframe, columns = colNames)
    
    fig = plt.figure()
    fig.suptitle('Features by channel')
    
    ax1= fig.add_subplot(2,3,1)
    ax2= fig.add_subplot(2,3,2)
    ax3= fig.add_subplot(2,3,3)
    
    ax4= fig.add_subplot(2,2,3)
    ax5= fig.add_subplot(2,2,4)
    
    sns.boxplot(ax=ax1, data=df, x='Channel', y='w')
    sns.boxplot(ax=ax2, data=df, x='Channel', y='Pw')
    sns.boxplot(ax=ax3, data=df, x='Channel', y='Pr')
    sns.boxplot(ax=ax4, data=df, x='Channel', y='zr')
    sns.boxplot(ax=ax5, data=df, x='Channel', y='r')
    
    #sns.kdeplot(ax=ax1, data=df, hue='Channel',  fill=True,x='w')
    #sns.kdeplot(ax=ax2, data=df, hue='Channel',  fill=True,x='Pw')
    #sns.kdeplot(ax=ax3, data=df, hue='Channel',  fill=True,x='Pr')
    #sns.kdeplot(ax=ax4, data=df, hue='Channel',  fill=True,x='zr')
    #sns.kdeplot(ax=ax5, data=df, hue='Channel',  fill=True,x='r')

    plt.show()
    return
