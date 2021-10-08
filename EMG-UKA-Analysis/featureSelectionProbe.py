from classifiers import *
import datasetManipulation
from dimensionalityReduction import *
import gatherDataIntoTable
import numpy as np
import os
import pandas as pd
import pickle
import pdb
import shutil
import tables
import telegramNotification
import time
import uuid

class Probe:
    def __init__(self,reductionMethod='',n_features=0,classificationMethod='',scoreFunction='',n_estimators=0,min_samples_leaf=0):
        self.name = uuid.uuid4().hex[:8]

        # Validation rules
        allowedReductionMethods = ['SelectKBest','LDAReduction']
        allowedClassificationMethods = ['GMMmodels','bagging']
        allowedScoreFunctions = ['f_classif','mutual_info_classif']

        # Validation of the reduction method
        if not reductionMethod in allowedReductionMethods:
            print("The allowed values for 'reduction' are the following ones:")
            for method in allowedReductionMethods:
                print(f"- {method}")
            raise ValueError
        else:
            self.reductionMethod = reductionMethod

        # Validation of n_features
        if n_features == 0:
            print("Please, give a value to 'n_features'")
            raise ValueError
        else:
            self.n_features = n_features

        # Validation of the classification method
        if not classificationMethod in allowedClassificationMethods:
            print("The allowed values for 'classificationMethod' are the following ones:")
            for method in allowedClassificationMethods:
                print(f"- {method}")
            raise ValueError
        else:
            self.classificationMethod = classificationMethod

        # If selected SelectKBest, validate the scoreFunction
        if reductionMethod == 'SelectKBest':
            if not scoreFunction in allowedScoreFunctions:
                print("The allowed values for 'scoreFunction' are the following ones:")
                for function in allowedScoreFunctions:
                    print(f"- {function}")
                raise ValueError
            else:
                self.scoreFunction = scoreFunction
        else:
            self.scoreFunction = ''

        if classificationMethod == 'bagging':
            if n_estimators == 0:
                print("Please, give a value to 'n_estimators'")
                raise ValueError
            if min_samples_leaf == 0:
                print("Please, give a value to 'min_samples_leaf'")
                raise ValueError

            self.n_estimators = n_estimators
            self.min_samples_leaf = min_samples_leaf

def probes2csv(probes,dirpath,probeName):
    df = pd.DataFrame(columns=['Name', 'Classification method', 'Reduction method', 'Score function', 'n_estimators', 'min_samples_leaf', 'n_features','Elapsed time in training','Elapsed time in testing with train subset','Elapsed time in testing with test subset','Accuracy with train subset','Accuracy with test subset'])

    for probe in probes:
        row = {}
        row['Name'] = probe.name
        row['Classification method'] = probe.classificationMethod
        if probe.classificationMethod == 'bagging':
            row['n_estimators'] = probe.n_estimators
            row['min_samples_leaf'] = probe.min_samples_leaf
        row['Reduction method'] = probe.reductionMethod
        row['n_features'] = probe.n_features
        if probe.reductionMethod == 'SelectKBest':
            row['Score function'] = probe.scoreFunction
        df = df.append(pd.Series(row),ignore_index=True)

    df.to_csv(f"{dirpath}/results/{probeName}/probeList.csv",index=False)

def saveExecutionResults(name,trainingTime,testingTrainTime,testingTestTime,trainScore,testScore,dirpath,probeName):
    df = pd.read_csv(f"{dirpath}/results/{probeName}/probeList.csv")

    # Time is stored in seconds, so it can be easily manipulated in the future
    df.loc[df['Name'] == name,['Elapsed time in training']] = trainingTime
    df.loc[df['Name'] == name,['Elapsed time in testing with train subset']] = testingTrainTime
    df.loc[df['Name'] == name,['Elapsed time in testing with test subset']] = testingTestTime

    # Accuracy scores are stored as float, representing the percentage
    df.loc[df['Name'] == name,['Accuracy with train subset']] = trainScore
    df.loc[df['Name'] == name,['Accuracy with test subset']] = testScore

    df.to_csv(f"{dirpath}/results/{probeName}/probeList.csv",index=0)

def trainAndTest(dirpath, probeName, trainFeatures, trainLabels, testFeatures, testLabels, uniqueLabels, probe):    
    # This function trains the GMM models and tests them with the train and the test features

    t0 = time.time()

    if probe.classificationMethod == 'GMMmodels':
        models = trainGMMmodels(trainFeatures, trainLabels, uniqueLabels)        
    elif probe.classificationMethod == 'bagging':
        clf = trainBaggingClassifier(trainFeatures, trainLabels, n_estimators=probe.n_estimators, min_samples_leaf=probe.min_samples_leaf)

    t1 = time.time()

    if probe.classificationMethod == 'GMMmodels':
        trainScore, trainConfusionMatrix = testGMMmodels(models, trainFeatures, trainLabels,uniqueLabels)
    else:
        trainScore, trainConfusionMatrix = testClassifier(clf, trainFeatures, trainLabels, uniqueLabels)

    t2 = time.time()

    if probe.classificationMethod == 'GMMmodels':
        testScore, testConfusionMatrix = testGMMmodels(models, testFeatures, testLabels, uniqueLabels)
    else:
        testScore, testConfusionMatrix = testClassifier(clf, testFeatures, testLabels, uniqueLabels)

    t3 = time.time()

    trainingTime = t1-t0
    testingTrainTime = t2-t1
    testingTestTime = t3-t2

    message = f"""{probe.name} finished.
    
    Accuracy:
    Train: {round(trainScore*100,3)}%
    Test: {round(testScore*100,3)}%

    Elapsed time:
    Training: {time.strftime("%H:%M:%S", time.gmtime(trainingTime))}
    Testing train dataset: {time.strftime("%H:%M:%S", time.gmtime(testingTrainTime))}
    Testing test dataset: {time.strftime("%H:%M:%S", time.gmtime(testingTestTime))}
    Total elapsed time: {time.strftime("%H:%M:%S", time.gmtime(t3-t0))}"""

    telegramNotification.sendTelegram(message) # Report the results on Telegram
    
    # Build table to put the results of the execution into a LaTeX document as a tabular
    message = """\\begin{center}
        \\begin{tabular}{r l}
            \\hline
            \\multicolumn{2}{l}{\\textbf{Accuracy:}} \\\\
            Using train dataset to test: & """
            
    message += f"{round(trainScore*100,3)}"
    
    message += """\\% \\\\
            Using test dataset to test: & """
            
    message += f"{round(testScore*100,3)}"
    
    message += """\\% \\\\
            \\hline
            \\multicolumn{2}{l}{\\textbf{Elapsed time:}} \\\\
            Training: & """
            
    message += f"""{time.strftime("%H hour %M min %S sec", time.gmtime(trainingTime))} \\\\
            Testing with train dataset: & {time.strftime("%H hour %M min %S sec", time.gmtime(testingTrainTime))} \\\\
            Testing with test dataset: & {time.strftime("%H hour %M min %S sec", time.gmtime(testingTestTime))} \\\\
            Total elapsed time: & {time.strftime("%H hour %M min %S sec", time.gmtime(t3-t0))} \\\\
            \\hline
            """

    message += """\\end{tabular}
        \\end{center}
    """

    with open(f"{dirpath}/results/{probeName}/{probe.name}_Execution.txt","w+") as file:
        file.write(message)

    saveExecutionResults(probe.name,trainingTime,testingTrainTime,testingTestTime,trainScore,testScore,dirpath,probeName)
    np.save(f"{dirpath}/results/{probeName}/{probe.name}_TrainConfusionMatrix", trainConfusionMatrix)
    np.save(f"{dirpath}/results/{probeName}/{probe.name}_TestConfusionMatrix", testConfusionMatrix)

def getUniqueLabels(list1):
# This function returns a list of the labels that exist in the dataset

    # intilize a null list
    unique_list = []
     
    # iterate over all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return sorted(unique_list)

def main(dirpath = 'C:/Users/Eder/Downloads/EMG-UKA-Trial-Corpus',scriptpath = 'C:/Users/Eder/source/repos/EMG-UKA-Trial-Analysis',uttType='audible',analyzedLabels='All',probeName='default',probes=[]):

    # To continue with execution, previous data must be removed
    if os.path.isdir(f"{dirpath}/results/{probeName}"):
        telegramNotification.sendTelegram("Pay attention to the execution!")
        print(f"{dirpath}/results/{probeName}/ already exists.\n")
        val = input("Do you want to remove previous results? (Y/N): ").upper()
        while (val != 'N') and (val != 'Y'):
            print("The introduced value is not valid.\n")
            val = input("Do you want to remove previous results? (Y/N): ").upper()

        if val == 'Y':
            shutil.rmtree(f"{dirpath}/results/{probeName}")
        else:
            print("The script will not run until that folder is deleted. Otherwise, try changing the experiment name.")
            return

    try:
        os.makedirs(f"{dirpath}/results/{probeName}")
    except:
        pass

    probes2csv(probes,dirpath,probeName)

    # Create a dictionary number -> phoneme and save it for further checking
    phoneDict = datasetManipulation.getPhoneDict(scriptpath)

    with open(f"{dirpath}/results/{probeName}/phoneDict.pkl","wb+") as file:
        pickle.dump(phoneDict,file)

    # Load the training and the testing datasets
    trainTableFile = tables.open_file(f"{dirpath}/{uttType}_train_Table.h5",mode='r')
    trainTable = trainTableFile.root.data

    testTableFile = tables.open_file(f"{dirpath}/{uttType}_test_Table.h5",mode='r')
    testTable = testTableFile.root.data

    nExamples = np.shape(trainTable)[0]

    trainBatch = trainTable[:]
    testBatch = testTable[:]

    trainTableFile.close()
    testTableFile.close()

    pdb.set_trace()

    # Define the parameters in order to calculate the row size and the name of the features
    nChannels = 6
    nFeatures = 5
    stackingWidth = 15
    featureNames = ['w','Pw','Pr','z','r']

    rowSize = nChannels*nFeatures*(stackingWidth*2 + 1)

    totalRemovedLabels = 0

    removedLabels = 0

    # Remove examples with any NaN in their features
    trainBatch = datasetManipulation.removeNaN(trainBatch)[0]
    testBatch = datasetManipulation.removeNaN(testBatch)[0]

    # If selected, take only the simple or the transition labels and discard the rest of the examples
    if analyzedLabels == 'Simple':
            trainBatch, removedLabels = datasetManipulation.removeTransitionPhonemes(trainBatch,phoneDict)
            testBatch = datasetManipulation.removeTransitionPhonemes(testBatch,phoneDict)[0]
    elif analyzedLabels == 'Transitions':
            trainBatch, removedLabels = datasetManipulation.removeSimplePhonemes(trainBatch,phoneDict)
            testBatch = datasetManipulation.removeSimplePhonemes(testBatch,phoneDict)[0]
    elif analizedLabels == 'PilotStudy':
            trainBatch, removedLabels = datasetManipulation.removeUnwantedPhonesPilotStudy(trainBatch,phoneDict)
            testBatch = datasetManipulation.removeUnwantedPhonesPilotStudy(testBatch,phoneDict)[0]

    totalRemovedLabels += removedLabels

    # Separate labels (first column) from the features (rest of the columns)
    trainFeatures = trainBatch[:,1:]
    trainLabels = trainBatch[:,0]

    testFeatures = testBatch[:,1:]
    testLabels = testBatch[:,0]

    # uniqueLabels is a list of the different labels existing in the dataset
    uniqueLabels = getUniqueLabels(trainLabels)

    np.save(f"{dirpath}/results/{probeName}/uniqueLabels",uniqueLabels)

    # Create the description of the experiment
    message = f"{probeName}:\nTrying:\n\n"

    for probe in probes:
        message += f"Name: {probe.name}\n"

        if probe.reductionMethod == 'SelectKBest':
            message += f"- SelectKBest with k={probe.n_features} and {probe.scoreFunction} function.\n"
        elif probe.reductionMethod == 'LDAReduction':
            message += f"- LDA reduction with {probe.n_features} components.\n"
        else:
            print("Invalid reduction method")
            return

        message += f"- Classificating with '{probe.classificationMethod}'.\n"

        if probe.classificationMethod == 'bagging':
            message += f"   - n_estimators: {probe.n_estimators}\n"
            message += f"   - min_samples_leaf: {probe.min_samples_leaf}\n"

        message += "\n"

    # Save the description to a text file
    with open(f"{dirpath}/results/{probeName}/description.txt","w+") as file:
        file.write(message)

    # Send the description as a Telegram message
    telegramNotification.sendTelegram(message)

    # Perform all the probes: Feature reduction, train and test
    for probe in probes:
        if probe.reductionMethod == "SelectKBest":
            reductedTrainFeatures, reductedTestFeatures = featureSelection(probe.n_features, probe.scoreFunction, trainFeatures, testFeatures, trainLabels, featureNames, nChannels, stackingWidth, dirpath,probeName)
        elif probe.reductionMethod == "LDAReduction":
            reductedTrainFeatures, reductedTestFeatures = featureLDAReduction(probe.n_features, testFeatures, trainFeatures, trainLabels)

        trainAndTest(dirpath, probeName, reductedTrainFeatures, trainLabels, reductedTestFeatures, testLabels, uniqueLabels, probe)
