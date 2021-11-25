from classifiers import *
import datasetManipulation
from dimensionalityReduction import *
import gatherDataIntoTable
from globalVars import DIR_PATH, N_CHANNELS, N_FEATURES, REMOVE_CONTEXT_PHONEMES, STACKING_WIDTH
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
    def __init__(self,reductionMethod='',n_features=0,classificationMethod='',scoreFunction='',n_estimators=0,min_samples_leaf=0,speaker='all',session='all',trainSpeaker='',trainSession='',testSpeaker='',testSession='',uttType='audible',analyzedLabels='simple', useChannels=[]):
        self.name = uuid.uuid4().hex[:8]

        # Validation rules
        allowedReductionMethods = ['SelectKBest','LDAReduction','NoReduction']
        allowedClassificationMethods = ['GMMmodels','bagging']
        allowedScoreFunctions = ['f_classif','mutual_info_classif']
        allowedUtteranceTypes = ['audible','whispered','silent']
        allowedAnalyzedLabels = ['simple','transitions','all']

        self.validateSpeakerAndSession(speaker, session, trainSpeaker, trainSession, testSpeaker, testSession)

        self.validateUttType(uttType, allowedUtteranceTypes)

        self.validateAnalyzedLabels(analyzedLabels, allowedAnalyzedLabels)

        self.validateReductionMethod(reductionMethod, n_features, scoreFunction, allowedReductionMethods, allowedScoreFunctions)

        self.validateClassificationMethod(classificationMethod, n_estimators, min_samples_leaf, allowedClassificationMethods)

        self.validateUseChannels(useChannels)

    def validateReductionMethod(self, reductionMethod, n_features, scoreFunction, allowedReductionMethods, allowedScoreFunctions):
        # Validation of the reduction method
        if not reductionMethod in allowedReductionMethods:
            print("The allowed values for 'reductionMethod' are the following ones:")
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

    def validateAnalyzedLabels(self, analyzedLabels, allowedAnalyzedLabels):
        # Validation of analyzed labels
        if not analyzedLabels in allowedAnalyzedLabels:
            print("The allowed values for 'analyzedLabels' are the following ones:")
            for lab in allowedAnalyzedLabels:
                print(f"- {lab}")
        else:
            self.analyzedLabels = analyzedLabels

    def validateUttType(self, uttType, allowedUtteranceTypes):
        # Validation of the utterance type
        if not uttType in allowedUtteranceTypes:
            print("The allowed values for 'uttType' are the following ones:")
            for utt in allowedUtteranceTypes:
                print(f"- {utt}")
            raise ValueError
        else:
            self.uttType = uttType

    def validateSpeakerAndSession(self, speaker, session, trainSpeaker, trainSession, testSpeaker, testSession):
        # Set speaker and session
        if trainSpeaker == '' and testSpeaker == '' and trainSession == '' and testSession == '': # If not different session or speaker has been specified for training and testing, use the speaker and session parameters for both them
            self.trainSpeaker = speaker
            self.testSpeaker = speaker
            if speaker == 'all':
                self.trainSession = 'all' # The session only can be specified for one concrete speaker
                self.testSession = 'all'
            else:
                self.trainSession = session
                self.testSession = session
        else: # If any of the specific parameters for testing or training have been initialized
            if trainSpeaker == '' or testSpeaker == '' or trainSession == '' or testSession == '': # Check if something is left unespecified
                print("To set different users or sessions to training and testing, all the parameters must be given:")
                print("\t- trainSpeaker\n\t- trainSession\n\t- testSpeaker\n\t- testSession")
                raise ValueError
            else: # The 4 parameters have been given, so use the given speakers
                self.trainSpeaker = trainSpeaker
                self.testSpeaker = testSpeaker
                if trainSpeaker == 'all':
                    self.trainSession = 'all' # The session only can be specified for one concrete speaker
                else:
                    self.trainSession = trainSession
                if testSpeaker == 'all':
                    self.testSession = 'all' # The session only can be specified for one concrete speaker
                else:
                    self.testSession = testSession

    def validateClassificationMethod(self, classificationMethod, n_estimators, min_samples_leaf, allowedClassificationMethods):
        # Validation of the classification method
        if not classificationMethod in allowedClassificationMethods:
            print("The allowed values for 'classificationMethod' are the following ones:")
            for method in allowedClassificationMethods:
                print(f"- {method}")
            raise ValueError
        else:
            self.classificationMethod = classificationMethod

        if classificationMethod == 'bagging':
            if n_estimators == 0:
                print("Please, give a value to 'n_estimators'")
                raise ValueError
            if min_samples_leaf == 0:
                print("Please, give a value to 'min_samples_leaf'")
                raise ValueError
                

            self.n_estimators = n_estimators
            self.min_samples_leaf = min_samples_leaf

        else:
            self.n_estimators = ''
            self.min_samples_leaf = ''

    def validateUseChannels(self, useChannels):
        useChannels_set = set(useChannels)
        if len(useChannels) != len(useChannels_set): # Check if there are repeated channels
            print("Some channel has been repeated. You only can select each channel once.")
            raise ValueError
        if True in [i >= N_CHANNELS for i in useChannels]: # Check if the selected channels are into the available range
            print(f"The number of available channels is {N_CHANNELS}. The selected channels must be between 0 and {N_CHANNELS - 1}.")
            raise ValueError
        self.useChannels = useChannels

def checkDifferences(lastProbe, currentProbe):
    trainSpeaker = (lastProbe.trainSpeaker != currentProbe.trainSpeaker)
    trainSession = (lastProbe.trainSession != currentProbe.trainSession)
    testSpeaker = (lastProbe.testSpeaker != currentProbe.testSpeaker)
    testSession = (lastProbe.testSession != currentProbe.testSession)
    analyzedLabels = (lastProbe.analyzedLabels != currentProbe.analyzedLabels)
    uttType = (lastProbe.uttType != currentProbe.uttType)
    nFeatures = (lastProbe.n_features != currentProbe.n_features)
    reductionMethod = (lastProbe.reductionMethod != currentProbe.reductionMethod)
    scoreFunction = (lastProbe.scoreFunction != currentProbe.scoreFunction)
    classificationMethod = (lastProbe.classificationMethod != currentProbe.classificationMethod)
    nEstimators = (lastProbe.n_estimators != currentProbe.n_estimators)
    minSamplesLeaf = (lastProbe.min_samples_leaf != currentProbe.min_samples_leaf)

    trainingBatchHasChanged = trainSpeaker or trainSession or analyzedLabels or uttType
    testingBatchHasChanged = testSpeaker or testSession or analyzedLabels or uttType

    trainReductionHasChanged = trainingBatchHasChanged or nFeatures or reductionMethod or scoreFunction
    testReductionHasChanged = testingBatchHasChanged or trainReductionHasChanged

    classifierHasChanged = trainingBatchHasChanged or trainReductionHasChanged or classificationMethod or nEstimators or minSamplesLeaf

    return trainingBatchHasChanged, testingBatchHasChanged, trainReductionHasChanged, testReductionHasChanged, classifierHasChanged

def probes2csv(probes,probeName):
    df = pd.DataFrame(columns=['Name', 'Train speaker', 'Train session','Test speaker', 'Test session', 'Utterance type', 'Analyzed labels', 'Classification method', 'Reduction method', 'Score function', 'n_estimators', 'min_samples_leaf', 'n_features','Elapsed time in training','Elapsed time in testing with train subset','Elapsed time in testing with test subset','Accuracy with train subset','Accuracy with test subset'])

    probe: Probe
    for probe in probes:
        row = {}
        row['Name'] = probe.name
        row['Train speaker'] = probe.trainSpeaker
        row['Train session'] = probe.trainSession
        row['Test speaker'] = probe.testSpeaker
        row['Test session'] = probe.testSession
        row['Utterance type'] = probe.uttType
        row['Analyzed labels'] = probe.analyzedLabels
        row['Classification method'] = probe.classificationMethod
        if probe.classificationMethod == 'bagging':
            row['n_estimators'] = probe.n_estimators
            row['min_samples_leaf'] = probe.min_samples_leaf
        row['Reduction method'] = probe.reductionMethod
        row['n_features'] = probe.n_features
        if probe.reductionMethod == 'SelectKBest':
            row['Score function'] = probe.scoreFunction
        df = df.append(pd.Series(row),ignore_index=True)

    df.to_csv(f"{DIR_PATH}/results/{probeName}/probeList.csv",index=False)

def saveExecutionResults(name,trainingTime,testingTrainTime,testingTestTime,trainScore,testScore,probeName):
    df = pd.read_csv(f"{DIR_PATH}/results/{probeName}/probeList.csv")

    # Time is stored in seconds, so it can be easily manipulated in the future
    df.loc[df['Name'] == name,['Elapsed time in training']] = trainingTime
    df.loc[df['Name'] == name,['Elapsed time in testing with train subset']] = testingTrainTime
    df.loc[df['Name'] == name,['Elapsed time in testing with test subset']] = testingTestTime

    # Accuracy scores are stored as float, representing the percentage
    df.loc[df['Name'] == name,['Accuracy with train subset']] = trainScore
    df.loc[df['Name'] == name,['Accuracy with test subset']] = testScore

    df.to_csv(f"{DIR_PATH}/results/{probeName}/probeList.csv",index=0)

def selectChannels(features: np.ndarray,useChannels: list):
    nFrames = features.shape[0]
    featuresPerChannel = N_FEATURES*(STACKING_WIDTH*2 + 1)
    nOutputFeatures = len(useChannels)*featuresPerChannel

    outputFeatures = np.zeros((nFrames,nOutputFeatures))

    for i, channel in enumerate(useChannels):
        outputFeatures[:,i*featuresPerChannel:(i+1)*featuresPerChannel] = features[:,channel*featuresPerChannel:(channel+1)*featuresPerChannel]

    return outputFeatures

def trainAndTest(probeName, trainFeatures, trainLabels, testFeatures, testLabels, uniqueLabels, probe, classifierHasChanged):    
    # This function trains the GMM models and tests them with the train and the test features

    t0 = time.time()

    if classifierHasChanged:
        if probe.classificationMethod == 'GMMmodels':
            clf = trainGMMmodels(trainFeatures, trainLabels, uniqueLabels)        
        elif probe.classificationMethod == 'bagging':
            clf = trainBaggingClassifier(trainFeatures, trainLabels, n_estimators=probe.n_estimators, min_samples_leaf=probe.min_samples_leaf)

    t1 = time.time()

    if classifierHasChanged:
        if probe.classificationMethod == 'GMMmodels':
            trainScore, trainConfusionMatrix = testGMMmodels(clf, trainFeatures, trainLabels,uniqueLabels)
        else:
            trainScore, trainConfusionMatrix = testClassifier(clf, trainFeatures, trainLabels, uniqueLabels)

        with open(f"{DIR_PATH}/results/{probeName}/lastTrainingResults.pkl",'wb') as file:
            pickle.dump([clf, trainScore, trainConfusionMatrix],file)

    else: # Load the results of last training
        with open(f"{DIR_PATH}/results/{probeName}/lastTrainingResults.pkl",'rb') as file:
            clf, trainScore, trainConfusionMatrix = pickle.load(file)

    t2 = time.time()

    if probe.classificationMethod == 'GMMmodels':
        testScore, testConfusionMatrix = testGMMmodels(clf, testFeatures, testLabels, uniqueLabels)
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

    with open(f"{DIR_PATH}/results/{probeName}/{probe.name}_Execution.txt","w+") as file:
        file.write(message)

    saveExecutionResults(probe.name,trainingTime,testingTrainTime,testingTestTime,trainScore,testScore,probeName)
    np.save(f"{DIR_PATH}/results/{probeName}/{probe.name}_TrainConfusionMatrix", trainConfusionMatrix)
    np.save(f"{DIR_PATH}/results/{probeName}/{probe.name}_TestConfusionMatrix", testConfusionMatrix)

def getUniqueLabels(list1):
# This function returns a list of the labels that exist in the dataset

    # intilize a null list
    unique_list = []

    # If an only set of labels has been passed as argument
    if isinstance(list1,np.ndarray):
        # iterate over all elements
        for x in list1:
            # check if exists in unique_list or not
            if x not in unique_list:
                unique_list.append(x)

    # If many sets of labels have been pased as argument, as a list: [list1,list2]
    elif isinstance(list1,list):
        for labelList in list1:
            # iterate over all elements
            for x in labelList:
                # check if exists in unique_list or not
                if x not in unique_list:
                    unique_list.append(x)

    return sorted(unique_list)

def loadData(speaker,session,uttType,analyzedLabels,useChannels,phoneDict,subset):
    gatherDataIntoTable.main(uttType,subset=subset,speaker=speaker,session=session)
            
    # If the probes are done with a specific speaker and session, build the name of the file where it is saved.
    basename = ""

    if speaker != 'all':
        basename += f"{speaker}_"

        if session != 'all':
            basename += f"{session}_"

    basename += f"{uttType}"

    # Load the training datasets
    tableFile = tables.open_file(f"{DIR_PATH}/{basename}_{subset}_Table.h5",mode='r')
    table = tableFile.root.data

    batch = table[:]

    tableFile.close()

    # Remove examples with any NaN in their features
    batch = datasetManipulation.removeNaN(batch)[0]

    totalRemovedLabels = 0

    removedLabels = 0

    # If selected, take only the simple or the transition labels and discard the rest of the examples
    if analyzedLabels == 'simple':
        batch, removedLabels = datasetManipulation.removeTransitionPhonemes(batch,phoneDict)
    elif analyzedLabels == 'transitions':
        batch, removedLabels = datasetManipulation.removeSimplePhonemes(batch,phoneDict)
                
    totalRemovedLabels += removedLabels

    # If the analyzed corpus is Pilot Study and removeContext is set to True, remove the context phonemes
    if REMOVE_CONTEXT_PHONEMES:
        batch, removedLabels = datasetManipulation.removeContextPhonesPilotStudy(batch,phoneDict)
        totalRemovedLabels += removedLabels

    # Separate labels (first column) from the features (rest of the columns)
    features = batch[:,1:]

    if useChannels != []: # If only some channels have been selected
        features = selectChannels(features,useChannels)

    labels = batch[:,0]
    return features, labels

def main(experimentName='default',probes=[]):

    # To continue with execution, previous data must be removed
    if os.path.isdir(f"{DIR_PATH}/results/{experimentName}"):
        telegramNotification.sendTelegram("Pay attention to the execution!")
        print(f"{DIR_PATH}/results/{experimentName}/ already exists.\n")
        val = input("Do you want to remove previous results? (Y/N): ").upper()
        while (val != 'N') and (val != 'Y'):
            print("The introduced value is not valid.\n")
            val = input("Do you want to remove previous results? (Y/N): ").upper()

        if val == 'Y':
            shutil.rmtree(f"{DIR_PATH}/results/{experimentName}")
        else:
            print("The script will not run until that folder is deleted. Otherwise, try changing the experiment name.")
            return

    try:
        os.makedirs(f"{DIR_PATH}/results/{experimentName}")
    except:
        pass

    probes2csv(probes,experimentName)

    # Create a dictionary number -> phoneme and save it for further checking
    phoneDict = datasetManipulation.getPhoneDict()

    with open(f"{DIR_PATH}/results/{experimentName}/phoneDict.pkl","wb+") as file:
        pickle.dump(phoneDict,file)

    # Create the description of the experiment
    message = f"{experimentName}:\nTrying:\n\n"

    probe: Probe
    for probe in probes:
        message += f"Name: {probe.name}\n"

        message += f"- Train speaker: {probe.trainSpeaker}\n"
        message += f"- Train session: {probe.trainSession}\n"
        message += f"- Test speaker: {probe.testSpeaker}\n"
        message += f"- Test session: {probe.testSession}\n"
        message += f"- Utterance type: {probe.uttType}\n"
        message += f"- Analyzed labels: {probe.analyzedLabels}\n"

        if probe.reductionMethod == 'SelectKBest':
            message += f"- SelectKBest with k={probe.n_features} and {probe.scoreFunction} function.\n"
        elif probe.reductionMethod == 'LDAReduction':
            message += f"- LDA reduction with {probe.n_features} components.\n"
        elif probe.reductionMethod == 'NoReduction':
            message += f"- No dimensionality reduction.\n"
        else:
            print("Invalid reduction method")
            raise ValueError
            return

        message += f"- Classificating with '{probe.classificationMethod}'.\n"

        if probe.classificationMethod == 'bagging':
            message += f"   - n_estimators: {probe.n_estimators}\n"
            message += f"   - min_samples_leaf: {probe.min_samples_leaf}\n"

        message += "\n"

    # Save the description to a text file
    with open(f"{DIR_PATH}/results/{experimentName}/description.txt","w+") as file:
        file.write(message)

    # Send the description as a Telegram message
    telegramNotification.sendTelegram(message)

    # The probes are ordered by speaker, session, utterance type and analyzed labels, so that when there are several test that use the same frames, the batch is no rebuild.
    probes = sorted(probes, key=lambda x: (x.trainSpeaker, x.trainSession, x.uttType, x.analyzedLabels))

    probe: Probe
    for index, probe in enumerate(probes):
        if index == 0: # If it's the first probe, every tasks must be done
            trainingBatchHasChanged, testingBatchHasChanged, trainReductionHasChanged, testReductionHasChanged, classifierHasChanged = (True, )*5
        else: # Check what has changed with respect to the previous test in order to save time avoiding unnecessary tasks
            lastProbe: Probe
            lastProbe = probes[index - 1]
            currentProbe = probe
            trainingBatchHasChanged, testingBatchHasChanged, trainReductionHasChanged, testReductionHasChanged, classifierHasChanged = checkDifferences(lastProbe, currentProbe)

        # If the speaker/session is different from the last probe, rebuild the batch
        if trainingBatchHasChanged:
            trainFeatures, trainLabels = loadData(probe.trainSpeaker,probe.trainSession,probe.uttType,probe.analyzedLabels,probe.useChannels,phoneDict,'train')

        if testingBatchHasChanged:
            testFeatures, testLabels = loadData(probe.testSpeaker,probe.testSession,probe.uttType,probe.analyzedLabels,probe.useChannels,phoneDict,'test')

        # uniqueLabels is a list of the different labels existing in the dataset
        if trainingBatchHasChanged or testingBatchHasChanged:
            uniqueLabels = getUniqueLabels([trainLabels,testLabels])

        np.save(f"{DIR_PATH}/results/{experimentName}/{probe.name}_uniqueLabels",uniqueLabels)

        # Perform: Feature reduction, train and test
        if trainReductionHasChanged: # If training batch or the dim. reduction options have changed: train the selector to reduce dimensionality
            if probe.reductionMethod == "SelectKBest":
                reductedTrainFeatures, reductedTestFeatures, selector = featureSelection(probe.n_features, probe.scoreFunction, trainFeatures, testFeatures, trainLabels, experimentName)
            elif probe.reductionMethod == "LDAReduction":
                uniqueLabelsTrain = getUniqueLabels(trainLabels)
                if probe.n_features > len(uniqueLabelsTrain) - 1:
                    print(f"The maximum number of components allowed in LDA reduction is n_classes - 1.\nThe n_classes in the train subset of this experiment is {len(uniqueLabelsTrain)}.")
                    raise ValueError
                else:
                    reductedTrainFeatures, reductedTestFeatures, selector = featureLDAReduction(probe.n_features, trainFeatures, testFeatures, trainLabels)
            elif probe.reductionMethod == 'NoReduction':
                print("Feature reduction ommited")
                reductedTrainFeatures = trainFeatures[:]
                reductedTestFeatures = testFeatures[:]
        else:   # If training batch remains the same, use the previous selector to reduce dimensionality of the testing subset
            if testReductionHasChanged:
                if probe.reductionMethod != 'NoReduction':
                    reductedTestFeatures = selector.transform(testFeatures)
                else:
                    print("Feature reduction ommited")
                    reductedTestFeatures = testFeatures[:]

        trainAndTest(experimentName, reductedTrainFeatures, trainLabels, reductedTestFeatures, testLabels, uniqueLabels, probe, classifierHasChanged)

    gatherDataIntoTable.removeTables()