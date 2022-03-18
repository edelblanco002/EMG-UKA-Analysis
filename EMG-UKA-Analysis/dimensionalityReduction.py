from matplotlib.pyplot import axis
import drawBarPlot
from globalVars import DIR_PATH, FEATURE_NAMES, N_CHANNELS, STACKING_WIDTH
import featureSelectionProbe
from datasetManipulation import mergeDataset
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import time



def featureSelection(nFeatures, method, trainFeatures, testFeatures, trainLabels, probeName):
    # This function performs the feature selection and also saves a table with the selected features sorted by the score given to them.

    t0 = time.time()

    if method == 'f_classif':
        scoreFunction = f_classif
    else:
        scoreFunction = mutual_info_classif

    # Gather all train features and labels from all utterances to fit the selector
    allTrainFeatures, allTrainLabels = mergeDataset(trainFeatures,trainLabels)

    selector = SelectKBest(scoreFunction,k = nFeatures)
    selector = selector.fit(allTrainFeatures,allTrainLabels) # Select the features and reduce the features on the train dataset in consonance
    # Reduce the features of the train and the test set in consonance
    newTrainFeatures = []
    newTestFeatures = []
    for i in range(len(trainFeatures)):
        newTrainFeatures.append(selector.transform(trainFeatures[i]))
    for i in range(len(testFeatures)):
        newTestFeatures.append(selector.transform(testFeatures[i]))

    colNames = drawBarPlot.getColNames() # Obtain the name of all the features
    ranking = drawBarPlot.printRanking(selector.scores_,colNames,nFeatures,method) # Obtain the ranking of the selected features as a string

    t1 = time.time()

    # Write the ranking into a text file
    with open(f"{DIR_PATH}/results/{probeName}/{method}{nFeatures}Ranking.txt","w+") as file:
        file.write(ranking)

    print("Feature selection results:")
    print("Shape of train features before feature selection: ",np.shape(trainFeatures[0]))
    print("Shape of train features after feature selection: ",np.shape(newTrainFeatures[0])) # This is displayed to make sure that the dimensions of the features have changed
    print("Execution time",time.strftime('%H:%M:%S', time.gmtime(t1-t0)))
    print("\n")

    return newTrainFeatures, newTestFeatures, selector

def featureLDAReduction(nComponents, trainFeatures, testFeatures, trainLabels):
    # This function performons the LDA reduction of the features

    t0 = time.time()
    
    # Gather all train features and labels from all utterances to fit the selector
    allTrainFeatures, allTrainLabels = mergeDataset(trainFeatures,trainLabels)

    LDASelector = LinearDiscriminantAnalysis(n_components=nComponents)

    LDASelector = LDASelector.fit(allTrainFeatures,allTrainLabels) # Train the selector with all the training features
    # Reduce the dimensionality of the train and test features in consonance
    newTrainFeatures = []
    newTestFeatures = []
    for i in range(len(trainFeatures)):
        newTrainFeatures.append(LDASelector.transform(trainFeatures[i]))

    for i in range(len(testFeatures)):
        newTestFeatures.append(LDASelector.transform(testFeatures[i]))

    t1 = time.time()
    
    print("LDA results:")
    print("Shape of train features before LDA transform: ",np.shape(trainFeatures[0]))
    print("Shape of train features after LDA transform: ",np.shape(newTrainFeatures[0])) # This is displayed to make sure that the dimensions of the features have changed
    print("Execution time",time.strftime('%H:%M:%S', time.gmtime(t1-t0)))
    print("\n")

    return newTrainFeatures, newTestFeatures, LDASelector
