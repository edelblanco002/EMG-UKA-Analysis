import drawBarPlot
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import time

def featureSelection(nFeatures, method, trainFeatures, testFeatures, trainLabels, featureNames, nChannels, stackingWidth, dirpath, probeName):
    # This function performs the feature selection and also saves a table with the selected features sorted by the score given to them.

    t0 = time.time()

    if method == 'f_classif':
        scoreFunction = f_classif
    else:
        scoreFunction = mutual_info_classif

    selector = SelectKBest(scoreFunction,k = nFeatures)
    newTrainFeatures = selector.fit_transform(trainFeatures,trainLabels) # Select the features and reduce the features on the train dataset in consonance
    newTestFeatures = selector.transform(testFeatures) # Reduce the features of the test set in consonance

    colNames = drawBarPlot.getColNames(nChannels,stackingWidth,featureNames) # Obtain the name of all the features
    ranking = drawBarPlot.printRanking(selector.scores_,colNames,nFeatures,method) # Obtain the ranking of the selected features as a string

    t1 = time.time()

    # Write the ranking into a text file
    with open(f"{dirpath}/results/{probeName}/{method}{nFeatures}Ranking.txt","w+") as file:
        file.write(ranking)

    print("Feature selection results:")
    print("Shape of train features before feature selection: ",np.shape(trainFeatures))
    print("Shape of train features after feature selection: ",np.shape(newTrainFeatures)) # This is displayed to make sure that the dimensions of the features have changed
    print("Execution time",time.strftime('%H:%M:%S', time.gmtime(t1-t0)))
    print("\n")

    return newTrainFeatures, newTestFeatures

def featureLDAReduction(nComponents, trainFeatures, testFeatures, trainLabels):
    # This function performons the LDA reduction of the features

    t0 = time.time()
    
    LDASelector = LinearDiscriminantAnalysis(n_components=nComponents)
    newTrainFeatures = LDASelector.fit_transform(trainFeatures,trainLabels) # Reduce the dimensionality of the features
    newTestFeatures = LDASelector.transform(testFeatures) # Reduce the dimensionality of the test features in consonance
    
    t1 = time.time()
    
    print("LDA results:")
    print("Shape of train features before LDA transform: ",np.shape(trainFeatures))
    print("Shape of train features after LDA transform: ",np.shape(newTrainFeatures)) # This is displayed to make sure that the dimensions of the features have changed
    print("Execution time",time.strftime('%H:%M:%S', time.gmtime(t1-t0)))
    print("\n")

    return newTrainFeatures, newTestFeatures
