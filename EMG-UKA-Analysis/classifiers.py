from datasetManipulation import mergeDataset, transitionLabelToSimple
from globalVars import KEEP_ALL_FRAMES_IN_TEST, REMOVE_SILENCES
from globalVars import PHONE_DICT, FIRST_TRANSITION_PHONEME
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split

def completeDummies(df, uniqueLabels):
    i = 0
    for label in uniqueLabels:
        if i < len(df.columns):
            if label != df.columns[i]:
                df.insert(i, label, 0)
        else:
            df[label] = 0
        i += 1

    return df

def crossValidationBaggingClassifier(trainFeatures, trainLabels, uniqueLabels, n_estimators=10, min_samples_leaf=10, kFolds=5):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=kFolds)

    score = 0.0
    nLabels = len(uniqueLabels)
    confusionMatrix = np.zeros((nLabels,nLabels))

    for trainIdx, testIdx in kf.split(trainFeatures):
        trainFeaturesSubset = []
        trainLabelsSubset = []
        testFeaturesSubset = []
        testLabelsSubset = []
        
        for idx in trainIdx:
            trainFeaturesSubset.append(trainFeatures[idx])
            trainLabelsSubset.append(trainLabels[idx])
        for idx in testIdx:
            testFeaturesSubset.append(trainFeatures[idx])
            testLabelsSubset.append(trainLabels[idx])

        clf = trainBaggingClassifier(trainFeaturesSubset, trainLabelsSubset,n_estimators=10, min_samples_leaf=10)
        actualScore, actualConfusionMatrix, _ = testClassifier(clf, testFeaturesSubset, testLabelsSubset, uniqueLabels)
        score += actualScore
        confusionMatrix += actualConfusionMatrix

    accuracy = score/kFolds

    return accuracy, confusionMatrix

def crossValidationGMM(trainFeatures, trainLabels, uniqueLabels, kFolds=5):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=kFolds)

    score = 0.0
    nLabels = len(uniqueLabels)
    confusionMatrix = np.zeros((nLabels,nLabels))
    for trainIdx, testIdx in kf.split(trainFeatures):
        trainFeaturesSubset = []
        trainLabelsSubset = []
        testFeaturesSubset = []
        testLabelsSubset = []
        
        for idx in trainIdx:
            trainFeaturesSubset.append(trainFeatures[idx])
            trainLabelsSubset.append(trainLabels[idx])
        for idx in testIdx:
            testFeaturesSubset.append(trainFeatures[idx])
            testLabelsSubset.append(trainLabels[idx])

        clf = trainGMMmodels(trainFeaturesSubset, trainLabelsSubset, uniqueLabels)
        actualScore, actualConfusionMatrix, _ = testGMMmodels(clf, testFeaturesSubset, testLabelsSubset, uniqueLabels)
        score += actualScore
        confusionMatrix += actualConfusionMatrix

    averagedScore = score/kFolds

    return averagedScore, confusionMatrix

def crossValidationNeuralNetwork(trainFeatures, trainLabels, uniqueLabels, batch_size, n_epochs, kFolds=5):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=kFolds)

    score = 0.0
    nLabels = len(uniqueLabels)
    confusionMatrix = np.zeros((nLabels,nLabels))
    for trainIdx, testIdx in kf.split(trainFeatures):
        trainFeaturesSubset = []
        trainLabelsSubset = []
        testFeaturesSubset = []
        testLabelsSubset = []
        
        for idx in trainIdx:
            trainFeaturesSubset.append(trainFeatures[idx])
            trainLabelsSubset.append(trainLabels[idx])
        for idx in testIdx:
            testFeaturesSubset.append(trainFeatures[idx])
            testLabelsSubset.append(trainLabels[idx])

        clf = trainNeuralNetwork(trainFeaturesSubset, trainLabelsSubset, uniqueLabels, batch_size, n_epochs)
        actualScore, actualConfusionMatrix, _ = testNeuralNetwork(clf, testFeaturesSubset, testLabelsSubset, uniqueLabels, batch_size)
        score += actualScore
        confusionMatrix += actualConfusionMatrix

    averagedScore = score/kFolds

    return averagedScore, confusionMatrix

def testClassifier(clf, uttTestFeatures, uttTestLabels, uniqueLabelsArg):
    # This function calculates the accuracy and saves the confusion matrix of the classification obtained with the given classifier
    from featureSelectionProbe import getUniqueLabels

    # Check if there is any unique label in the testing subset that is not in the training subset
    uniqueLabels = uniqueLabelsArg.copy()

    testUniqueLabels = getUniqueLabels(uttTestLabels)
    for label in testUniqueLabels:
        if not label in uniqueLabels:
            #print(f"Label added to uniqueLabels: {PHONE_DICT[label]}")
            np.append(uniqueLabels,label)

    testFeatures, testLabels = mergeDataset(uttTestFeatures, uttTestLabels)

    nSamples = len(testLabels)
    nLabels = len(uniqueLabels)

    confusionMatrix = np.zeros((nLabels,nLabels))
    score = 0.0

    predictions = clf.predict(testFeatures)

    for n in range(nSamples):
        predictedLabel = predictions[n]
        trueLabel = testLabels[n]

        # Update the score if necessary
        if predictedLabel == trueLabel:
            score += 1

        # Add the example to the confusion matrix
        predictedIdx = np.where(uniqueLabels == predictedLabel)[0][0]
        trueIdx = np.where(uniqueLabels == trueLabel)[0][0]

        confusionMatrix[trueIdx,predictedIdx] += 1

    score = score/nSamples

    return score, confusionMatrix, uniqueLabels

def testGMMmodels(models, uttTestFeatures, uttTestLabels, uniqueLabelsArg):
    # This function calculates the accuracy and saves the confusion matrix of the model
    from featureSelectionProbe import getUniqueLabels

    # Check if there is any unique label in the testing subset that is not in the training subset
    uniqueLabels = uniqueLabelsArg.copy()

    testUniqueLabels = getUniqueLabels(uttTestLabels)
    for label in testUniqueLabels:
        if not label in uniqueLabels:
            #print(f"Label added to uniqueLabels: {PHONE_DICT[label]}")
            np.append(uniqueLabels,label)

    testFeatures, testLabels = mergeDataset(uttTestFeatures,uttTestLabels)

    predictedLabels = []

    nSamples = len(testLabels)
    nLabels = len(uniqueLabels)
    
    confusionMatrix = np.zeros((nLabels,nLabels))
    score = 0.0
    
    for n in range(nSamples):
        probs = np.zeros(np.shape(uniqueLabels)) # The array to save the likelihood of the sample with each model
        # Calculate the probability for each model. There should be as many models as labels in uniqueLabels
        for l in range(len(uniqueLabels)):
            # It could be possible that a label of unique labels were not present in the training dataset, but it's present in the test dataset, so it appears in uniqueLabels
            # If that label were not present in the training dataset, there is no model trained for that label
            if uniqueLabels[l] in models.keys(): 
                probs[l] = models[uniqueLabels[l]].score([testFeatures[n]])
            else:
                probs[l] = -100000
    
        # Find the label whose model has obtained the maximum likelihood
        predictedLabel = uniqueLabels[probs.argmax()]
        predictedLabels.append(predictedLabel) # Just for debugging
        trueLabel = testLabels[n]
    
        # Update the score (number of well classificated examples)
        if predictedLabel == trueLabel:
            score += 1
    
        # Add the example to the confusion matrix
        predictedIdx = probs.argmax()
        trueIdx = np.where(uniqueLabels == trueLabel)[0][0]
    
        confusionMatrix[trueIdx,predictedIdx] += 1 # Confusion matrix: rows -> true labels; columns -> predicted labels

    score = score/nSamples

    return score, confusionMatrix, uniqueLabels

def testNeuralNetwork(clf, uttTestFeatures, uttTestLabels, uniqueLabelsArg, batch_size):
    import pandas as pd
    import numpy as np
    from featureSelectionProbe import getUniqueLabels
    
    # Check if there is any unique label in the testing subset that is not in the training subset
    uniqueLabels = uniqueLabelsArg.copy()

    testUniqueLabels = getUniqueLabels(uttTestLabels)
    for label in testUniqueLabels:
        if not label in uniqueLabels:
            #print(f"Label added to uniqueLabels: {PHONE_DICT[label]}")
            np.append(uniqueLabels,label)

    testFeatures, testLabels = mergeDataset(uttTestFeatures, uttTestLabels)
    
    nSamples = len(testLabels)
    nLabels = len(uniqueLabels)

    confusionMatrix = np.zeros((nLabels,nLabels))
    score = 0.0

    predictions = clf.predict(testFeatures)
    predictions=np.argmax(predictions,axis=1)
    
    nBatches = int(nSamples/batch_size)

    for batch in range(1,nBatches+1):
        for n in range(1,batch_size+1):
            idx = (batch*n)-1
            predictedLabel = predictions[idx]
            predictedLabel = uniqueLabels[predictedLabel]
            trueLabel = testLabels[idx]

            # Update the score if necessary
            if predictedLabel == trueLabel:
                score += 1

            # Add the example to the confusion matrix
            predictedIdx = np.where(uniqueLabels == predictedLabel)[0][0]
            if np.where(uniqueLabels == trueLabel)[0]:
                trueIdx = np.where(uniqueLabels == trueLabel)[0][0]

            confusionMatrix[trueIdx,predictedIdx] += 1
        

    score = score/nSamples

    return score, confusionMatrix, uniqueLabels

def trainBaggingClassifier(uttTrainFeatures, uttTrainLabels,n_estimators=10, min_samples_leaf=10):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier

    trainFeatures, trainLabels = mergeDataset(uttTrainFeatures,uttTrainLabels)

    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=43, min_samples_leaf=min_samples_leaf),n_estimators=n_estimators,random_state=0).fit(trainFeatures, trainLabels)

    return clf

def trainGMMmodels(uttTrainFeatures, uttTrainLabels, uniqueLabels):
    # This function trains many GMM models for each label and saves the one that minimizes the Bayesian Information Criterion.
    # It returns the optimal GMM model for each feature into a dictionary, whose keys are the label

    trainFeatures, trainLabels = mergeDataset(uttTrainFeatures,uttTrainLabels)

    max_components = 21
    
    label_models = {}
    
    for label in uniqueLabels:        
        # It could be possible that one of the unique labels is not present in the training dataset, because it appears only in the test dataset.
        # A GMM only can be fitted for a label when that label is present into the training subset.
        if np.count_nonzero(trainLabels == label) > 1: # The minimum number of examples that a label must have to fit a model is 2
            # Select only the features labeled with 'label'
            nz = trainLabels == label

            labelFeatures = trainFeatures[nz == True, :]
    
            # The first model is trained with just 1 component
            modelA = GMM(1,covariance_type='full',random_state=0).fit(labelFeatures)
        
            lastBic = modelA.bic(labelFeatures)
        
            # Many number of components are tryied until finding the minimum or reaching 'max_components'
            for n_components in range(2,max_components):
                modelB = GMM(n_components,covariance_type='full',random_state=0).fit(labelFeatures)
                actualBic = modelB.bic(labelFeatures)
                if actualBic > lastBic: # If the new bic is greater than previous, that means that previous model has the minimum bic
                    break
                else: # Set actual model as previous model and keep with the algorithm
                    lastBic = actualBic
                    modelA = modelB
        
            label_models[label] = modelA # The algorithm stops when modelB.bic > modelA.bic, so modelA is better then modelB

    return label_models

def trainNeuralNetwork(trainFeatures, trainLabels, uniqueLabels, batch_size, n_epochs):
    from keras.callbacks import EarlyStopping
    from keras.models import Sequential
    from keras.layers import Dense
    import random
    import pandas as pd
    
    from featureSelectionProbe import getUniqueLabels

    lenValidationSubset = round(len(trainFeatures)*0.2) # The number of utterances we want to use as validation subset

    # Select randomly the same subset of utterances as validation subset
    # Get the indexes of the utterances that will be into the validation subset
    random.seed(19)
    validationIdx = random.sample([x for x in range(0,len(trainLabels))], lenValidationSubset)

    # Divide the utterances between training subset and validation subset
    trainFeaturesSubset = []
    trainLabelsSubset = []
    valFeaturesSubset = []
    valLabelsSubset = []
    for i in range(0,len(trainLabels)):
        if i in validationIdx:
            valFeaturesSubset.append(trainFeatures[i])
            valLabelsSubset.append(trainLabels[i])
        else:
            trainFeaturesSubset.append(trainFeatures[i])
            trainLabelsSubset.append(trainLabels[i])

    # Merge features into batches
    trainFeaturesBatch, trainLabelsBatch = mergeDataset(trainFeaturesSubset,trainLabelsSubset)
    valFeaturesBatch, valLabelsBatch = mergeDataset(valFeaturesSubset,valLabelsSubset)

    trainLabelsBatch = pd.get_dummies(trainLabelsBatch)
    valLabelsBatch = pd.get_dummies(valLabelsBatch)
    print(len(uniqueLabels))
    
    # Maybe some batches don't contain all labels. If not, insert new column
    trainLabelsBatch = completeDummies(trainLabelsBatch, uniqueLabels)
    valLabelsBatch = completeDummies(valLabelsBatch, uniqueLabels)

    # define the keras model
    model = Sequential()
    #model.add(Dense(len(trainFeatures[0])*2, input_dim=len(trainFeatures[0]), activation='relu'))
    model.add(Dense(32, input_dim=len(trainFeaturesBatch[0]), activation='relu'))
    model.add(Dense(len(uniqueLabels), activation='softmax'))
    
    # compile the keras model
    model.compile(loss="categorical_crossentropy", optimizer= "adam", metrics=['accuracy'])
    
    early_stopping = EarlyStopping(patience=10)

    # fit the keras model on the dataset
    model.fit(trainFeaturesBatch, trainLabelsBatch, validation_data=(valFeaturesBatch, valLabelsBatch), epochs=n_epochs, batch_size=batch_size, callbacks=[early_stopping])
    
    return model

def testClassifierAllFrames(clf, uttTestFeatures, uttTestLabels, uniqueLabelsArg):
    # This function calculates the accuracy and saves the confusion matrix of the classification obtained with the given classifier
    # This function is different from 'testClassifier()' because this one assumes that there are only simple labels in train subset, but all labels in test subset
    # The transition labels in test subset will be converted to simple labels, and the classification will be tested as if they were simple labels.
    # This function will be return, as well as the absolute classification score (using all labels), the simple classification score (classification score obtained classifying only simple labels)
    # and the transition classification score, that is, the score obtained classifying the transition labels converted to simple labels.

    from featureSelectionProbe import getUniqueLabels

    # Check if there is any unique label in the testing subset that is not in the training subset
    uniqueLabels = uniqueLabelsArg.copy()

    testUniqueLabels = getUniqueLabels(uttTestLabels)
    for label in testUniqueLabels:
        if not label in uniqueLabels:
            #print(f"Label added to uniqueLabels: {PHONE_DICT[label]}")
            uniqueLabels = np.append(uniqueLabels,label)

    uniqueSimpleLabels = []
    uniqueTransitionLabels = []

    for label in uniqueLabels:
        if label < FIRST_TRANSITION_PHONEME:
            uniqueSimpleLabels.append(label)
        else:
            uniqueTransitionLabels.append(label)

    testFeatures, testLabels = mergeDataset(uttTestFeatures, uttTestLabels)

    nSamples = len(testLabels)
    #nLabels = len(uniqueLabels)
    
    nSimpleLabels = len(uniqueSimpleLabels)
    #nTransitionLabels = len(uniqueTransitionLabels)

    # This confusion matrix will contain the classification of all kind of labels (simple and transitions, no transition to simple conversion)
    confusionMatrix = np.zeros((nSimpleLabels,nSimpleLabels))

    # This confusion matrix will contain the classification of simple true labels
    simpleConfusionMatrix = np.zeros((nSimpleLabels,nSimpleLabels))
    # This confusion matrix will contain the classification of transition labels converted to simple labels
    transitionConfusionMatrix = np.zeros((nSimpleLabels,nSimpleLabels))

    # The absolute classification score. Includes both classification of simple true labels and transition labels converted to simple
    score = 0.0
    simpleScore = 0.0
    transitionScore = 0.0

    nSimpleExamples = 0 # Number of examples whose true label is simple
    nTransitionExamples = 0 # Number of examples whose true label is transition

    predictions = clf.predict(testFeatures)

    for n in range(nSamples):
        predictedLabel = predictions[n]
        trueLabel = testLabels[n]

        if trueLabel < FIRST_TRANSITION_PHONEME: # If the true label is a simple labels
            
            nSimpleExamples += 1

            # Update the score if necessary
            if predictedLabel == trueLabel:
                score += 1
                simpleScore += 1

            # Add the example to the confusion matrix
            predictedIdx = np.where(uniqueLabels == predictedLabel)[0][0]
            trueIdx = np.where(uniqueLabels == trueLabel)[0][0]

            confusionMatrix[trueIdx,predictedIdx] += 1
            simpleConfusionMatrix[trueIdx,predictedIdx] += 1

        else: # If the true label is a transition label

            nTransitionExamples += 1

            trueSimpleLabel = transitionLabelToSimple(trueLabel) # Convert the true label to a simple label

            # Update the score if necessary
            if predictedLabel == trueLabel:
                score += 1
                transitionScore += 1

            # Add the example to confusion matrices
            predictedIdx = np.where(uniqueLabels == predictedLabel)[0][0]
            trueIdx = np.where(uniqueLabels == trueLabel)[0][0]
            trueSimpleIdx = np.where(uniqueLabels == trueSimpleLabel)

            confusionMatrix[trueSimpleIdx,predictedIdx] += 1
            transitionConfusionMatrix[trueSimpleIdx,predictedIdx] += 1

    print(f"N. simple examples: {nSimpleExamples}")
    print(f"N. transition examples: {nTransitionExamples}")
    print(f"N. exampres: {nSamples}")

    score = score/nSamples
    simpleScore = simpleScore/nSimpleExamples
    transitionScore = transitionScore/nTransitionExamples

    return score, simpleScore, transitionScore, confusionMatrix, simpleConfusionMatrix, transitionConfusionMatrix, uniqueLabels, uniqueSimpleLabels, uniqueTransitionLabels

def testGMMmodelsAllFrames(models, uttTestFeatures, uttTestLabels, uniqueLabelsArg):
    # This function calculates the accuracy and saves the confusion matrix of the classification obtained with the given classifier
    # This function is different from 'testClassifier()' because this one assumes that there are only simple labels in train subset, but all labels in test subset
    # The transition labels in test subset will be converted to simple labels, and the classification will be tested as if they were simple labels.
    # This function will be return, as well as the absolute classification score (using all labels), the simple classification score (classification score obtained classifying only simple labels)
    # and the transition classification score, that is, the score obtained classifying the transition labels converted to simple labels.
    from featureSelectionProbe import getUniqueLabels

    # Check if there is any unique label in the testing subset that is not in the training subset
    uniqueLabels = uniqueLabelsArg.copy()

    testUniqueLabels = getUniqueLabels(uttTestLabels)
    for label in testUniqueLabels:
        if not label in uniqueLabels:
            #print(f"Label added to uniqueLabels: {PHONE_DICT[label]}")
            uniqueLabels = np.append(uniqueLabels,label)

    uniqueSimpleLabels = []
    uniqueTransitionLabels = []

    #predictedLabels = []

    for label in uniqueLabels:
        if label < FIRST_TRANSITION_PHONEME:
            uniqueSimpleLabels.append(label)
        else:
            uniqueTransitionLabels.append(label)

    testFeatures, testLabels = mergeDataset(uttTestFeatures, uttTestLabels)

    nSamples = len(testLabels)
    #nLabels = len(uniqueLabels)
    
    nSimpleLabels = len(uniqueSimpleLabels)
    #nTransitionLabels = len(uniqueTransitionLabels)

    # This confusion matrix will contain the classification of all kind of labels (simple and transitions, no transition to simple conversion)
    confusionMatrix = np.zeros((nSimpleLabels,nSimpleLabels))

    # This confusion matrix will contain the classification of simple true labels
    simpleConfusionMatrix = np.zeros((nSimpleLabels,nSimpleLabels))
    # This confusion matrix will contain the classification of transition labels converted to simple labels
    transitionConfusionMatrix = np.zeros((nSimpleLabels,nSimpleLabels))

    # The absolute classification score. Includes both classification of simple true labels and transition labels converted to simple
    score = 0.0
    simpleScore = 0.0
    transitionScore = 0.0

    nSimpleExamples = 0 # Number of examples whose true label is simple
    nTransitionExamples = 0 # Number of examples whose true label is transition
    
    for n in range(nSamples):
        probs = np.zeros(np.shape(uniqueLabels)) # The array to save the likelihood of the sample with each model
        # Calculate the probability for each model. There should be as many models as labels in uniqueLabels
        for l in range(len(uniqueLabels)):
            # It could be possible that a label of unique labels were not present in the training dataset, but it's present in the test dataset, so it appears in uniqueLabels
            # If that label were not present in the training dataset, there is no model trained for that label
            if uniqueLabels[l] in models.keys(): 
                probs[l] = models[uniqueLabels[l]].score([testFeatures[n]])
            else:
                probs[l] = -100000
    
        # Find the label whose model has obtained the maximum likelihood
        predictedLabel = uniqueLabels[probs.argmax()]
        #predictedLabels.append(predictedLabel) # Just for debugging
        trueLabel = testLabels[n]

        if trueLabel < FIRST_TRANSITION_PHONEME: # If the true label is a simple labels
            
            nSimpleExamples += 1

            # Update the score if necessary
            if predictedLabel == trueLabel:
                score += 1
                simpleScore += 1

            # Add the example to the confusion matrix
            predictedIdx = np.where(uniqueLabels == predictedLabel)[0][0]
            trueIdx = np.where(uniqueLabels == trueLabel)[0][0]

            confusionMatrix[trueIdx,predictedIdx] += 1
            simpleConfusionMatrix[trueIdx,predictedIdx] += 1

        else: # If the true label is a transition label

            nTransitionExamples += 1

            trueSimpleLabel = transitionLabelToSimple(trueLabel) # Convert the true label to a simple label

            # Update the score if necessary
            if predictedLabel == trueLabel:
                score += 1
                transitionScore += 1

            # Add the example to confusion matrices
            predictedIdx = np.where(uniqueLabels == predictedLabel)[0][0]
            trueIdx = np.where(uniqueLabels == trueLabel)[0][0]
            trueSimpleIdx = np.where(uniqueLabels == trueSimpleLabel)

            confusionMatrix[trueSimpleIdx,predictedIdx] += 1
            transitionConfusionMatrix[trueSimpleIdx,predictedIdx] += 1

    print(f"N. simple examples: {nSimpleExamples}")
    print(f"N. transition examples: {nTransitionExamples}")
    print(f"N. exampres: {nSamples}")

    score = score/nSamples
    simpleScore = simpleScore/nSimpleExamples
    transitionScore = transitionScore/nTransitionExamples

    return score, simpleScore, transitionScore, confusionMatrix, simpleConfusionMatrix, transitionConfusionMatrix, uniqueLabels, uniqueSimpleLabels, uniqueTransitionLabels

def testNeuralNetworkAllFrames(clf, uttTestFeatures, uttTestLabels, uniqueLabelsArg, batch_size):
    import pandas as pd
    import numpy as np
    from featureSelectionProbe import getUniqueLabels
    
    # Check if there is any unique label in the testing subset that is not in the training subset
    uniqueLabels = uniqueLabelsArg.copy()

    testUniqueLabels = getUniqueLabels(uttTestLabels)
    for label in testUniqueLabels:
        if not label in uniqueLabels:
            #print(f"Label added to uniqueLabels: {PHONE_DICT[label]}")
            uniqueLabels = np.append(uniqueLabels,label)

    uniqueSimpleLabels = []
    uniqueTransitionLabels = []

    #predictedLabels = []

    for label in uniqueLabels:
        if label < FIRST_TRANSITION_PHONEME:
            uniqueSimpleLabels.append(label)
        else:
            uniqueTransitionLabels.append(label)

    testFeatures, testLabels = mergeDataset(uttTestFeatures, uttTestLabels)

    nSamples = len(testLabels)
    #nLabels = len(uniqueLabels)
    
    nSimpleLabels = len(uniqueSimpleLabels)
    #nTransitionLabels = len(uniqueTransitionLabels)

    # This confusion matrix will contain the classification of all kind of labels (simple and transitions, no transition to simple conversion)
    confusionMatrix = np.zeros((nSimpleLabels,nSimpleLabels))

    # This confusion matrix will contain the classification of simple true labels
    simpleConfusionMatrix = np.zeros((nSimpleLabels,nSimpleLabels))
    # This confusion matrix will contain the classification of transition labels converted to simple labels
    transitionConfusionMatrix = np.zeros((nSimpleLabels,nSimpleLabels))

    # The absolute classification score. Includes both classification of simple true labels and transition labels converted to simple
    score = 0.0
    simpleScore = 0.0
    transitionScore = 0.0

    nSimpleExamples = 0 # Number of examples whose true label is simple
    nTransitionExamples = 0 # Number of examples whose true label is transition
    predictions = clf.predict(testFeatures)
    predictions=np.argmax(predictions,axis=1)
    
    nBatches = int(nSamples/batch_size)

    for batch in range(1,nBatches+1):
        for n in range(1,batch_size+1):
            idx = (batch*n)-1
            predictedLabel = predictions[idx]
            predictedLabel = uniqueLabels[predictedLabel]
            trueLabel = testLabels[idx]

            if trueLabel < FIRST_TRANSITION_PHONEME: # If the true label is a simple labels
            
                nSimpleExamples += 1

                # Update the score if necessary
                if predictedLabel == trueLabel:
                    score += 1
                    simpleScore += 1

                # Add the example to the confusion matrix
                predictedIdx = np.where(uniqueLabels == predictedLabel)[0][0]
                trueIdx = np.where(uniqueLabels == trueLabel)[0][0]

                confusionMatrix[trueIdx,predictedIdx] += 1
                simpleConfusionMatrix[trueIdx,predictedIdx] += 1

            else: # If the true label is a transition label

                nTransitionExamples += 1

                trueSimpleLabel = transitionLabelToSimple(trueLabel) # Convert the true label to a simple label

                # Update the score if necessary
                if predictedLabel == trueLabel:
                    score += 1
                    transitionScore += 1

                # Add the example to confusion matrices
                predictedIdx = np.where(uniqueLabels == predictedLabel)[0][0]
                trueIdx = np.where(uniqueLabels == trueLabel)[0][0]
                trueSimpleIdx = np.where(uniqueLabels == trueSimpleLabel)

                confusionMatrix[trueSimpleIdx,predictedIdx] += 1
                transitionConfusionMatrix[trueSimpleIdx,predictedIdx] += 1

    print(f"N. simple examples: {nSimpleExamples}")
    print(f"N. transition examples: {nTransitionExamples}")
    print(f"N. exampres: {nSamples}")

    score = score/nSamples
    simpleScore = simpleScore/nSimpleExamples
    transitionScore = transitionScore/nTransitionExamples

    return score, simpleScore, transitionScore, confusionMatrix, simpleConfusionMatrix, transitionConfusionMatrix, uniqueLabels, uniqueSimpleLabels, uniqueTransitionLabels