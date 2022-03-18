import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split

def crossValidationBaggingClassifier(trainFeatures, trainLabels,n_estimators=10, min_samples_leaf=10, kFolds=9):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.model_selection import cross_val_score

    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=43, min_samples_leaf=min_samples_leaf),n_estimators=n_estimators,random_state=0)
    scores = cross_val_score(clf, trainFeatures, trainLabels, cv=kFolds)

    accuracy = sum(scores)/len(scores)

    return accuracy

def crossValidationGMM(trainFeatures, trainLabels, uniqueLabels, kFolds=9):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=kFolds)

    scores = []

    for trainIdx, testIdx in kf.split(trainFeatures):
        xTrain, xTest = trainFeatures[trainIdx], trainFeatures[testIdx]
        yTrain, yTest = trainLabels[trainIdx], trainLabels[testIdx]
        clf = trainGMMmodels(xTrain, yTrain, uniqueLabels)
        actualScore, _ = testGMMmodels(clf, xTest, yTest, uniqueLabels)
        scores.append(actualScore)

    averagedScore = sum(scores)/len(scores)

    return averagedScore

def crossValidationNeuralNetwork(trainFeatures, trainLabels, uniqueLabels, batch_size, n_epochs):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=kFolds)

    scores = []

    for trainIdx, testIdx in kf.split(trainFeatures):
        xTrain, xTest = trainFeatures[trainIdx], trainFeatures[testIdx]
        yTrain, yTest = trainLabels[trainIdx], trainLabels[testIdx]
        clf = trainNeuralNetwork(trainFeatures, xTrain, yTrain, batch_size, n_epochs)
        actualScore, _ = testNeuralNetwork(clf, xTest, yTest, uniqueLabels, batch_size)
        scores.append(actualScore)

    averagedScore = sum(scores)/len(scores)

    return averagedScore

def testClassifier(clf, testFeatures, testLabels, uniqueLabels):
    # This function calculates the accuracy and saves the confusion matrix of the classification obtained with the given classifier

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

    return score, confusionMatrix

def testGMMmodels(models, testFeatures, testLabels, uniqueLabels):
    # This function calculates the accuracy and saves the confusion matrix of the model

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

    return score, confusionMatrix

def trainBaggingClassifier(trainFeatures, trainLabels,n_estimators=10, min_samples_leaf=10):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier

    clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=43, min_samples_leaf=min_samples_leaf),n_estimators=n_estimators,random_state=0).fit(trainFeatures, trainLabels)

    return clf

def trainGMMmodels(trainFeatures, trainLabels, uniqueLabels):
    # This function trains many GMM models for each label and saves the one that minimizes the Bayesian Information Criterion.
    # It returns the optimal GMM model for each feature into a dictionary, whose keys are the label

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
    import pandas as pd
    

    print(f"len(trainLabels): {len(trainLabels)}")

    trainFeatures, validationFeatures, trainLabels, validationLabels = train_test_split(trainFeatures, trainLabels, test_size=0.33, random_state=42)

    print(f"len(trainLabels): {len(trainLabels)}")
    print(f"len(validationLabels): {len(validationLabels)}")

    trainLabels = pd.get_dummies(trainLabels)
    validationLabels = pd.get_dummies(validationLabels)
    print(len(uniqueLabels))
    

    # define the keras model
    model = Sequential()
    #model.add(Dense(len(trainFeatures[0])*2, input_dim=len(trainFeatures[0]), activation='relu'))
    model.add(Dense(32, input_dim=len(trainFeatures[0]), activation='relu'))
    model.add(Dense(len(uniqueLabels), activation='softmax'))
    
    # compile the keras model
    model.compile(loss="categorical_crossentropy", optimizer= "adam", metrics=['accuracy'])
    
    early_stopping = EarlyStopping(patience=10)

    # fit the keras model on the dataset
    model.fit(trainFeatures, trainLabels, validation_data=(validationFeatures, validationLabels), epochs=n_epochs, batch_size=batch_size, callbacks=[early_stopping])
    
    return model

def testNeuralNetwork(clf, testFeatures, testLabels, uniqueLabels, batch_size):
    import pandas as pd
    import numpy as np
    
    
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

    return score, confusionMatrix