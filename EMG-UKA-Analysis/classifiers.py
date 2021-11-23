import numpy as np
from sklearn.mixture import GaussianMixture as GMM

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
