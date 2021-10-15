classifiers module
==================

.. automodule:: classifiers
   :members:
   :undoc-members:
   :show-inheritance:

Dependencies
------------

The libraries needed to use this moddules are the following ones::

   import numpy as np
   from sklearn.mixture import GaussianMixture as GMM
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.ensemble import BaggingClassifier

Functions
---------

.. function:: classifiers.testClassifier(clf, testFeatures, testLabels, uniqueLabels)
   
   This function calculates the accuracy and saves the confusion matrix of the classification obtained with the given classifier.


   :param clf: A classifier from sklearn. It must be have the `predict` method, so it can perform predictions on input samples.
   :param testFeatures: A matrix with the features of the subset the classifier is going to be tested with.
   :type testFeatures: numpy.ndarray
   :param testLabels: An array with the features of the subset the classifier is going to be tested with.
   :type testLabels: numpy.ndarray
   :param uniqueLabels: An array containing a list of the labels that were present in the training batch.
   :type uniqueLabels: list
   :rtype: tuple
   :return: (**score**,**confusionMatrix**)

      * **score** (*float*) - The accuracy resulting from testing the classifier with the given subset.
      * **confusionMatrix** (*numpy.ndarray*) . The confusion matrix resulting from testing the classifier with the given subset. It's not normalized.

   Example of use::

      >>> from sklearn.datasets import load_iris
      >>> from sklearn.model_selection import train_test_split
      >>> from sklearn.tree import DecisionTreeClassifier
      >>> import classifiers
      >>> import featureSelectionProbe
      >>> iris = load_iris()
      >>> trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(iris.data,iris.target, test_size=0.33)
      >>> clf = DecisionTreeClassifier()
      >>> clf = clf.fit(trainFeatures,trainLabels)
      >>> uniqueLabels = featureSelectionProbe.getUniqueLabels(trainLabels)
      >>> score, confusionMatrix = classifiers.testClassifier(clf, testFeatures, testLabels, uniqueLabels)
      >>> print(f"Achieved accuracy: {score}")
      Achieved accuracy: 0.9
      >>> print(confusionMatrix)
      [[17.  0.  0.]
      [ 0. 12.  3.]
      [ 0.  2. 16.]]

.. function:: classifiers.testGMMmodels(models, testFeatures, testLabels, uniqueLabels)

   This function serves for testing the classifiers created with :func:`classifiers.trainGMMmodels`. This function calculates the accuracy and saves the confusion matrix of the classifier testing with the given subset.

   :param models: A dictionary that contains GMM models (:class:`sklearn.mixture.GaussianMixture`) obtained with the :func:`classifiers.trainGMMmodels`. There is a GMM model for each label, and the key is the number of the label.
   :type models: dict
   :param testFeatures: A matrix with the features of the subset the classifier is going to be tested with.
   :type testFeatures: numpy.ndarray
   :param testLabels: An array with the features of the subset the classifier is going to be tested with.
   :type testLabels: numpy.ndarray
   :param uniqueLabels: An array containing a list of the labels that were present in the training batch.
   :type uniqueLabels: list
   :rtype: tuple
   :return: (**score**,**confusionMatrix**)

      * **score** (*float*) - The accuracy resulting from testing the classifier with the given subset.
      * **confusionMatrix** (*numpy.ndarray*) . The confusion matrix resulting from testing the classifier with the given subset. It's not normalized.

   Example of use::

      >>> from sklearn.datasets import load_iris
      >>> from sklearn.model_selection import train_test_split
      >>> import classifiers
      >>> import featureSelectionProbe
      >>> iris = load_iris()
      >>> trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(iris.data,iris.target, test_size=0.33)
      >>> uniqueLabels = featureSelectionProbe.getUniqueLabels(trainLabels)
      >>> clf = classifiers.trainGMMmodels(trainFeatures,trainLabels,uniqueLabels)
      >>> score, confusionMatrix = classifiers.testGMMmodels(clf,testFeatures,testLabels,uniqueLabels)
      >>> print(f"Achieved accuracy: {score}")
      Achieved accuracy: 0.98
      >>> print(confusionMatrix)
      [[18.  0.  0.]
      [ 0. 13.  1.]
      [ 0.  0. 18.]]

.. function:: classifiers.trainBaggingClassifier(trainFeatures, trainLabels[, n_estimators=10, min_samples_leaf=10])

   This function trains a bagging classifier with the given subset. The base classifier is :class:`sklearn.tree.DecisionTreeClassifier`.

   :param trainFeatures: A matrix with the features of the subset the classifier is going to be trained with.
   :type trainFeatures: numpy.ndarray
   :param trainLabels: An array with the features of the subset the classifier is going to be trained with.
   :type trainLabels: numpy.ndarray
   :param n_estimators: The number of base estimators in the ensemble.
   :type n_estimators: int, default=10
   :param min_samples_leaf: The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least ``min_samples_leaf`` training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.
      * If int, then consider ``min_samples_leaf`` as the minimum number.
      * If float, then ``min_samples_leaf`` is a fraction and ``ceil(min_samples_leaf * n_samples)`` are the minimum number of samples for each node.

   :type min_samples_leaf: int or float
   :rtype: <class 'sklearn.ensemble._bagging.BaggingClassifier'>
   :return: A trained bagging classifier.

   Example of use::

      >>> from sklearn.datasets import load_iris
      >>> from sklearn.model_selection import train_test_split
      >>> import classifiers
      >>> import featureSelectionProbe
      >>> iris = load_iris()
      >>> trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(iris.data,iris.target, test_size=0.33)
      >>> clf = classifiers.trainBaggingClassifier(trainFeatures, trainLabels, n_estimators=12, min_samples_leaf=5)
      >>> type(clf)
      <class 'sklearn.ensemble._bagging.BaggingClassifier'>

.. function:: classifiers.trainGMMmodels(trainFeatures, trainLabels, uniqueLabels)

   This function trains many GMM models for each label and saves the one that minimizes the Bayesian Information Criterion. It returns the optimal GMM model for each feature into a dictionary, whose keys are the label.

   :param trainFeatures: A matrix with the features of the subset the classifier is going to be trained with.
   :type trainFeatures: numpy.ndarray
   :param trainLabels: An array with the features of the subset the classifier is going to be trained with.
   :type trainLabels: numpy.ndarray
   :param uniqueLabels: An array containing a list of the labels that were present in the training batch.
   :type uniqueLabels: list
   :rtype: dict
   :return: A dictionary whose keys are the labels and the content is a GMM model (:class:`sklearn.mixture.GaussianMixture`) for each label.

   Example of use:

      >>> from sklearn.datasets import load_iris
      >>> from sklearn.model_selection import train_test_split
      >>> import classifiers
      >>> import featureSelectionProbe
      >>> # %%
      >>> iris = load_iris()
      >>> trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(iris.data,iris.target, test_size=0.33)
      >>> uniqueLabels = featureSelectionProbe.getUniqueLabels(trainLabels)
      >>> clf = classifiers.trainGMMmodels(trainFeatures,trainLabels,uniqueLabels)
      >>> print([key for key in clf.keys()])
      [0, 1, 2]
      >>> print([clf[key] for key in clf.keys()])
      [GaussianMixture(random_state=0), GaussianMixture(random_state=0), GaussianMixture(random_state=0)]

