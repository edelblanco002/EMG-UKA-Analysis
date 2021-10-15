classifierEvaluation module
===========================

.. automodule:: classifierEvaluation
   :members:
   :undoc-members:
   :show-inheritance:

Dependencies
------------
The following imports are necessary to use this module::

   import matplotlib.pyplot as plt
   import numpy as np
   import pandas as pd
   import pickle
   import seaborn as sns

Classes
-------

   .. class::  classifierEvaluation.LabelOutcome([TP=0,TN=0,FP=0,FN=0])

      This class stores the number of true positives, true negatives, false positives and false negatives for one class obtained from testing a classifier with a set of examples, and calculates some outcomes with those data. The object can be created with its attributes or add them later.

      :param TP: Number of true positives.
      :type TP: int, default=0
      :param TN: Number of true negatives.
      :type TN: int, default=0
      :param FP: Number of false positives.
      :type FP: int, default=0
      :param FN: Number of false negatives.
      :type FN: int, default=0

      .. method:: classifierEvaluation.LabelOutcome.setTP(TP)

         This method sets the number of true positives.

         :param TP: Number of true positives.
         :type TP: int

      .. method:: classifierEvaluation.LabelOutcome.setTN(TN)

         This method sets the number of true negatives.

         :param TN: Number of true negatives.
         :type TN: int

      .. method:: classifierEvaluation.LabelOutcome.setFP(FP)

         This method sets the number of false positives.

         :param FP: Number of false positives.
         :type FP: int

      .. method:: classifierEvaluation.LabelOutcome.setFN(FN)

         This method sets the number of false negatives.

         :param FN: Number of false negatives.
         :type FN: int

      .. method:: classifierEvaluation.LabelOutcome.getTP()

         This method returns the number of true positives.

         :rtype: int
         :return: The number of true positives.

      .. method:: classifierEvaluation.LabelOutcome.getTN()

         This method returns the number of true negatives.

         :rtype: int
         :return: The number of true negatives.

      .. method:: classifierEvaluation.LabelOutcome.getFP()

         This method returns the number of false positives.

         :rtype: int
         :return: The number of false positives.

      .. method:: classifierEvaluation.LabelOutcome.getFN()

         This method returns the number of false negatives.

         :rtype: int
         :return: The number of false negatives.

      .. method:: classifierEvaluation.LabelOutcome.getTotalPopulation()

         This method returns the total numbers of examples that have been classified.

         :rtype: int
         :return: The number of false negatives.

      .. method:: classifierEvaluation.LabelOutcome.getSensitivity()

         This method returns the sensitivity of the class.

         :rtype: float
         :return: The sensitivity of the class.

      .. method:: classifierEvaluation.LabelOutcome.getSpecificity()

         This method returns the specificity of the class.

         :rtype: float
         :return: The specificity of the class.

Functions
---------

   .. function:: classifierEvaluation.loadProbeResults(dirPath,scriptPath,experimentName,probe,subset)
      
      This function loads the confusion matrix, the phone dict and the list of unique labels.

      :param dirPath: The base path of the corpus that is going to be analyzed.
      :type dirPath: str
      :param scriptPath: The path where the ``EMG-UKA-Analysis`` scripts are saved.
      :type scriptPath: str
      :param experimentName: The name of the set of experiments that are being executed.
      :type experimentName: str
      :param probe: The probe that is being analyzed.
      :type probe: <:class:`featureSelectionProbe.Probe`>
      :param subset: The subset that is being analyzed ('Train' or 'Test').
      :type subset: str
      :rtype: tuple
      :return: (**confusionMatrix**, **phoneDict**, **uniqueLabels**, **uniquePhones**)

         * **confusionMatrix** (*numpy.ndarray*) - The confusion matrix resulting from testing the classifier with the given subset. It’s not normalized.
         * **phoneDict** (*dict*) - A dictionary that saves the relations between the name of the label and the number assigned to it. The keys are the numbers corresponding to the labels and the returned values are the labels as strings.
         * **uniqueLabels** (*list*) - A list of the labels that were present in the training batch.
         * **uniquePhones** (*list*) - A list of the phonemes that were present in the training batch. It is the same as ``uniqueLabels``, but it contains the phonemes in ``str`` format instead of numeric labels.

   .. function:: classifierEvaluation.drawConfusionMatrix(dirPath,scriptPath,experimentName,probe,subset)

      This function draws a normalized confusion matrix and export the resulting figure as a ``.png`` image. This image will be saved into the ``{dirPath}/results/{experimentName}`` folder.

      :param dirPath: The base path of the corpus that is going to be analyzed.
      :type dirPath: str
      :param scriptPath: The path where the ``EMG-UKA-Analysis`` scripts are saved.
      :type scriptPath: str
      :param experimentName: The name of the set of experiments that are being executed.
      :type experimentName: str
      :param probe: The probe that is being analyzed.
      :type probe: <:class:`featureSelectionProbe.Probe`>
      :param subset: The subset that is being analyzed ('Train' or 'Test').
      :type subset: str

      The imagen will be similar to this:

      .. image:: images/confusionMatrixExample.png
         :width: 600

   .. function:: classifierEvaluation.getOutcomes(dirPath,scriptPath,experimentName,probe,subset)

      This function generates a text file with a table in LaTeX format (tabular) that contains the outcomes for each label: True Positives, True Negatives, False Positives, False Negatives, Sensitivity, Specificity, Precision and Recall. The text file will be saved into the ``{dirPath}/results/{experimentName}`` folder.

      :param dirPath: The base path of the corpus that is going to be analyzed.
      :type dirPath: str
      :param scriptPath: The path where the ``EMG-UKA-Analysis`` scripts are saved.
      :type scriptPath: str
      :param experimentName: The name of the set of experiments that are being executed.
      :type experimentName: str
      :param probe: The probe that is being analyzed.
      :type probe: <:class:`featureSelectionProbe.Probe`>
      :param subset: The subset that is being analyzed ('Train' or 'Test').
      :type subset: str

      When the obtained text file is inserted into a LaTeX, the table in the compiled document will look like this:

      .. image:: images/outcomesTableExample.png

   .. function:: classifierEvaluation.main(dirPath,scriptPath,experimentName,probes)

      This is the main function of the module. It draws a confusion matrix and creates an outcome table for train and test subset for every probe programed in the execution.

      :param dirPath: The base path of the corpus that is going to be analyzed.
      :type dirPath: str
      :param scriptPath: The path where the ``EMG-UKA-Analysis`` scripts are saved.
      :type scriptPath: str
      :param experimentName: The name of the set of experiments that are being executed.
      :type experimentName: str
      :param probe: The probe that is being analyzed.
      :type probe: <:class:`featureSelectionProbe.Probe`>