from re import sub
import featureSelectionProbe
from globalVars import DIR_PATH
from globalVars import KEEP_ALL_FRAMES_IN_TEST
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn import feature_selection

class LabelOutcome:
    # The object can be created with its attributes or add them later
    def __init__(self,TP=0,TN=0,FP=0,FN=0):
        self.__TP = TP
        self.__TN = TN
        self.__FP = FP
        self.__FN = FN

    # Setters
    def setTP(self,TP):
        self.__TP = TP

    def setTN(self,TN):
        self.__TN = TN

    def setFP(self,FP):
        self.__FP = FP

    def setFN(self,FN):
        self.__FN = FN

    # Getters
    def getTP(self):
        return self.__TP

    def getTN(self):
        return self.__TN

    def getFP(self):
        return self.__FP

    def getFN(self):
        return self.__FN

    # Outcome calculation
    def getTotalPopulation(self):
        return self.__TP + self.__FP + self.__TN + self.__FN

    def getSensitivity(self):
        if (self.__TP + self.__FN) == 0:
            return np.nan
        else:
            return self.__TP/(self.__TP + self.__FN)

    def getSpecificity(self):
        if (self.__TN + self.__FP) == 0:
            return np.nan
        else:
            return self.__TN/(self.__TN + self.__FP)

def loadProbeResults(experimentName,probe: featureSelectionProbe.Probe,subset):
    # This function loads the confussion matrix, the phone dict and the list of unique labels

    confusionMatrix = np.load(f"{DIR_PATH}/results/{experimentName}/{probe.name}_{subset}ConfusionMatrix.npy").astype(int) # [True Labels x Predicted labels]
    if subset == 'Train':
        uniqueLabels = np.load(f"{DIR_PATH}/results/{experimentName}/{probe.name}_uniqueTrainLabels.npy").astype(int)
    else:
        uniqueLabels = np.load(f"{DIR_PATH}/results/{experimentName}/{probe.name}_uniqueLabels.npy").astype(int)
    
    with open(f"{DIR_PATH}/results/{experimentName}/phoneDict.pkl","rb") as file:
        phoneDict = pickle.load(file)
    
    # uniquePhones: list of phones that have been used for classification
    uniquePhones = []
    
    for elem in uniqueLabels:
        uniquePhones.append(phoneDict[elem])
    return confusionMatrix, uniquePhones

def loadProbeResultsAllFrames(experimentName,probe: featureSelectionProbe.Probe):
    # This function loads the confussion matrix, the phone dict and the list of unique labels

    confusionMatrix = np.load(f"{DIR_PATH}/results/{experimentName}/{probe.name}_TestConfusionMatrix.npy").astype(int) # [True Labels x Predicted labels]
    uniqueLabels = np.load(f"{DIR_PATH}/results/{experimentName}/{probe.name}_uniqueLabels.npy").astype(int)
    uniqueSimpleLabels = np.load(f"{DIR_PATH}/results/{experimentName}/{probe.name}_uniqueSimpleLabels.npy").astype(int)

    with open(f"{DIR_PATH}/results/{experimentName}/phoneDict.pkl","rb") as file:
        phoneDict = pickle.load(file)
    
    # uniquePhones: list of phones that have been used for classification
    uniquePhones = []
    uniqueSimplePhones = []
    
    for elem in uniqueLabels:
        uniquePhones.append(phoneDict[elem])

    for elem in uniqueSimpleLabels:
        uniqueSimplePhones.append(phoneDict[elem])

    return confusionMatrix, uniquePhones, uniqueSimplePhones

def drawConfusionMatrix(experimentName,probe: featureSelectionProbe.Probe,subset):
    # This function draws a normalized confusion matrix and export the resulting figure as a image

    if subset == 'Test' and KEEP_ALL_FRAMES_IN_TEST:
        confusionMatrix, uniquePhones, uniqueSimplePhones = loadProbeResultsAllFrames(experimentName,probe)
        dfConfusionMatrix = pd.DataFrame(data=confusionMatrix,index=uniqueSimplePhones,columns=uniqueSimplePhones)
    else:
        confusionMatrix, uniquePhones = loadProbeResults(experimentName,probe,subset)
        dfConfusionMatrix = pd.DataFrame(data=confusionMatrix,index=uniquePhones,columns=uniquePhones)
    
    dfConfusionMatrix = dfConfusionMatrix*100/dfConfusionMatrix.sum(axis=0) # Normalization done by columns: each column represents the 100% of the examples labeled as a same label.
    
    fig, ax = plt.subplots(figsize=(11,11))
    
    ax = sns.heatmap(dfConfusionMatrix, fmt="d", yticklabels=True, xticklabels=True, cmap="Greys", cbar_kws={'format': '%.0f%%'})
    
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    plt.savefig(f"{DIR_PATH}/results/{experimentName}/{probe.name}_{subset}ConfusionMatrix.png")

def getOutcomes(experimentName,probe,subset):
    # This function generates a text file with a table in LaTeX format (tabular) that contains the outcomes for each label:
    # True Positives, True Negatives, False Positives, False Negatives, Sensitivity, Specificity, Precision and Recall.

    confusionMatrix, uniquePhones = loadProbeResults(experimentName,probe,subset)

    outcomes = {} # The outcomes of each label are saved in a dictionary, whose keys are the phoneme that corresponds to each label
    
    for i in range(len(uniquePhones)):
        outcome = LabelOutcome()
        
        TP = confusionMatrix[i,i] # True Positives: Predicted label = True label
        FP = confusionMatrix.sum(axis=0)[i]-TP # False Positives: The sum of the column corresponding to one predicted label (every examples classified as that label) minus the correct labeled examples (True Positives)
        FN = confusionMatrix.sum(axis=1)[i]-TP # False Negatives: The sum of the row corresponding to one true label (every examples whose true label is that label) minus the correct labeled examples (True Positives)
        
        outcome.setTP(TP)
        outcome.setFP(FP)
        outcome.setFN(FN)
        outcome.setTN(confusionMatrix.sum() - TP - FP - FN) # True Negatives: Every examples whose true label was a label different from the actual label and that weren't labeled with the actual label. Every example less TP, FP and FN.
    
        outcomes[uniquePhones[i]] = outcome
    
    # Build the LaTeX table into a text file
    with open(f"{DIR_PATH}/results/{experimentName}/{probe.name}_{subset}OutcomesTable.txt","a+") as file:
        # Build the heading of the table
        file.write('\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n')
        file.write('\t\\textbf{Label} & \\textbf{TP} & \\textbf{TN} & \\textbf{FP} & \\textbf{FN} & \\textbf{Sens.} & \\textbf{Spec.} \\\\\n')
        file.write('\t\\hline\\hline\n')
    
        # Create a row for each label
        for key in outcomes.keys():
            file.write(f"\t\\textbf{{ {key} }} & {outcomes[key].getTP()} & {outcomes[key].getTN()} & {outcomes[key].getFP()} & {outcomes[key].getFN()} & {round(outcomes[key].getSensitivity()*100,2)}\\% & {round(outcomes[key].getSpecificity()*100,2)}\\%\\\\\n")
            file.write("\t\\hline\n")

        file.write('\\end{tabular}')

def main(experimentName,probes):
    
    subsets = ["Train","Test"]

    # For every probe:
    probe: featureSelectionProbe.Probe
    for probe in probes:
        reductionMethod = probe.reductionMethod+probe.scoreFunction+str(probe.n_features)
        classificationMethod = probe.classificationMethod

        # Draw a confusion matrix and an outcomes table both for Train and Test subset
        for subset in subsets:
            drawConfusionMatrix(experimentName,probe,subset)
            if not KEEP_ALL_FRAMES_IN_TEST:
                getOutcomes(experimentName,probe,subset)
