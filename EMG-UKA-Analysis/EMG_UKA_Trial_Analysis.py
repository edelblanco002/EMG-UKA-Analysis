import classifierEvaluation
import drawHeatMap
import drawBarPlot
import gatherDataIntoTable
import getFeatureScores
import featureSelectionProbe
from featureSelectionProbe import Probe
import telegramNotification
import traceback

#dirpath = 'C:/Users/Eder/Downloads/EMG-UKA-Trial-Corpus'
#scriptpath = 'C:/Users/Eder/source/repos/EMG-UKA-Trial-Analysis'

dirpath = '/mnt/ldisk/eder/EMG-UKA-Trial-Corpus'
scriptpath = '/home/aholab/eder/scripts/EMG-UKA-Analysis/EMG-UKA-Analysis'

uttType = 'audible' # 'audible', 'whispered' or 'silent'
analyzedLabels = 'Simple' # 'All', 'Simple', 'Transitions' or 'PilotStudy'

#gatherDataIntoTable.main(dirpath,uttType,'train')
#gatherDataIntoTable.main(dirpath,uttType,'test')

experimentName = 'ExperimentBagging6'


# Available reduction methods: 'SelectKBest', 'LDAReduction'
# Available scoreFunction (only for 'SelectKBest'): 'f_classif', 'mutual_into_classif'
# Available classificationMethods = 'GMMmodels', 'bagging'
"""probes = [
   Probe(
       reductionMethod = 'LDAReduction',
       n_features = 38,
       classificationMethod = 'bagging',
       n_estimators = 100,
       min_samples_leaf = 50
       )
]"""

probes = []

for n_estimators in [90, 100, 110, 120, 130, 140, 150]:
    for min_samples_leaf in [30, 40, 50, 60, 70]:
        probes.append(
            Probe(
                reductionMethod = 'SelectKBest',
                scoreFunction = 'f_classif',
                n_features = 100,
                classificationMethod = 'bagging',
                n_estimators = n_estimators,
                min_samples_leaf = min_samples_leaf
            )
        )

try:
    featureSelectionProbe.main(dirpath,scriptpath,uttType,analyzedLabels,experimentName,probes)
    classifierEvaluation.main(dirpath,scriptpath,experimentName,probes)

except:
	telegramNotification.sendTelegram('Execution failed')
	traceback.print_exc()

telegramNotification.sendTelegram("End of execution")
