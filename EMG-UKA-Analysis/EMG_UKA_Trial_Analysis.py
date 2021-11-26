import classifierEvaluation
import featureSelectionProbe
from featureSelectionProbe import Probe
import telegramNotification
import traceback

experimentName = 'Session_Dependent_SelectKBest_100_Bagging_100_50'

# Available reduction methods: 'SelectKBest', 'LDAReduction'
# Available scoreFunction (only for 'SelectKBest'): 'f_classif', 'mutual_into_classif'
# Available classificationMethods = 'GMMmodels', 'bagging'

speakersAndSessions = [
    ('002','001'),
    ('002','003'),
    ('002','101'),
    ('004','001'),
    ('006','001'),
    ('008','001'),
    ('008','002'),
    ('008','003'),
    ('008','004'),
    ('008','005'),
    ('008','006'),
    ('008','007'),
    ('008','008')
]

probes = []

for speaker, session in speakersAndSessions:
    probes.append(
        Probe(
            uttType='audible',
            analyzedLabels='simple',
            trainSession='all',
            trainSpeaker='all',
            testSpeaker=speaker,
            testSession=session,
            reductionMethod = 'SelectKBest',
            scoreFunction='f_classif',
            n_features = 100,
            classificationMethod = 'bagging',
            n_estimators=100,
            min_samples_leaf=50
        )
    )

try:
    featureSelectionProbe.main(experimentName,probes)
    classifierEvaluation.main(experimentName,probes)

except:
	telegramNotification.sendTelegram('Execution failed')
	traceback.print_exc()

telegramNotification.sendTelegram("End of execution")
