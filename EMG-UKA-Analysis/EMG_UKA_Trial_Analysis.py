import classifierEvaluation
import featureSelectionProbe
from featureSelectionProbe import Probe
import telegramNotification
import traceback

experimentName = 'MFCCs_Paper'

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
            trainSpeaker=speaker,
            trainSession=session,
            testSpeaker=speaker,
            testSession=session,
            reductionMethod = 'LDAReduction',
            n_features = 32,
            classificationMethod = 'GMMmodels',
            analyzeMFCCs=True
        )
    )

try:
    featureSelectionProbe.main(experimentName,probes)
    classifierEvaluation.main(experimentName,probes)

except:
	telegramNotification.sendTelegram('Execution failed')
	traceback.print_exc()

telegramNotification.sendTelegram("End of execution")
