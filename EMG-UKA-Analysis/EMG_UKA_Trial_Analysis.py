import classifierEvaluation
import featureSelectionProbe
from featureSelectionProbe import Probe
import telegramNotification
import traceback

uttType = 'audible' # 'audible', 'whispered' or 'silent'
analyzedLabels = 'Simple' # 'All', 'Simple', 'Transitions' or 'PilotStudy'

#gatherDataIntoTable.main(dirpath,uttType,'train')
#gatherDataIntoTable.main(dirpath,uttType,'test')

experimentName = 'Speaker-008_TrainSession-001_TestSession-005'


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

"""speakersAndSessions = [
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
            speaker=speaker,
            session=session,
            reductionMethod = 'LDAReduction',
            n_features = 32,
            classificationMethod = 'GMMmodels'
        )
    )"""

probes = []

probes.append(
    Probe(
        uttType='audible',
        analyzedLabels='simple',
        trainSpeaker='008',
        trainSession='001',
        testSpeaker='008',
        testSession='005',
        reductionMethod = 'LDAReduction',
        n_features = 32,
        classificationMethod = 'GMMmodels'
    )
)

try:
    featureSelectionProbe.main(experimentName,probes)
    classifierEvaluation.main(experimentName,probes)

except:
	telegramNotification.sendTelegram('Execution failed')
	traceback.print_exc()

telegramNotification.sendTelegram("End of execution")
