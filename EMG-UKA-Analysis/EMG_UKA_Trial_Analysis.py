from imghdr import tests
import classifierEvaluation
import featureSelectionProbe
from featureSelectionProbe import Probe
import telegramNotification
import traceback

experimentName = 'Session-Ind-Speaker-Dep_SelectKBest_100_Bagging_100_50'

# Available reduction methods: 'SelectKBest', 'LDAReduction'
# Available scoreFunction (only for 'SelectKBest'): 'f_classif', 'mutual_into_classif'
# Available classificationMethods = 'GMMmodels', 'bagging'

""" speakersAndSessions = [
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
            trainSession=all,
            testSpeaker=speaker,
            testSession=all,
            reductionMethod = 'LDAReduction',
            n_features = 32,
            classificationMethod = 'GMMmodels',
            analyzeMFCCs=True
        )
    )
 """

set = dict()
set['002'] = ['001','003','101']
set['008'] = [str(i).zfill(3) for i in range(1,9)]

probes = []

for speaker in set.keys():
    speaker = speaker
    sessions = set[speaker]

    for currentSession in sessions:
        trainSessions = ()
        for session in sessions:
            if session != currentSession:
                trainSessions += ( session + '-all',)
        testSession = currentSession + '-all'
        print(f"Speaker: {speaker}; Train sessions: {trainSessions}; Test session: {testSession}")
        probes.append(
            Probe(
                uttType='audible',
                analyzedLabels='simple',
                trainSpeaker=speaker,
                testSpeaker=speaker,
                trainSession=trainSessions,
                testSession=testSession,
                reductionMethod = 'SelectKBest',
                scoreFunction='f_classif',
                n_features = 100,
                classificationMethod = 'bagging',
                min_samples_leaf=50,
                n_estimators=100,
                analyzeMFCCs=False
            )
        )

try:
    featureSelectionProbe.main(experimentName,probes)
    classifierEvaluation.main(experimentName,probes)

except:
	telegramNotification.sendTelegram('Execution failed')
	traceback.print_exc()

telegramNotification.sendTelegram("End of execution")
