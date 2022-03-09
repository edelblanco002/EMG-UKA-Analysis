from imghdr import tests
import classifierEvaluation
import featureSelectionProbe
from featureSelectionProbe import Probe
import telegramNotification
import traceback

experimentName = 'EMG-Hilbert_Speaker-Session-Dep_Paper-configuration'

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
            speaker=speaker,
            session=session,
            reductionMethod = 'LDAReduction',
            n_features = 32,
            classificationMethod = 'GMMmodels',            
            analyzedData='emg-hilbert'
        )
    )


"""set = dict()
set['002'] = ['001','003','101']
set['004'] = ['001']
set['006'] = ['001']
set['008'] = [str(i).zfill(3) for i in range(1,9)]

probes = []"""

"""for currentSpeaker in set.keys():
    sessions = set[currentSpeaker]

    for currentSession in sessions:
        trainSessions = ()
        for session in sessions:
            if session != currentSession:
                trainSessions += (session + '-all', )
        
        probes.append(
            Probe(
                uttType='audible',
                analyzedLabels='simple',
                trainSpeaker=currentSpeaker,
                testSpeaker=currentSpeaker,
                trainSession=trainSessions,
                testSession=currentSession,
                reductionMethod = 'LDAReduction',
                #scoreFunction='f_classif',
                n_features = 12,
                classificationMethod = 'GMMmodels',
                #min_samples_leaf=50,
                #n_estimators=100,
                analyzeMFCCs=True
            )
        )"""


"""for currentSpeaker in set.keys():
    sessions = set[currentSpeaker]

    trainSpeakers = ()

    for speaker in set.keys():
        if speaker != currentSpeaker:
            trainSpeakers += (speaker + '-all',)

    for session in sessions:
        testSession = session + '-all'
        probes.append(
            Probe(
                uttType='audible',
                analyzedLabels='simple',
                trainSpeaker=trainSpeakers,
                testSpeaker=currentSpeaker,
                trainSession='all',
                testSession=testSession,
                reductionMethod = 'LDAReduction',
                #scoreFunction='f_classif',
                n_features = 12,
                classificationMethod = 'GMMmodels',
                #min_samples_leaf=50,
                #n_estimators=100,
                analyzeMFCCs=True
            )
        )"""

try:
    featureSelectionProbe.main(experimentName,probes)
    classifierEvaluation.main(experimentName,probes)

except:
	telegramNotification.sendTelegram('Execution failed')
	traceback.print_exc()

telegramNotification.sendTelegram("End of execution")
