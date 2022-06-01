from imghdr import tests
import classifierEvaluation
import featureSelectionProbe
from featureSelectionProbe import speakerDependent_SessionDependentClassification, speakerDependent_SessionIndependentClassification, speakerIndependent_SessionIndependentClassification
from featureSelectionProbe import Probe
import telegramNotification
import traceback

# Available reduction methods: 'SelectKBest', 'LDAReduction'
# Available scoreFunction (only for 'SelectKBest'): 'f_classif', 'mutual_into_classif'
# Available classificationMethods = 'GMMmodels', 'bagging', 'NN'

speakersAndSessions = {}
speakersAndSessions['002'] = ['001']
speakersAndSessions['002'] = ['001','003','101']
speakersAndSessions['004'] = ['001']
speakersAndSessions['006'] = ['001']
speakersAndSessions['008'] = [str(i).zfill(3) for i in range(1,9)]

allProbes = []

##############################################################

experimentName = 'transcodedContractiveEmg5_SPK_D-SS_D-Bagging-All'

referenceProbe = Probe(
    uttType='audible',
    analyzedLabels='all',
    reductionMethod = 'LDAReduction',
    n_features=32,
    classificationMethod = 'bagging',
    n_estimators=100,
    min_samples_leaf=50,
    analyzedData='emg'
)

probes = speakerDependent_SessionDependentClassification(speakersAndSessions,referenceProbe)

allProbes.append([experimentName,probes])

##############################################################

experimentName = 'contractiveAutoencoder5_SPK_D-SS_D-Bagging-All'

referenceProbe = Probe(
    uttType='audible',
    analyzedLabels='all',
    reductionMethod = 'LDAReduction',
    n_features=32,
    classificationMethod = 'bagging',
    n_estimators=100,
    min_samples_leaf=50,
    analyzedData='encodedFeatures'
)

probes = speakerDependent_SessionDependentClassification(speakersAndSessions,referenceProbe)

allProbes.append([experimentName,probes])

##############################################################

for experimentName, probes in allProbes:
    try:
        featureSelectionProbe.main(experimentName,probes)
        classifierEvaluation.main(experimentName,probes)

    except:
        telegramNotification.sendTelegram('Execution failed')
        traceback.print_exc()

    telegramNotification.sendTelegram("End of execution")
telegramNotification.sendTelegram("End of complete execution")