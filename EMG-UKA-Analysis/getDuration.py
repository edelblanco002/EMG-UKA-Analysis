from globalVars import DIR_PATH
from gatherDataIntoTable import getFilesList
from scipy.io import wavfile

def getSubsetDuration(utteranceFiles, duration):
    for utteranceFile in utteranceFiles:
        utteranceFile = utteranceFile.replace('emg_','')
        speaker, session, utt = utteranceFile.split('-')

        fs, signal = wavfile.read(f"{DIR_PATH}/audio/{speaker}/{session}/a_{speaker}_{session}_{utt}.wav")
        duration += len(signal)/fs

    return duration

sessionSet = {}
sessionSet['002'] = ['001','003','101']
sessionSet['004'] = ['001']
sessionSet['006'] = ['001']
sessionSet['008'] = [str(x).zfill(3) for x in range(1,9)]

for speaker in sessionSet.keys():
	for session in sessionSet[speaker]:
		duration = 0.0

		trainFiles = getFilesList('audible','train',speaker,session)
		testFiles = getFilesList('audible','test',speaker,session)
		trainDuration = getSubsetDuration(trainFiles, duration)
		testDuration = getSubsetDuration(testFiles, duration)

		print(f"{speaker}-{session}\tTraining subset: {trainDuration} s\tTesting subset: {testDuration} s\tTotal: {testDuration + trainDuration} s")
	print('\n')

trainFiles = getFilesList('audible','train','all','all')
testFiles = getFilesList('audible','test','all','all')

duration = 0.0

trainDuration = getSubsetDuration(trainFiles, duration)
testDuration = getSubsetDuration(testFiles, duration)


print(f"Total length of data per session in training subset: {trainDuration} s")
print(f"Total length of data per session in testing subset: {testDuration} s")
print(f"Total length of data per session: {(trainDuration + testDuration)}")

print(f"Average length of data per session in training subset: {trainDuration/13} s")
print(f"Average length of data per session in testing subset: {testDuration/13} s")
print(f"Average length of data per session: {(trainDuration + testDuration)/13}")
