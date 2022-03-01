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

trainFiles = getFilesList('audible','train','all','all')
testFiles = getFilesList('audible','test','all','all')

duration = 0.0

trainDuration = getSubsetDuration(trainFiles, duration)
testDuration = getSubsetDuration(testFiles, duration)

print(f"Average length of data per session in training subset: {trainDuration/13} s")
print(f"Average length of data per session in testing subset: {testDuration/13} s")
print(f"Average length of data per session: {(trainDuration + testDuration)/13}")