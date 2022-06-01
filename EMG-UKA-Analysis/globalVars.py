import math

#DIR_PATH = "/mnt/ldisk/eder/ReSSInt/Pilot2/Session2"
DIR_PATH = "/mnt/ldisk/eder/EMG-UKA-Trial-Corpus"
SCRIPT_PATH = "/home/aholab/eder/scripts/EMG-UKA-Analysis/EMG-UKA-Analysis"

#####################################
#   Function to get phoneme dict    #
#####################################

def getPhoneDict():
    # This function loads a dictionary that links each label number with its corresponding phoneme.

    phoneDict = {}
    
    # Reads the map that contains the relation between each phoneme and its number
    file = open(f"{SCRIPT_PATH}/phoneMap",'r')
    lines = file.readlines()
    file.close()
    
    # Builds the dictionary using the number as key and the phoneme as value
    for line in lines:
        if line.strip():
            line.replace('\n','')
            columns = line.split(' ')
            phone = columns[0]                     
            key = int(columns[1])
            phoneDict[key] = phone

    return phoneDict

def getFirstTransitionPhoneme(phoneDict):
    # This function returns the number label of the first transition phoneme (the first phoneme that contains a '-' or '+' sign)

    # In the phoneme dictionary, the label 'no_label' and the simple phonemes are at the top of the list
    # The transition phonemes contains a '+' or '-' mark, so the first transition phoneme is detected by looking if it contains any of those symbols
    # The last considered phoneme is the one before the first transition phoneme
    for key in sorted(phoneDict.keys()):
        if ('+' in phoneDict[key]) or ('-' in phoneDict[key]):
            lastKey = key
            break

    return lastKey

################################################

KEEP_ALL_FRAMES_IN_TEST = True # If True, all frames from the testing subset (unless silences) will be used for testing regardless of if classifier has been trained only with simple or transition labels
REMOVE_CONTEXT_PHONEMES = False # Related to Pilot Study. Don't touch it
REMOVE_SILENCES = False # Remove the frames labeled as silences 
REMOVE_SILENCES_AT_ENDS = True # Remove the frames labeled as silences only at the beginning and the end of the utterances

if REMOVE_SILENCES_AT_ENDS and REMOVE_SILENCES: # There is not point on setting this two variables to on at the same time
    raise ValueError

CORPUS = "EMG-UKA" # "Pilot Study" or "EMG-UKA"

# Features:
# - Mw: Low Frequency Mean
# - Pw: Low Frequency Power
# - Mr: High Frequency Rectified Mean
# - Pr: High Frequency Rectified Power
# - zp: High Frequency Zero Crossing Rate
# - H: Hilbert Transform

# In paper: ['Wm','Pw','Pr','zp','Mr']
FEATURE_NAMES = ['Mw','Pw','Pr','zp','Mr']

PHONE_DICT = getPhoneDict() # Dictionary that takes the number label as key and returns the string label (e.g. PHONEDICT['0'] = 'A')
FIRST_TRANSITION_PHONEME = getFirstTransitionPhoneme(PHONE_DICT) # First transition label. Lower labels are simple lables, and higher labels are transition labels

N_FEATURES = len(FEATURE_NAMES) # Number of calculated features
N_CHANNELS = 6 # Number of channels (whitout the syncronization channel)
N_BATCHES = 1
STACKING_WIDTH = 15
STACKING_MODE = 'symmetric' # 'symmetric' or 'backwards' 

if STACKING_MODE == 'symmetric':
    ROW_SIZE = N_CHANNELS*N_FEATURES*(STACKING_WIDTH*2 + 1) + 1
elif STACKING_MODE == 'backwards':
    ROW_SIZE = N_CHANNELS*N_FEATURES*(STACKING_WIDTH + 1) + 1

#########################
#   Parameters for MFCC #
#########################

AUDIO_FRAME_SIZE = 0.016
AUDIO_FRAME_SHIFT = 0.01
N_FILTERS = 30
N_COEF = 13
if STACKING_MODE == 'symmetric':
    MFCC_ROW_SIZE = N_COEF*(STACKING_WIDTH*2 + 1) + 1
elif STACKING_MODE == 'backwards':
    MFCC_ROW_SIZE = N_COEF*(STACKING_WIDTH + 1) + 1
########################################
#   Parameters for Hilbert transform   #
########################################
FRAME_SIZE = 0.025 # Size of the frame used to calculate the EMG features, also the Hilbert transform
FRAME_SHIFT = 0.005
FS = 600
HILBERT_FS = 200 # The sampling frequency at which the Hilbert transform is downsampled after being calculated
HILBERT_ROW_SIZE = math.floor(HILBERT_FS*FRAME_SIZE)*(2*STACKING_WIDTH + 1) + 1 # The size of the Hilbert transform corresponding to the central and the stacked frames
