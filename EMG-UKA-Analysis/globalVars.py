#DIR_PATH = "/mnt/ldisk/eder/ReSSInt/Pilot2/Session2"
DIR_PATH = "/mnt/ldisk/eder/EMG-UKA-Trial-Corpus"
SCRIPT_PATH = "/home/aholab/eder/scripts/EMG-UKA-Analysis/EMG-UKA-Analysis"

REMOVE_CONTEXT_PHONEMES = False
REMOVE_SILENCES = True

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

N_FEATURES = len(FEATURE_NAMES) # Number of calculated features
N_CHANNELS = 6 # Number of channels (whitout the syncronization channel)
N_BATCHES = 1
STACKING_WIDTH = 10

ROW_SIZE = N_CHANNELS*N_FEATURES*(STACKING_WIDTH*2 + 1) + 1

#########################
#   Parameters for MFCC #
#########################

AUDIO_FRAME_SIZE = 0.016
AUDIO_FRAME_SHIFT = 0.01
N_FILTERS = 30
N_COEF = 13
MFCC_ROW_SIZE = N_COEF*(STACKING_WIDTH*2 + 1) + 1