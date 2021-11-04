DIR_PATH = "/mnt/ldisk/eder/ReSSInt/Pilot2/Session2"
#DIR_PATH = "/mnt/ldisk/eder/EMG-UKA-Trial-Corpus"
SCRIPT_PATH = "/home/aholab/eder/scripts/EMG-UKA-Analysis/EMG-UKA-Analysis"

REMOVE_CONTEXT_PHONEMES = True

FEATURE_NAMES = ["w","Pw","Pr","z","r"]
N_CHANNELS = 14
N_BATCHES = 1
STACKING_WIDTH = 15

N_FEATURES = len(FEATURE_NAMES)
ROW_SIZE = N_CHANNELS*N_FEATURES*(STACKING_WIDTH*2 + 1) + 1