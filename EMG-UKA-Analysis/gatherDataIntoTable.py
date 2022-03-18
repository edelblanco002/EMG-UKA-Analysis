from bar import printProgressBar
from globalVars import CORPUS, DIR_PATH, N_CHANNELS, MFCC_ROW_SIZE, ROW_SIZE
from globalVars import HILBERT_ROW_SIZE, HILBERT_FS, FRAME_SHIFT
import math
import numpy as np
import os
import tables

def buildEMGHilbertTable(utteranceFiles,tableFileName='table',uttType='audible'):
    # This function reads the data from all files in the utteranceFiles list
    # and writes all together into a HDF5 file

    tableFile = tables.open_file(f"{DIR_PATH}/{tableFileName}.h5",mode='w', title="Features from audible utterances")
    atom = tables.Float32Atom()
    
    # Create the array_c to write into the HDF5 file. Each example is appended as a new row
    array_c = tableFile.create_earray(tableFile.root, 'data', atom, (0, ROW_SIZE + N_CHANNELS*(HILBERT_ROW_SIZE - 1))) # -1 because labels are already considered in the features matrix
    
    i = 0
    printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = f'{i}/{len(utteranceFiles)}', length = 50)
    for utteranceFile in utteranceFiles:
        utteranceFile = utteranceFile.replace('emg_','')
        speaker, session, utt = utteranceFile.split('-')

        if CORPUS == "EMG-UKA":
            featuresFile = open(f"{DIR_PATH}/features/{speaker}/{session}/e{str(N_CHANNELS+1).zfill(2)}_{speaker}_{session}_{utt}.npy",'rb')
            hilbertFile = open(f"{DIR_PATH}/hilbert/{speaker}/{session}/e{str(N_CHANNELS+1).zfill(2)}_{speaker}_{session}_{utt}.npy",'rb')

        elif CORPUS == "Pilot Study":
            round = session[-1]
            featuresFile = open(f"{DIR_PATH}/features/{speaker}/{session}/emg_{str(N_CHANNELS+1).zfill(2)}ch_{speaker}_{uttType}{round}_{utt}.npy",'rb')
            hilbertFile = open(f"{DIR_PATH}/hilbert/{speaker}/{session}/emg_{str(N_CHANNELS+1).zfill(2)}ch_{speaker}_{uttType}{round}_{utt}.npy",'rb')
        else:
            print("The CORPUS parameter is wrong defined in the globalVars.py file")
            raise ValueError

        auxFeaturesMat = np.load(featuresFile)
        auxHilbertMat = np.load(hilbertFile)
        featuresFile.close()
        hilbertFile.close()
    
        # Remove labels from hilbert matrice
        auxHilbertMat = auxHilbertMat[:,:,1:]
        # Stack the hilberd transformed signals corresponding to each channels [ch1, ch2, ch3...]
        auxHilbertMat = auxHilbertMat.reshape(-1,N_CHANNELS*(HILBERT_ROW_SIZE - 1))
        # If the step took for the window to extract the Hilbert transform was less than 1, it has been rounded to 1
        # This means that the number of frames obtained has been less than the needed
        # Frames should be duplicated to match the number of frames used in features
        if (HILBERT_FS*FRAME_SHIFT) < 1:
            auxHilbertMat = np.repeat(auxHilbertMat,math.ceil(1/(HILBERT_FS*FRAME_SHIFT)),axis=0)

        nFramesHilbert = np.shape(auxHilbertMat)[0]
        nFramesFeatures = np.shape(auxFeaturesMat)[0]

        # Maybe, because of difference in sampling rate, there could be a frame more in one matrix than in the other
        # If the difference is greather than a frame, probably there is an error 
        if abs(nFramesHilbert - nFramesFeatures) > 1:
            frameDifference = abs(nFramesHilbert - nFramesFeatures)
            print(f"There is a difference of {frameDifference} between hilbert and emg features in {utteranceFile}.")
            raise Exception

        # If, as expected, the difference is of one frame, remove the end frame in the largest matrix
        if nFramesFeatures > nFramesHilbert:
            auxFeaturesMat = np.delete(auxFeaturesMat,-1,axis=0)
            nFramesFeatures = np.shape(auxFeaturesMat)[0]

        elif nFramesHilbert > nFramesFeatures:
            auxHilbertMat = np.delete(auxHilbertMat,-1,axis=0)
            nFramesHilbert = np.shape(auxHilbertMat)[0]

        # Join features and Hilbert into one single matrix
        auxMat = np.zeros((nFramesFeatures,ROW_SIZE + N_CHANNELS*(HILBERT_ROW_SIZE - 1)))
        auxMat[:,0:ROW_SIZE] = auxFeaturesMat[:]
        auxMat[:,ROW_SIZE:] = auxHilbertMat[:]


        for idx in range(np.shape(auxMat)[0]):
            try:
                array_c.append([auxMat[idx]])
            except:
                tableFile.close()
                raise Exception
    
        i += 1
        printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = f'{i}/{len(utteranceFiles)}', length = 50)
    
    del auxMat
    tableFile.close()

    return

def buildHilbertTable(utteranceFiles,tableFileName='table',uttType='audible'):
    # This function reads the data from all files in the utteranceFiles list
    # and writes all together into a HDF5 file

    tableFile = tables.open_file(f"{DIR_PATH}/{tableFileName}.h5",mode='w', title="Features from audible utterances")
    atom = tables.Float32Atom()
    
    # Create the array_c to write into the HDF5 file. Each example is appended as a new row
    array_c = tableFile.create_earray(tableFile.root, 'data', atom, (0, N_CHANNELS, HILBERT_ROW_SIZE))
    
    i = 0
    printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = f'{i}/{len(utteranceFiles)}', length = 50)
    for utteranceFile in utteranceFiles:
        utteranceFile = utteranceFile.replace('emg_','')
        speaker, session, utt = utteranceFile.split('-')

        if CORPUS == "EMG-UKA":
            file = open(f"{DIR_PATH}/hilbert/{speaker}/{session}/e{str(N_CHANNELS+1).zfill(2)}_{speaker}_{session}_{utt}.npy",'rb')

        elif CORPUS == "Pilot Study":
            round = session[-1]
            file = open(f"{DIR_PATH}/hilbert/{speaker}/{session}/emg_{str(N_CHANNELS+1).zfill(2)}ch_{speaker}_{uttType}{round}_{utt}.npy",'rb')
        else:
            print("The CORPUS parameter is wrong defined in the globalVars.py file")
            raise ValueError

        auxMat = np.load(file)
        file.close()
    
        for idx in range(np.shape(auxMat)[0]):
            try:
                array_c.append([auxMat[idx]])
            except:
                tableFile.close()
                raise Exception
    
        i += 1
        printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = f'{i}/{len(utteranceFiles)}', length = 50)
    
    del auxMat
    tableFile.close()

    return

def buildMFCCTable(utteranceFiles,tableFileName='table',uttType='audible'):
    # This function reads the data from all files in the utteranceFiles list
    # and writes all together into a HDF5 file

    tableFile = tables.open_file(f"{DIR_PATH}/{tableFileName}.h5",mode='w', title="Features from audible utterances")
    atom = tables.Float32Atom()
    
    # Create the array_c to write into the HDF5 file. Each example is appended as a new row
    array_c = tableFile.create_earray(tableFile.root, 'data', atom, (0, MFCC_ROW_SIZE))
    
    i = 0
    printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = f'{i}/{len(utteranceFiles)}', length = 50)
    for utteranceFile in utteranceFiles:
        utteranceFile = utteranceFile.replace('emg_','')
        speaker, session, utt = utteranceFile.split('-')

        if CORPUS == "EMG-UKA":
            file = open(f"{DIR_PATH}/mfccs/{speaker}/{session}/a_{speaker}_{session}_{utt}.npy",'rb')

        elif CORPUS == "Pilot Study":
            round = session[-1]
            file = open(f"{DIR_PATH}/mfccs/{speaker}/{session}/a_{speaker}_{uttType}{round}_{utt}.npy",'rb')
        else:
            print("The CORPUS parameter is wrong defined in the globalVars.py file")
            raise ValueError

        auxMat = np.load(file)
        file.close()
    
        for idx in range(np.shape(auxMat)[0]):
            array_c.append([auxMat[idx]])
    
        i += 1
        printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = f'{i}/{len(utteranceFiles)}', length = 50)
    
    del auxMat
    tableFile.close()

    return

def buildTable(utteranceFiles,tableFileName='table',uttType='audible'):
    # This function reads the data from all files in the utteranceFiles list
    # and writes all together into a HDF5 file

    tableFile = tables.open_file(f"{DIR_PATH}/{tableFileName}.h5",mode='w', title="Features from audible utterances")
    atom = tables.Float32Atom()
    
    # Create the array_c to write into the HDF5 file. Each example is appended as a new row
    array_c = tableFile.create_earray(tableFile.root, 'data', atom, (0, 1 + ROW_SIZE))
    
    i = 0
    printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = f'{i}/{len(utteranceFiles)}', length = 50)
    for utteranceFile in utteranceFiles:
        utteranceFile = utteranceFile.replace('emg_','')
        speaker, session, utt = utteranceFile.split('-')

        if CORPUS == "EMG-UKA":
            file = open(f"{DIR_PATH}/features/{speaker}/{session}/e{str(N_CHANNELS+1).zfill(2)}_{speaker}_{session}_{utt}.npy",'rb')

        elif CORPUS == "Pilot Study":
            round = session[-1]
            file = open(f"{DIR_PATH}/features/{speaker}/{session}/emg_{str(N_CHANNELS+1).zfill(2)}ch_{speaker}_{uttType}{round}_{utt}.npy",'rb')
        else:
            print("The CORPUS parameter is wrong defined in the globalVars.py file")
            raise ValueError

        auxMat = np.load(file)
        file.close()

        # Create a column with a number to identify to with utterances belong the set of frames
        # And append it as the first column of the matrix
        nFrames = np.shape(auxMat)[0]
        fileLabel = np.ones((nFrames,1))*i
        auxMat = np.hstack((fileLabel,auxMat))

        for idx in range(np.shape(auxMat)[0]):
            array_c.append([auxMat[idx]])
    
        i += 1
        printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = f'{i}/{len(utteranceFiles)}', length = 50)
    
    del auxMat
    tableFile.close()

    return


def getFilesList(uttType,subset='both',speaker='all',session='all'):
    # This function returns the list of all utterances of the given uttType (audible, whispered or silent) and the given subset (train, test or both). Also a specific speaker and session can be selected.

    utteranceFiles = []

    if subset == 'both':
        utteranceListFiles = [f"train.{uttType}",f"test.{uttType}"]
    elif subset == 'train':
        utteranceListFiles = [f"train.{uttType}"]
    else:
        utteranceListFiles = [f"test.{uttType}"]

    for listFile in utteranceListFiles:
        file = open(f"{DIR_PATH}/Subsets/{listFile}",'r')
        lines = file.readlines()
        file.close()

        if speaker == 'all': # Extract files from every session of every users
            for line in lines:
                if line.strip():
                    line = line.replace('\n','')
                    for element in line.split(' ')[1:]:
                        if element != '':
                            utteranceFiles.append(element)

        else: # Extract the files from an specific user
            for line in lines:
                if line.strip():
                    line = line.replace('\n','')
                    id = line.split(':')[0]
                    id = id.replace('emg_','')
                    idSpeaker, idSession = id.split('-')
                    if speaker == idSpeaker and (session == 'all' or session == idSession):
                    # Only enters here if the line corresponds to the desired user. Also, only enters here if every session of the user are wanted or, otherwise, the line corresponds to the desired session
                        for element in line.split(' ')[1:]:
                            if element != '':
                                utteranceFiles.append(element)

    return utteranceFiles

def removeTables():
    # This function serves to remove every table at the end of execution
    dirFiles = os.listdir(DIR_PATH)
    filteredFiles = [file for file in dirFiles if file.endswith(".h5")]
    for file in filteredFiles:
        filePath = os.path.join(DIR_PATH,file)
        os.remove(filePath)


def main(uttType,subset='both',speaker='all',session='all',analyzedData='emg'):        

    files = getFilesList(uttType,subset,speaker=speaker,session=session)

    filename = ""

    if speaker != 'all':
        filename += f"{speaker}_"

        if session != 'all':
            filename += f"{session}_"

    filename += f"{uttType}"

    if subset != 'both':
        filename += f"_{subset}_"

    filename += 'Table'

    print(f"Building {filename}...")

    if analyzedData == 'MFCCs':
        buildMFCCTable(files,filename,uttType)
    elif analyzedData == 'emg':
        buildTable(files,filename,uttType)
    elif analyzedData == 'hilbert':
        buildHilbertTable(files,filename,uttType)
    elif analyzedData == 'emg-hilbert':
        buildEMGHilbertTable(files,filename,uttType)

if __name__ == '__main__':
    main()
