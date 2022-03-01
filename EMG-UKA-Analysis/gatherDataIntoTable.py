from bar import printProgressBar
from globalVars import CORPUS, DIR_PATH, N_CHANNELS, MFCC_ROW_SIZE, ROW_SIZE
import numpy as np
import os
import tables

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
            print("The CORPUS parameter is bat defined in the globalVars.py file")
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
    array_c = tableFile.create_earray(tableFile.root, 'data', atom, (0, ROW_SIZE))
    
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
            print("The CORPUS parameter is bat defined in the globalVars.py file")
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


def main(uttType,subset='both',speaker='all',session='all',analyzeMFFCs=False):        

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

    if analyzeMFFCs:
        buildMFCCTable(files,filename,uttType)
    else:
        buildTable(files,filename,uttType)


if __name__ == '__main__':
    main()
