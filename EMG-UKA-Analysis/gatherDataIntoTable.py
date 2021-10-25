from bar import printProgressBar
import numpy as np
import os
import tables

def buildTable(dirPath,utteranceFiles,rowSize,tableFileName='table'):
    # This function reads the data from all files in the utteranceFiles list
    # and writes all together into a HDF5 file

    tableFile = tables.open_file(f"{dirPath}/{tableFileName}.h5",mode='w', title="Features from audible utterances")
    atom = tables.Float32Atom()
    
    # Create the array_c to write into the HDF5 file. Each example is appended as a new row
    array_c = tableFile.create_earray(tableFile.root, 'data', atom, (0, rowSize))
    
    i = 0
    printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = 'Complete', length = 50)
    for utteranceFile in utteranceFiles:
        utteranceFile = utteranceFile.replace('emg_','')
        speaker, session, utt = utteranceFile.split('-')
        
        file = open(f"{dirPath}/features/{speaker}/{session}/e07_{speaker}_{session}_{utt}.npy",'rb')
        auxMat = np.load(file)
        file.close()
    
        for idx in range(np.shape(auxMat)[0]):
            array_c.append([auxMat[idx]])
    
        i += 1
        printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = f'{i}/{len(utteranceFiles)}', length = 50)
    
    del auxMat
    tableFile.close()

    return


def getFilesList(dirPath,uttType,subset='both',speaker='all',session='all'):
    # This function returns the list of all utterances of the given uttType (audible, whispered or silent) and the given subset (train, test or both). Also a specific speaker and session can be selected.

    utteranceFiles = []

    if subset == 'both':
        utteranceListFiles = [f"train.{uttType}",f"test.{uttType}"]
    elif subset == 'train':
        utteranceListFiles = [f"train.{uttType}"]
    else:
        utteranceListFiles = [f"test.{uttType}"]

    for listFile in utteranceListFiles:
        file = open(f"{dirPath}/Subsets/{listFile}",'r')
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

def removeTables(dirPath):
    # This function serves to remove every table at the end of execution
    dirFiles = os.listdir(dirPath)
    filteredFiles = [file for file in dirFiles if file.endswith(".h5")]
    for file in filteredFiles:
        filePath = os.path.join(dirPath,file)
        os.remove(filePath)


def main(dirPath,uttType,subset='both',speaker='all',session='all'):
    
    nChannels = 6
    nFeatures = 5
    stackingWidth = 15

    rowSize = 6*5*(stackingWidth*2+1) + 1
    
    files = getFilesList(dirPath,uttType,subset,speaker=speaker,session=session)

    filename = ""

    if speaker != 'all':
        filename += f"{speaker}_"

        if session != 'all':
            filename += f"{session}_"

    filename += f"{uttType}"

    if subset != 'both':
        filename += f"_{subset}_"

    filename += 'Table'

    buildTable(dirPath,files,rowSize,filename)


if __name__ == '__main__':
    main()
