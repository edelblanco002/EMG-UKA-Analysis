from bar import printProgressBar
import numpy as np
import os
import tables

def buildTable(dirpath,utteranceFiles,rowSize,tableFileName='table'):
    # This function reads the data from all files in the utteranceFiles list
    # and writes all together into a HDF5 file

    tableFile = tables.open_file(f"{dirpath}/{tableFileName}.h5",mode='w', title="Features from audible utterances")
    atom = tables.Float32Atom()
    
    # Create the array_c to write into the HDF5 file. Each example is appended as a new row
    array_c = tableFile.create_earray(tableFile.root, 'data', atom, (0, rowSize))
    
    i = 0
    printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = 'Complete', length = 50)
    for utteranceFile in utteranceFiles:
        utteranceFile = utteranceFile.replace('emg_','')
        speaker, session, utt = utteranceFile.split('-')
        
        file = open(f"{dirpath}/features/{speaker}/{session}/e07_{speaker}_{session}_{utt}.npy",'rb')
        auxMat = np.load(file)
        file.close()
    
        for idx in range(np.shape(auxMat)[0]):
            array_c.append([auxMat[idx]])
    
        i += 1
        printProgressBar(i, len(utteranceFiles), prefix = 'Progress:', suffix = f'{i}/{len(utteranceFiles)}', length = 50)
    
    del auxMat
    tableFile.close()

    return


def getFilesList(dirpath,uttType,subset='both'):
    # This function returns the list of all utterances of the given uttType (audible, whispered or silent) and the given subset (train, test or both)

    utteranceFiles = []

    if subset == 'both':
        utteranceListFiles = [f"train.{uttType}",f"test.{uttType}"]
    elif subset == 'train':
        utteranceListFiles = [f"train.{uttType}"]
    else:
        utteranceListFiles = [f"test.{uttType}"]

    for listFile in utteranceListFiles:
        file = open(f"{dirpath}/Subsets/{listFile}",'r')
        lines = file.readlines()
        file.close()
    
        for line in lines:
            if line.strip():
                line = line.replace('\n','')
                for element in line.split(' ')[1:]:
                    if element != '':
                        utteranceFiles.append(element)
    return utteranceFiles


def main(dirpath='C:/Users/Eder/Downloads/EMG-UKA-Trial-Corpus',uttType = 'audible',subset='both'):
    
    nChannels = 6
    nFeatures = 5
    stackingWidth = 15

    rowSize = 6*5*(stackingWidth*2+1) + 1
    
    files = getFilesList(dirpath,uttType,subset)

    if subset == 'both':
        filename = f"{uttType}Table"
    else:
        filename = f"{uttType}_{subset}_Table"

    buildTable(dirpath,files,rowSize,filename)


if __name__ == '__main__':
    main()
