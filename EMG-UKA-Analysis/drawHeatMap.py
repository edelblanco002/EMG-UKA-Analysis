import math
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns

def createDataFrame(scores, channelNames, featureNames, frameNames):
    # Creates a new dataframe where every single score has an entry

    colNames = ['Channel','Frame','Feature','Score']
    
    df = pd.DataFrame(columns=colNames)
    
    i = 0
    for channelName in channelNames:
        for frameName in frameNames:
            for featureName in featureNames:
                df.loc[i] = [channelName,frameName,featureName,scores[i]]
                i += 1

    return df


def getChannelNames(channelN):
    # Creates the name of the channels like 'Channel 1', 'Channel 2'...

    channelNames = []
    for i in range(channelN):
        channelNames.append(f"Channel {i+1}")
    return channelNames

def getFrameNames(stackingWidth):
    # Creates the name for the frames like '-15','-14',...,'-1','0','1',...,'14','15'

    frameNames = []
    
    for i in range(-stackingWidth,stackingWidth+1):
        frameNames.append(str(i))

    return frameNames


def main(dirpath = 'C:/Users/Eder/Downloads/EMG-UKA-Trial-Corpus',fileBase='audibleAll',channelsPerPlot=6):
    
    # Read the scores from matrices
    scoresFClass = np.load(dirpath + f'/scoresFClass{fileBase}.npy')
    scoresMutual = np.load(dirpath + f'/scoresMutual{fileBase}.npy')
    
    # Sum all scores for each feature
    scoresFClass = np.sum(scoresFClass,axis=0)
    scoresMutual = np.sum(scoresMutual,axis=0)
    
    # Find the max and the min values in order to anchor the different subplots to the same range
    maxScoreFClass = max(scoresFClass)
    minScoreFClass = min(scoresFClass)
    maxScoreMutual = max(scoresMutual)
    minScoreMutual = min(scoresMutual)
    
    stackingWidth = 15
    channelN = 6
    featureNames = ['w','Pw','Pr','z','r']
    channelNames = getChannelNames(channelN)
    
    frameNames = getFrameNames(stackingWidth)
    
    # Create DataFrames with one entry for each single score
    dfFClass = createDataFrame(scoresFClass, channelNames, featureNames, frameNames)
    dfMutual = createDataFrame(scoresMutual, channelNames, featureNames, frameNames)    

    # How many plots are needed to plot every channel? channelsPerPlot = 6 by default
    nPlots = math.ceil(channelN/channelsPerPlot)
    plotsFClass = [i for i in range(1,nPlots+1)] # Create as many plots as needed. First plots are for f_class
    plotsMutual = [i + nPlots for i in plotsFClass] # Last plots are for mutual

    # For each figure (scores of f_classif), create a 'channelsPerPlot' number of subplots in a column
    # and one vertical plot on the right, with the width relation of 1:40 for the vertical plot and 39:40 for the horizontal plots
    axisFClass = []
    colorBarAxisFClass = []

    for i in plotsFClass: # For each plot corresponding to FClass
        plt.figure(i)
        if i == plotFClass[-1]: # In the last plot, draw just the resting subplots
            channelsInPlot = channelN%channelsPerPlot
        else:
            channelsInPlot = channelsPerPlot

        gsFClass = gridspec.GridSpec(channelsInPlot,40)

        for j in range(channelsInPlot):
            axisFClass.append(plt.subplot(gsFClass[j,0:39]))
        colorBarAxisFClass.append(plt.subplot(gsFClass[:,39]))

    # The same for the second figure (scores of mutual_info_classif)
    axisMutual = []
    colorBarAxisMutual = []

    for i in plotsMutual:  # For each plot corresponding to mutual_info_classif
        plt.figure(i)
        if i == plotsMutual[-1]: # If it's the last plot, draw just the resting subplots
            channelsInPlot = channelN%channelsPerPlot
        else:
            channelsInPlot = channelsPerPlot

        gsMutual = gridspec.GridSpec(channelsInPlot,40)

        for j in range(channelsInPlot):
            axisMutual.append(plt.subplot(gsMutual[j,0:39]))
        colorBarAxisMutual.append(plt.subplot(gsMutual[:,39]))

    # Draw one subplot for each channel
    for i in range(channelN):
        # From the dataset, find the entries corresponding to the actual channel
        dfFClass_ = dfFClass[dfFClass['Channel'] == channelNames[i]]
        dfMutual_ = dfMutual[dfMutual['Channel'] == channelNames[i]]

        # Create a rectangular dataframe with the features as rows, the frames as columns and the scores as the values
        dfFClass_ = dfFClass_.pivot(index="Feature",columns="Frame",values="Score")
        dfFClass_ = dfFClass_[frameNames]
        dfMutual_ = dfMutual_.pivot(index="Feature",columns="Frame",values="Score")
        dfMutual_ = dfMutual_[frameNames]

        # Draw the heatmap for the channel, anchoring the scale to the absolute max and min score, and drawing the color bar in its corresponding vertical plot
        # There is a color bar axis in each plot. The selected color bar axis is the corresponding to the actual plot
        sns.heatmap(dfFClass_,ax=axisFClass[i], linewidths=.2, vmax=maxScoreFClass, vmin=minScoreFClass, yticklabels=True, xticklabels=True, cbar_ax=colorBarAxisFClass[math.floor(i/channelsPerPlot)])
        sns.heatmap(dfMutual_,ax=axisMutual[i], linewidths=.2, vmax=maxScoreMutual, vmin=minScoreMutual, yticklabels=True, xticklabels=True, cbar_ax=colorBarAxisMutual[math.floor(i/channelsPerPlot)])

        axisFClass[i].set_title('')
        axisMutual[i].set_title('')

        # Left the xlabels just for the last subplot (on the bottom)
        if i%channelsPerPlot == channelsPerPlot - 1: # If it's the last subplot of the plot
            axisFClass[i].set_xticks([])
            axisFClass[i].set_xlabel('')
            axisMutual[i].set_xticks([])
            axisMutual[i].set_xlabel('')

        axisFClass[i].set_ylabel(channelNames[i])
        axisMutual[i].set_ylabel(channelNames[i])

    # Set the title for the figure 1
    plt.figure(1)
    plt.suptitle("Feature scores with f_classif")
    # And for the figure 2
    plt.figure(2)
    plt.suptitle("Feature scores with mutual_info_classif")

    plt.show()

    return

if __name__ =='__main__':
    main()
