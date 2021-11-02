from globalVars import DIR_PATH, FEATURE_NAMES, N_CHANNELS, STACKING_WIDTH
import math
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
import seaborn as sns

def createDataFrame(scores, channelNames, frameNames):
    # Creates a new dataframe where every single score has an entry

    colNames = ['Channel','Frame','Feature','Score']
    
    df = pd.DataFrame(columns=colNames)
    
    i = 0
    for channelName in channelNames:
        for frameName in frameNames:
            for featureName in FEATURE_NAMES:
                df.loc[i] = [channelName,frameName,featureName,scores[i]]
                i += 1

    return df


def getChannelNames():
    # Creates the name of the channels like 'Channel 1', 'Channel 2'...

    channelNames = []
    for i in range(N_CHANNELS):
        channelNames.append(f"Channel {i+1}")
    return channelNames

def getFrameNames():
    # Creates the name for the frames like '-15','-14',...,'-1','0','1',...,'14','15'

    frameNames = []
    
    for i in range(-STACKING_WIDTH,STACKING_WIDTH+1):
        frameNames.append(str(i))

    return frameNames


def main(fileBase='audibleall',channelsPerPlot=6):
    
    # Read the scores from matrices
    scoresFClass = np.load(DIR_PATH + f'/scoresFClass{fileBase}.npy')
    scoresMutual = np.load(DIR_PATH + f'/scoresMutual{fileBase}.npy')
    
    # Sum all scores for each feature
    scoresFClass = np.sum(scoresFClass,axis=0)
    scoresMutual = np.sum(scoresMutual,axis=0)
    
    # Find the max and the min values in order to anchor the different subplots to the same range
    maxScoreFClass = max(scoresFClass)
    minScoreFClass = min(scoresFClass)
    maxScoreMutual = max(scoresMutual)
    minScoreMutual = min(scoresMutual)
    
    channelNames = getChannelNames()
    
    frameNames = getFrameNames()
    
    # Create DataFrames with one entry for each single score
    dfFClass = createDataFrame(scoresFClass, channelNames, frameNames)
    dfMutual = createDataFrame(scoresMutual, channelNames, frameNames)    

    # How many plots are needed to plot every channel? channelsPerPlot = 6 by default
    nPlots = math.ceil(N_CHANNELS/channelsPerPlot)
    plotsFClass = [i for i in range(1,nPlots+1)] # Create as many plots as needed. First plots are for f_class
    plotsMutual = [i + nPlots for i in plotsFClass] # Last plots are for mutual

    # For each figure (scores of f_classif), create a 'channelsPerPlot' number of subplots in a column
    # and one vertical plot on the right, with the width relation of 1:40 for the vertical plot and 39:40 for the horizontal plots
    axisFClass = []
    colorBarAxisFClass = []

    for i in plotsFClass: # For each plot corresponding to FClass
        plt.figure(i)
        if i == plotsFClass[-1] and N_CHANNELS%channelsPerPlot > 0: # In the last plot, draw just the resting subplots
            channelsInPlot = N_CHANNELS%channelsPerPlot
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
        if i == plotsMutual[-1] and N_CHANNELS%channelsPerPlot > 0: # If it's the last plot, draw just the resting subplots
            channelsInPlot = N_CHANNELS%channelsPerPlot
        else:
            channelsInPlot = channelsPerPlot

        gsMutual = gridspec.GridSpec(channelsInPlot,40)

        for j in range(channelsInPlot):
            axisMutual.append(plt.subplot(gsMutual[j,0:39]))
        colorBarAxisMutual.append(plt.subplot(gsMutual[:,39]))

    # Draw one subplot for each channel
    for i in range(N_CHANNELS):
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
        if (i%channelsPerPlot != channelsPerPlot - 1) and i != N_CHANNELS - 1: # If it's not the last subplot of the plot
            #import pdb
            #pdb.set_trace()
            axisFClass[i].set_xticks([])
            axisFClass[i].set_xlabel('')
            axisMutual[i].set_xticks([])
            axisMutual[i].set_xlabel('')

        axisFClass[i].set_ylabel(channelNames[i])
        axisMutual[i].set_ylabel(channelNames[i])

    # Set the title for the figures of f_classif scores
    for i in plotsFClass:
        plt.figure(i)
        firstChannel = (i - 1)*channelsPerPlot + 1
        lastChannel = (i)*channelsPerPlot
        if lastChannel > N_CHANNELS:
            lastChannel = N_CHANNELS
        plt.suptitle(f"Feature scores with f_classif, channel {firstChannel} - {lastChannel}")
        plt.savefig(f"{DIR_PATH}/heatmap_fclass_{firstChannel}-{lastChannel}.png")
    # And for the figure 2
    for i in plotsMutual:
        plt.figure(i)
        firstChannel = (i - nPlots - 1)*channelsPerPlot + 1
        lastChannel = (i - nPlots)*channelsPerPlot
        if lastChannel > N_CHANNELS:
            lastChannel = N_CHANNELS
        plt.suptitle(f"Feature scores with mutual_info_classif, channel {firstChannel} - {lastChannel}")
        plt.savefig(f"{DIR_PATH}/heatmap_mutual_{firstChannel}-{lastChannel}.png")

    #plt.show()

    return

if __name__ =='__main__':
    main()
