from globalVars import DIR_PATH, FEATURE_NAMES, N_CHANNELS, STACKING_WIDTH
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pdb

def getColNames():
    # This function creates an array with the names that identifies each feature column

    # Channel names like 'Ch1', 'Ch2'...
    channelNames = []
    for i in range(N_CHANNELS):
        channelNames.append(f"Ch{i+1}")
    
    # Frame names like 'Mn' for the frame -n, 'Pn' for the frame +n and '0' for the central frame
    frameNames = []
    for i in range(-STACKING_WIDTH,STACKING_WIDTH+1):
        if i < 0:
            frameNames.append(f"M{i*-1}")
        elif i > 0:
            frameNames.append(f"P{i}")
        else:
            frameNames.append('0')
    
    # Combines the name of the channel, the name of the frames and the name of the features
    # to create the name of the columns.
    colNames = ['Position']
    for i in channelNames:
        for j in frameNames:
            for k in FEATURE_NAMES:
                colNames.append(f"{i}_{j}_{k}")

    return colNames

def getIndexesDataFrame(scores, colNames,maxPosition=4):
    # This function creates a DataFrame with the count of how many times has been ranked each feature in the n position

    # maxPosition is the last position considered. If 3 is selected, only is going to be count how many times each feature has been ranked in the first, second or third position

    maxIndexesMatrix = np.zeros((maxPosition,len(colNames)),dtype=int)
    
    # Counts how many times have been ranked each feature as one of each considered positions
    for i in range(1,maxPosition + 1):
        maxIndexes = scores.argsort()[:,-i]
        maxIndexesCount = [i]
        for j in range(np.shape(scores)[1]):
            maxIndexesCount.append(np.count_nonzero(maxIndexes == j))
    
        maxIndexesMatrix[i-1] = maxIndexesCount
    
    
    df = pd.DataFrame(columns=colNames,data=maxIndexesMatrix)
    
    return df

def printRanking(scores,colNames,maxPosition,scoresName,latexFormat=True):
    # This function returns a ranking with the highest scores as a string that can be displayed or saved to a file.
    # If latexFormat = True, the ranking will be written as a LaTeX tabular, so it can be inserted directly in a document.
    # If latexFormat = False, the ranking will be written as plain text

    ranking = ""
    sortedScores = sorted(scores)

    if latexFormat:
        ranking += '\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n'
        ranking += '\t\\textbf{Position} & \\textbf{Channel} & \\textbf{Frame} & \\textbf{Feature} & \\textbf{Score} \\\\\n'
        ranking += '\t\\hline\\hline\n'
        for i in range(1,maxPosition+1):
            searchedValue = sortedScores[-i] # Selects the i. highest value
            idx = np.where(scores == searchedValue)[0][0]
            channel, frame, feature = colNames[idx + 1].split('_')
            frame = frame.replace('P','+')
            frame = frame.replace('M', '-')
            ranking += f"\t\\textbf{{ {i} }} & {channel} & {frame} & {feature} & {str(round(searchedValue,3))} \\\\\n"
            ranking += "\t\\hline\n"
        ranking += '\\end{tabular}'
    else:
        ranking += f"Score ranking for {scoresName}\n"

        ranking += "{:<8} {:<15} {:<15}\n".format('Position','Feature','Score')
        for i in range(1,maxPosition+1):
            searchedValue = sortedScores[-i] # Selects the i. highest value
            idx = np.where(scores == searchedValue)[0][0]
            ranking += "{:<8} {:<15} {:<15}\n".format(str(i),colNames[idx + 1],str(searchedValue))

    ranking = ranking.replace('& r &','& $\\bar{r}$ &')
    ranking = ranking.replace('& w &','& $\\bar{w}$ &')
    ranking = ranking.replace('& Pw &','& $P_w$ &')
    ranking = ranking.replace('& Pr &','& $P_r$ &')
    return ranking

def main(uttType,analyzedLabels):

    fileBase = uttType + analyzedLabels

    # Load scores from numpy files
    scoresFClass = np.load(DIR_PATH + f'/scoresFClass{fileBase}.npy')
    scoresMutual = np.load(DIR_PATH + f'/scoresMutual{fileBase}.npy')
    
    channelN = 6
    stackingWidth = 15
    featureNames = ['w','Pw','Pr','z','r']
    
    colNames = getColNames()
    
    pdb.set_trace()

    # If the scores have been calculated by processing all examples at the same time (1 big single batch)
    # Don't draw bar plot, just print the ranking
    if np.shape(scoresFClass)[0] == 1:
        print(printRanking(scoresFClass,colNames,15,'f_classif'))
        print(printRanking(scoresMutual,colNames,15,'mutual_info_classif'))
    # If examples have been divided in multple batches
    # Draw bar plot
    else:

        # Get a dataframe with how many times each feature has been ranked as one of the considered positions
        dfFClass = getIndexesDataFrame(scoresFClass, colNames)
        dfMutual = getIndexesDataFrame(scoresMutual, colNames)
    
        # Discards those features that never have been ranked in any of the considered positions
        dfFClass = dfFClass.loc[:, (dfFClass != 0).any(axis=0)]
        # Discards those features that haven't been ranked at least twice in any of the considered positions
        dfMutual = dfMutual.loc[:, (dfMutual > 1).any(axis=0)]

        # Decomposes the dataframe content into individual entries where the name of the feature is a value in a column
        dfFClass_tidy=pd.melt(dfFClass,id_vars=['Position'],var_name="Feature",value_name="Count")
        dfMutual_tidy=pd.melt(dfMutual,id_vars=['Position'],var_name="Feature",value_name="Count")
    
        # Plots the bar plots
        fig = plt.figure()
        sns.barplot(data=dfFClass_tidy,y="Count",x="Feature",hue="Position")
        plt.xticks(rotation=90)
        plt.savefig(f"{DIR_PATH}/{fileBase}FClassBarplot.png")
    
        sns.barplot(data=dfMutual_tidy,y="Count",x="Feature",hue="Position")
        plt.xticks(rotation=90)
        plt.savefig(f"{DIR_PATH}/{fileBase}MutualBarplot.png")

    return

if __name__ == '__main__':
    main()
