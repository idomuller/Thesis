import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, scale# use to normalize the data
from sklearn.decomposition import PCA# Use to perform the PCA transform
import matplotlib.pyplot as plt
from utils import GetStudyData, GetHandFingersDataRelateToCenter, GetGravityData
import plotPCA
from scipy import stats
import seaborn as sns

normlizeFlag = True
#allFeatures = GetHandFingersDataRelateToCenter()

allFeatures = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                   'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                   'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
#[groupedFeature, names, subject_id] = GetStudyData(allFeatures, normlizeFlag)
[groupedFeature, names, subject_id, data] = GetGravityData(allFeatures, normlizeFlag)

rCol = []
for ind in range(len(subject_id)):
    rCol.append('C'+str(ind))

# Seperate subjects by their reaction
data_mean = data.mean(axis=1, keepdims=True)
data_std = data.std(axis=1, keepdims=True)
data_mean_std = np.array([data_mean[:, 0] - data_std[:, 0], data_mean[:, 0] + data_std[:, 0]]).T

positive_reaction_ids = np.array([data_mean_std[:, 0] > 0]).T
negative_rection_ids = np.array([data_mean_std[:, 1] < 0]).T
balance_reaction_ids = (positive_reaction_ids == False) & (negative_rection_ids == False)
reactions_ids = np.array([negative_rection_ids[:, 0], balance_reaction_ids[:, 0], positive_reaction_ids[:, 0]]).T
if len(data_mean_std) > 30:
    positive_reaction_ids = positive_reaction_ids[['r' in s for s in names]]
    negative_rection_ids = negative_rection_ids[['r' in s for s in names]]
    balance_reaction_ids = balance_reaction_ids[['r' in s for s in names]]

# Plot the reaction by groups of reactions
fig, ax = plt.subplots()
plt.title('Group Comparison', fontsize=20)
plt.ylabel('Ratio', fontsize=12)
plt.xlabel("Index of time", fontsize=12)
grouped_feature_plot = groupedFeature[['r' in s for s in names]]

groups = [np.where(negative_rection_ids), np.where(balance_reaction_ids), np.where(positive_reaction_ids)]
Legend = ['Negative reaction', 'Balance reaction', 'Possitive reaction']
for ind, groupInds in enumerate(groups):
    if groupInds:
        tempDF = pd.DataFrame()
        tempDF = grouped_feature_plot.iloc[groupInds[0], :]
        tempDFVal = tempDF.values
        dataByTimes = np.resize(tempDFVal, (tempDFVal.shape[0]*12, np.shape(data)[1]))

        if not(normlizeFlag):
            for i in range(dataByTimes.shape[0]):
                dataByTimes[i,:] = (dataByTimes[i, :]-dataByTimes[i, 0])/ dataByTimes[i, 0]
        groupMeans = dataByTimes.mean(axis=0)
        groupStd = dataByTimes.std(axis=0)
        ax.plot(groupMeans, color=rCol[ind])
        x = np.linspace(0, len(groupStd) - 1, len(groupStd))
        ax.fill_between(x, groupMeans - groupStd, groupMeans + groupStd, color=rCol[ind], alpha=0.3)

plt.legend(Legend[0:ind+1], loc='center left', bbox_to_anchor=(0, 0.9))
plt.show()


# Data Visualization
x = groupedFeature.values
#x_norm = StandardScaler().fit_transform(x) # normalizing the features
min_max_scaler = MinMaxScaler()
x_norm = min_max_scaler.fit_transform(x)
y=scale(groupedFeature.values)

groupedFeature['label']='n'

# convert the normalized features into a tabular format with the help of DataFrame.
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_Data = pd.DataFrame(x_norm, columns=feat_cols)
normalised_Data.tail()

pca_Data = PCA(n_components=2)
principalComponents = pca_Data.fit_transform(x)

# create a DataFrame that will have the principal component values for all samples
principal_breast_Df = pd.DataFrame(data = principalComponents, columns=['principal component 1', 'principal component 2'])
principal_breast_Df.tail() #Return the last n rows.

# Show for each principle component how much of the informaion it holds
print('Explained variation per principal component: {}'.format(pca_Data.explained_variance_ratio_))

# Plot the visualization of the two PCs
charectaristcs_PD = pd.read_excel(r"C:\Users\ido.DM\Google Drive\Thesis\Data\Characteristics.xlsx")
charectaristcs_PD = charectaristcs_PD.set_index('Ext_name')
exData = charectaristcs_PD.loc[:, [col for col in charectaristcs_PD.columns if 'PP' in col]]

plotPCA.plotPCA2D(principal_breast_Df, names, 'r', exData)

minVal = min(exData.values)*0.8
maxVal = max(exData.values)
r = 100*(exData.values-minVal)/(maxVal-minVal)
rDF = pd.DataFrame(data=r, index=exData.index)
#plt.subplot()
markers = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "o", "v", "^", "<", ">", "1"]
rgb_values = sns.color_palette("muted",  len(subject_id))

# Plot both hands together + collect distances
in_subject_hand_dist = []
between_subject_hand_dist=[]
perm = []
fig, ax = plt.subplots()
for i, s in enumerate(subject_id):
    is_match = np.where(np.char.find(names, s) == 0)
    xa = principal_breast_Df.iloc[is_match[0][0], 0]
    ya = principal_breast_Df.iloc[is_match[0][0], 1]
    xb = principal_breast_Df.iloc[is_match[0][1], 0]
    yb = principal_breast_Df.iloc[is_match[0][1], 1]

    relevantR = rDF[[s in t for t in rDF.index]]
    ax.scatter(xa, ya, c=rCol[i], linewidths=0.5, edgecolors='r', s=relevantR.values*6, alpha=0.5)#marker=markers[i],
    plt.text(xa, ya, s.split('_')[1], horizontalalignment='center', verticalalignment='center')
    ax.scatter(xb, yb, c=rCol[i], linewidths=0.5, edgecolors='g', s=relevantR.values*6, alpha=0.5)
    plt.text(xb, yb, s.split('_')[1], horizontalalignment='center', verticalalignment='center')
    in_subject_hand_dist.append(np.linalg.norm(np.array(xa, ya) - np.array(xb, yb)))
    perm.append([s + s])
    for j, s_n in enumerate(subject_id):
        current_match_1 = [s + s_n]
        current_match_2 = [s_n + s]
        if not(any(np.char.find(perm, current_match_1) == 0) or any(np.char.find(perm, current_match_2) == 0)):
            perm.append(current_match_1)
            for hand_i in range(1):
                xa = principal_breast_Df.iloc[is_match[0][hand_i], 0]
                ya = principal_breast_Df.iloc[is_match[0][hand_i], 1]
                xb = principal_breast_Df.iloc[j, 0]
                yb = principal_breast_Df.iloc[j, 1]
                between_subject_hand_dist.append(np.linalg.norm(np.array(xa, ya) - np.array(xb, yb)))
#ax.legend(names, loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)
# Statistics of the difference between hands
t, p = stats.ttest_ind(in_subject_hand_dist, between_subject_hand_dist)
print('Match hands dist = ' + str(np.round(np.average(in_subject_hand_dist), 4)) + ' + ' + str(np.round(np.nanstd(in_subject_hand_dist), 4)))
print('Not match dist = ' + str(np.round(np.average(between_subject_hand_dist), 4)) + ' + ' + str(np.round(np.nanstd(between_subject_hand_dist), 4)))
print('p = ' + str(np.round(p, 4)))

prc_75 = np.percentile(in_subject_hand_dist, 75)
#plt.show()

# Start plot lines by groups
#x_thresh = input('Enter X threshold: ')
x_thresh = 0 #int(x_thresh) if x_thresh else None
y_thresh =0 #input('Enter Y threshold: ')
#y_thresh = int(y_thresh) if y_thresh else None
relevant_data = principalComponents[['r' in s for s in names]]
#relevant_data = relevant_data.values#reset_index(drop=True).
groupedFeature = groupedFeature[['r' in s for s in names]]
groups = [[], [], [], []]
if not(x_thresh == None) and not (y_thresh == None):
    choice = np.logical_and(np.greater_equal(relevant_data[:,0], x_thresh), np.greater_equal(relevant_data[:,1], y_thresh))
    groups[0] = np.where(choice)
    choice = np.logical_and(np.greater_equal(relevant_data[:, 0], x_thresh), np.less(relevant_data[:, 1], y_thresh))
    groups[1] = np.where(choice)
    choice = np.logical_and(np.less(relevant_data[:, 0], x_thresh), np.less(relevant_data[:, 1], y_thresh))
    groups[2] = np.where(choice)
    choice = np.logical_and(np.less(relevant_data[:, 0], x_thresh), np.greater_equal(relevant_data[:, 1], y_thresh))
    groups[3] = np.where(choice)
    Legend = ['1Qtr', '2Qtr', '3Qtr', '4Qtr']

elif not(x_thresh==None) and (y_thresh == None):
    choice = np.greater_equal(relevant_data[:, 0], x_thresh)
    groups[0] = np.where(choice)
    choice = np.less(relevant_data[:, 0], x_thresh)
    groups[1] = np.where(choice)
    Legend = ['X_Possitive', 'X_Negative']

elif (x_thresh == None) and not (y_thresh == None):
    choice = np.greater_equal(relevant_data[:, 1], y_thresh)
    groups[0] = np.where(choice)
    choice = np.less(relevant_data[:, 1], y_thresh)
    groups[1] = np.where(choice)
    Legend = ['Y_Possitive', 'Y_Negative']

colors = ['C0', 'C1', 'C2', 'C3', 'C4']

# plot all on the same figure
fig, ax = plt.subplots()
plt.title('Group Comparison', fontsize=20)
plt.ylabel('Ratio', fontsize=12)
plt.xlabel("Index of time", fontsize=12)
for ind, groupInds in enumerate(groups):
    if groupInds:
        tempDF = pd.DataFrame()
        tempDF = groupedFeature.iloc[groupInds[0], :-1]
        tempDFVal = tempDF.values
        dataByTimes = np.resize(tempDFVal, (tempDFVal.shape[0]*12, 39))

        if not(normlizeFlag):
            for i in range(dataByTimes.shape[0]):
                dataByTimes[i,:] = (dataByTimes[i, :]-dataByTimes[i, 0])/ dataByTimes[i, 0]
        groupMeans = dataByTimes.mean(axis=0)
        groupStd = dataByTimes.std(axis=0)
        ax.plot(groupMeans,color=colors[ind])
        x = np.linspace(0, len(groupStd) - 1, len(groupStd))
        ax.fill_between(x, groupMeans - groupStd, groupMeans + groupStd, color=rCol[ind], alpha=0.3)
plt.legend(Legend[0:ind+1])
plt.show()

# plot all on different figures
fig, ax = plt.subplots(np.shape(groups)[0],1)
fig.suptitle('Group Comparison', fontsize=20)

plt.xlabel("Index of time", fontsize=12)
for ind, groupInds in enumerate(groups):
    if groupInds:
        tempDF = pd.DataFrame()
        tempDF = groupedFeature.iloc[groupInds[0], :-1]
        tempDFVal = tempDF.values
        dataByTimes = np.resize(tempDFVal, (tempDFVal.shape[0]*12, 39))

        if not(normlizeFlag):
            for i in range(dataByTimes.shape[0]):
                dataByTimes[i,:] = (dataByTimes[i, :]-dataByTimes[i, 0])/ dataByTimes[i, 0]
        groupMeans = dataByTimes.mean(axis=0)
        groupStd = dataByTimes.std(axis=0)
        ax[ind].plot(groupMeans,color=colors[ind])
        x = np.linspace(0, len(groupStd) - 1, len(groupStd))
        ax[ind].fill_between(x, groupMeans - groupStd, groupMeans + groupStd, color=rCol[ind], alpha=0.3)
        ax[ind].set(xlabel="Index of time", ylabel="Ratio", title=Legend[ind])
        ax[ind].set_ylim([-0.4, 0.8])
plt.show()