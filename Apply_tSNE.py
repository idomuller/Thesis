import numpy as np
import pandas as pd
from utils import fashion_scatter
from sklearn.manifold import TSNE
from utils import GetStudyData
import matplotlib.pyplot as plt

'''
    Load training data:
    Rows = N samples
    Columns = features
'''
RS=150
normlizeFlag = True
allFeatures = ['Thumbs_dist_Intence', 'Thumbs_proxy_Intence', 'Index_dist_Intence', 'Index_proxy_Intence',
                   'Middle_dist_Intence', 'Middle_proxy_Intence', 'Ring_dist_Intence', 'Ring_proxy_Intence',
                   'Pinky_dist_Intence', 'Pinky_proxy_Intence', 'Palm_arch_Intence', 'Palm_Center_Intence']
[groupedFeature, names] = GetStudyData(normlizeFlag)

charectaristcsPD = pd.read_excel(r"C:\Users\ido.DM\Google Drive\Thesis\Data\Characteristics.xlsx")
charectaristcsPD = charectaristcsPD.set_index('Ext_name')
rCol = [col for col in charectaristcsPD.columns if 'PP' in col]
exData = charectaristcsPD.loc[:, rCol]
minVal = min(exData.values)*0.8
maxVal = max(exData.values)
r = 100*(exData.values-minVal)/(maxVal-minVal)
rDF = pd.DataFrame(data=r, index=exData.index)

#testVals = np.linspace(5, 30, 30/5)#perplexity
testVals = np.linspace(50, 1000, 20)
fig, axs = plt.subplots(5, 4, figsize=(15, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.001)
axs = axs.ravel()
plt.title('tSNE- Vs perplexity', fontsize=20)

perplexityVal = 5
#learning_rate = 200.0
for i, learning_rate in enumerate(testVals):

    fashion_tsne = TSNE(n_components=2, perplexity=perplexityVal, early_exaggeration=12.0,
                    learning_rate=learning_rate, n_iter=1000, n_iter_without_progress=300,
                    min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0,
                    random_state=None, method='barnes_hut', angle=0.5).fit_transform(groupedFeature.values)
    for j, txt in enumerate(names):
        relevantR = rDF[[txt in s for s in rDF.index]]
        #axs[i].scatter(dataF.loc[i, 'principal component 1'], dataF.loc[i, 'principal component 2'], c='b', s=relevantR.values*6, alpha=0.3)
        #plt.text(dataF.loc[i, 'principal component 1'], dataF.loc[i, 'principal component 2'], txt)
        axs[i].scatter(fashion_tsne[j, 0], fashion_tsne[j, 1], lw=0, c='b', alpha=0.3, s=relevantR.values)
    axs[i].set_title('learning_rate= ' + str(learning_rate), fontsize=8)


fashion_scatter(fashion_tsne, np.array([0]*fashion_tsne.shape[0]))