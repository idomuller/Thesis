from sklearn.datasets import load_breast_cancer
from keras.datasets import cifar10
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, scale# use to normalize the data
from sklearn.decomposition import PCA# Use to perform the PCA transform
import matplotlib.pyplot as plt

# Principle Component Analysis (PCA)
# Example I
#Load the data- Breast Cancer
breast = load_breast_cancer()
breast_data = breast.data

# Extract the lables
breast_labels = breast.target
# Create a column with the lables
labels = np.reshape(breast_labels,(569,1))

final_breast_data = np.concatenate([breast_data,labels],axis=1)

breast_dataset = pd.DataFrame(final_breast_data)

features = breast.feature_names
features_labels = np.append(features,'label')
breast_dataset.columns = features_labels

# Change lables to 0='Benign' and 1='Malignant'
breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)

# Data Visualization
x = breast_dataset.loc[:, features].values
x = StandardScaler().fit_transform(x) # normalizing the features
y=scale(breast_dataset.loc[:, features].values)

# convert the normalized features into a tabular format with the help of DataFrame.
feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_breast = pd.DataFrame(x,columns=feat_cols)
normalised_breast.tail()

pca_breast = PCA(n_components=3)
principalComponents_breast = pca_breast.fit_transform(x)

# create a DataFrame that will have the principal component values for all samples
principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])
principal_breast_Df.tail()#Return the last n rows.

# Show for each principle component how much of the informaion it holds
print('Explained variation per principal component: {}'.format(pca_breast.explained_variance_ratio_))

# Plot the visualization of the two PCs
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=20)
targets = ['Benign', 'Malignant']
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = breast_dataset['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})

# Example II
# Load the CIFAR data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()