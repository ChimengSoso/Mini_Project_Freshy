import pandas as pd
import numpy as np
from sklearn import tree, preprocessing
from multi_column_label_encoder import MultiColumnLabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('data.csv')

dataset = dataset.drop(['data'], 1)

for i in dataset:
    if dataset[i].dtypes == float:
        dataset[i] = dataset[i].fillna(0)

y = np.array(dataset['n_person'])
dataset = dataset.drop(['n_person'], 1)

print(y)

print(dataset)

# plt.plot()