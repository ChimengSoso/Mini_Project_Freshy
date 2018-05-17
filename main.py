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
        dataset[i] = dataset[i].fillna(0.0)

y = np.array(dataset['n_person'])
dataset = dataset.drop(['n_person'], 1)

# min_max_scaler = preprocessing.MinMaxScaler()
# dataset[[i for i in dataset if dataset[i].dtype == float]] = min_max_scaler.fit_transform(dataset[[i for i in dataset if dataset[i].dtypes == float]])

# x = np.array(dataset)

x = [np.array(dataset[i]) for i in dataset]

plt.plot(x[0], y, 'yo', x[1], y, 'ro', x[2], y, 'go', x[3], y, 'bo')

# plt.axis([1, 40, 0, 2000])
plt.xlabel('Tempereture')
plt.ylabel('N person')
plt.title('Rate of Number Population')

plt.show()

# print(x)
# print(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# lr = LinearRegression()
# lr.fit(x_train, y_train)

# prediction = lr.predict(x_test)

# print("Prediction : ")
# print(prediction)

# print("\nSould be happen:")
# print(y_test)

# print("\nR2 score:")
# print(r2_score(prediction, y_test))