import pandas as pd
import numpy as np
from sklearn import tree, preprocessing
from multi_column_label_encoder import MultiColumnLabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score