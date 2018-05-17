from sklearn import preprocessing

class MultiColumnLabelEncoder:

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = preprocessing.LabelEncoder(
                ).fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = preprocessing.LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)