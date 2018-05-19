from tkinter import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from imblearn.combine import SMOTEENN
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from multi_column_label_encoder import MultiColumnLabelEncoder

def ML(data_for_prediction = None, showPrediction = True, showHappen = True, showAccurate = True, showMatrixConfusion = True):

    dataset = pd.read_csv('data.csv')

    dataset = dataset.drop(['day','month', 'year'], 1)

    y = np.array(dataset['ET'])
    dataset = dataset.drop(['ET'], 1)
    sm = SMOTEENN()
    for i in dataset:
        if dataset[i].dtypes == float:
            dataset[i] = dataset[i].fillna(0.0)
        if dataset[i].dtypes == object:
            dataset[i] = dataset[i].fillna('None')

    dataset = MultiColumnLabelEncoder().fit_transform(dataset)

    # Normalrize data
    x = np.array(dataset)

    # Over Sampling data
    x, y = sm.fit_sample(x, y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    L_svc = LinearSVC()
    L_svc.fit(x_train, y_train) # Train data

    prediction = L_svc.predict(x_test) # Prediction

    if showPrediction:
        print("\nPrediction : ")
        print(prediction)
        print('')

    if showHappen:
        print("\nSould be happen:")
        print(y_test)
        print('')

    acc = L_svc.score(x_test, y_test)

    if showAccurate:
        print("\nAccurate:")
        print(acc)
        print('')

    if showMatrixConfusion:
        print("\nConfusion Matrix")
        print(confusion_matrix(prediction, y_test))
        print('')

    if not isinstance(data_for_prediction, type(None)):

        data_for_prediction = data_for_prediction.drop(['day','month', 'year'], 1)

        for i in data_for_prediction:
            if data_for_prediction[i].dtypes == float:
                data_for_prediction[i] = data_for_prediction[i].fillna(0.0)
            if data_for_prediction[i].dtypes == object:
                data_for_prediction[i] = data_for_prediction[i].fillna('None')

        data_for_prediction = MultiColumnLabelEncoder().fit_transform(data_for_prediction)

        data_for_prediction = np.array(data_for_prediction)
        prediction = L_svc.predict(data_for_prediction)
        
        return prediction, acc

    return None

def app():
    datainput = pd.read_csv('input.csv')
    rlt = ML(datainput, 0, 0, 0, 0)

    print("Processing...\n")
    
    n_predict = 10
    avg_acc = 0.0
    n_success = 0
    for i in range(n_predict):
        if rlt != None:
            avg_acc += rlt[1]
            n_success += 1.0
            rlt = ML(datainput, 0, 0, 0, 0)
        print("Processing %.2f%%..."%((i+1) * 100.0/ n_predict))

    avg_acc /= n_success
    
    if rlt == None:
        print('Can not tell anything')
        return 'Can not tell anything'
    else:
        rlt_output = ""
        len_rlt = len(rlt[0])
        showMessage = "Consider %d place"
        if len_rlt > 1:
            showMessage += 's'
        showMessage += '. Accurate Ans : %.2f%%\n'
        accurateVal = avg_acc
        showMessage = showMessage%(len_rlt, accurateVal*100.0)
        
        print(showMessage)
        rlt_output += showMessage + '\n'

        idx = 1
        for ans in rlt[0]:
            if ans == 1:
                partial_output = ""
                partial_output += "  + Place %d: Have UFO!\n"%idx
                print(partial_output)
            else:
                partial_output = ""
                partial_output += "  + Place %d: Not anything~\n"%idx
                print(partial_output)
            idx += 1
            rlt_output += partial_output + '\n'
        return rlt_output

def hello():
    global bg

    b = Label(text=app(),fg="WHITE",bg="BLACK")
    b.pack()

    bg.pack(side = BOTTOM)


root = Tk()
root.configure(background='black')
root.title('UFO')
    

photo = PhotoImage(file="new.png")
bg = Label(root,image=photo,width = 500,height = 600,borderwidth=0)
bg.pack(side = TOP)

root.option_add("*Font", "consolas 20")
b = Button(root, text="Predict", width=10,background = 'WHITE',borderwidth=0)
b.pack(side=BOTTOM, padx=20, pady=15)
b.config(command=hello)

root.mainloop()