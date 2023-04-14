#Most of task are in this file, it makes it easier to have all data already here,
#but makes it worse if we want to change certain model onlu, because we need to rerun evrything then

import pandas as pd
import sklearn
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import keras_tuner as kt
from models.HM_Task_2 import heuristic_model

# Task 1
data = pd.read_csv('covtype.csv')
labels = data['Cover_Type']
columns = data.columns[0:2]     # we will use only 2 features for our models to keep it simple and fast
features = data[columns]




# We split data into training and test sets then scale it so our models work better
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
scaler.fit(X_test)

#Task 3 I created 2 basic model KNN and RandomForest because both of them are really good when it comes to classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

RFC = RandomForestClassifier(max_depth= 10)
RFC.fit(X_train.values,y_train.values)

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train.values,y_train.values)

# Task 4 Creating function to create basic model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(hp):
    neurons = hp.Choice('neurons', values=[16, 32, 64])  # defining values of hyperparameter to check for our model
    dropout = hp.Choice('dropout', values=[0.0, 0.2, 0.4])

    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(dropout),
        Dense(neurons, activation='relu'),
        Dropout(dropout),
        Dense(8, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Changing Covet_type column into catorogical values so our model can use them
y_train_nn = tf.keras.utils.to_categorical(y_train, 8)
y_test_nn = tf.keras.utils.to_categorical(y_test, 8)

# usuing tuner from keres to find best hyperparameter
tuner = kt.Hyperband(create_model, objective='val_accuracy', project_name='Tuner' )
tuner.search(X_train, y_train_nn, epochs=10, validation_split=0.2) #finidng best hyperparemeter

best = tuner.get_best_hyperparameters()[0]
print(f'Best nr of neurons: {best.get("neurons")}, Best value for dropout: {best.get("dropout")}')

NN = tuner.hypermodel.build(best) # and builidng our model with them
NN.summary()

history = NN.fit(X_train.values, y_train_nn, epochs=5, validation_data = (X_test.values,y_test_nn))

y_rfc = RFC.predict(X_test.values)
y_knn = KNN.predict(X_test.values)
y_nn = list(np.argmax(NN.predict(X_test), axis=-1))
y_hm = (list(map(heuristic_model,data['Elevation'], data['Aspect'])))

from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score

#Task 5 Using 3 metricks to check my models
bal_acc = [balanced_accuracy_score(y_test,y_rfc),balanced_accuracy_score(y_test,y_knn), balanced_accuracy_score(y_test,y_nn), balanced_accuracy_score(data['Cover_Type'],y_hm)]
pr = [precision_score(y_test,y_rfc, average="micro"),precision_score(y_test,y_knn, average="micro"), precision_score(y_test,y_nn, average="micro"),precision_score(data['Cover_Type'],y_hm,average="micro")]
re = [recall_score(y_test,y_rfc, average="macro"),recall_score(y_test,y_knn, average="macro"), recall_score(y_test,y_nn, average="macro"),recall_score(data['Cover_Type'],y_hm,average="macro")]

print(f'Metrics for RFC model: Balanced Accuracy {bal_acc[0]}, Precision {pr[0]}, Recall {re[0]}')
print(f'Metrics for KNN model: Balanced Accuracy {bal_acc[1]}, Precision {pr[1]}, Recall {re[1]}')
print(f'Metrics for NN model: Balanced Accuracy {bal_acc[2]}, Precision {pr[2]}, Recall {re[2]}')
print(f'Metrics for Heuristic model: Balanced Accuracy {bal_acc[3]}, Precision {pr[3]}, Recall {re[3]}')

import pickle
pickle.dump(RFC, open('saved_models/RFC_model.pkl', 'wb'))
pickle.dump(KNN, open('saved_models/KNN_model.pkl', 'wb'))
pickle.dump(NN, open('saved_models/NN_model.pkl', 'wb'))

#printing learning curves and metricks Task 4 plus 5

plt.figure()
plt.subplot(3,1,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Traning', 'Validation'])
plt.title('Loss of the model')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(3,1,3)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Traning', 'Validation'])
plt.title('Accuracy of the model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')



Names = ['RandomForest', 'K_neigh','Neural Network', 'Heuristic Model']
plt.figure()
plt.subplot(3,1,1)
plt.bar(Names, bal_acc,  width = 0.4)
plt.ylabel("Balanced Accuracy")
plt.title("Metrics")

plt.subplot(3,1,2)
plt.bar(Names, pr, color = 'red',  width = 0.4)
plt.ylabel("Precision")

plt.subplot(3,1,3)
plt.bar(Names, re, color = 'green',  width = 0.4)
plt.xlabel("Type of model")
plt.ylabel("Recall")

plt.show()
