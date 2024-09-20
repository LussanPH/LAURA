import pandas as pd
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def normalizar(valorM, valorm, valores):
    i = 0
    for valor in valores:
        valor = (valor - valorm)/(valorM - valorm)
        valores[i] = valor
        i+=1    
    return valores    

iris = pd.read_csv("iris.csv")
Y = iris["Species"]
X = iris.drop(labels = ["Species", "Id"], axis = 1)
maiores = []
menores = []
for i in range(4):
    maiores.append(np.max(X.iloc[:, i]))
    menores.append(np.min(X.iloc[:, i]))
for i in range(4):
    X.iloc[:, i] = normalizar(maiores[i], menores[i], X.iloc[:, i])  
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = 0.3, random_state=9)
X_treino2, X_val, Y_treino2, Y_val = train_test_split(X_treino, Y_treino, test_size = 0.1, random_state=9)
X_treino = X_treino.astype('float32')
num_classes = 3
Y_treino2 = keras.utils.to_categorical(Y_treino2, num_classes)
Y_val = keras.utils.to_categorical(Y_val, num_classes)
Y_teste = keras.utils.to_categorical(Y_teste, num_classes)
model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(4,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
batch_size = 64
epochs = 45
history = model.fit(X_treino2, Y_treino2,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_val, Y_val))
fig, ax = plt.subplots(1,2, figsize=(16,8))
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation loss")
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()
predicted_classes = model.predict(X_teste)
print(predicted_classes)
print(Y_teste)
Y_pred = np.argmax(np.round(predicted_classes), axis=1)
Y_test_c = np.argmax(Y_teste, axis = 1)
target_names = ['0', '1', '2']
print('Classification Report')
print(classification_report(Y_test_c, Y_pred, target_names=target_names))