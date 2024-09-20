from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.metrics import R2Score# type: ignore
import matplotlib.pyplot as plt
df = fetch_california_housing()
X_treino, X_teste, Y_treino, Y_teste = train_test_split(df.data, df.target)
X_treino2, X_val, Y_treino2, Y_val = train_test_split(X_treino, Y_treino)
scaler = StandardScaler()
X_treino2 = scaler.fit_transform(X_treino2)
X_val = scaler.fit_transform(X_val)
X_teste = scaler.fit_transform(X_teste)
model = tf.keras.Sequential()
model = tf.keras.Sequential([tf.keras.layers.Dense(30, activation = 'relu', input_shape = X_treino2.shape[1:]), tf.keras.layers.Dense(1)])
model.compile(loss = "mse", optimizer = Adam(), metrics = [R2Score])
batch_size = 64
hostory = model.fit(X_treino2, Y_treino2, epochs=20, validation_data=(X_val, Y_val), batch_size = batch_size)
plt.plot(hostory.history["loss"], label = "loss")
plt.plot(hostory.history["val_loss"], label = "val_loss")
plt.xlabel("Epochs")
plt.ylabel("MSE")
plt.legend()
plt.show()
y_pred = model.predict(X_teste)
print(y_pred)