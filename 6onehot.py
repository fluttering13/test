#data generation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
y=[5,3,4]
#放在第六個
y_train_ohe=tf.one_hot(y, depth=10).numpy()
#轉換1D
y_kera=tf.keras.utils.to_categorical(y,num_classes=None,dtype='float32')
#print(y,"is",y_kera,"one-hot type")
print(y,"is",y_kera,"one-hot type")