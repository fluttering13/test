import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 

trN=200
teN=60

#mock data
dsX = np.linspace(-1, 1, trN + teN).transpose() #260 paramaters 
liY = 2.0 * pow(dsX,2) + 2.0 * dsX + 1.0 # 2*x^2+2x+1
dsY = liY + np.random.randn(*dsX.shape) * 0.3 #noise

# plt.figure() # Create a new figure
# plt.title('Original data')
# plt.plot(dsX,liY, color='gray', linewidth=3.0)
# plt.scatter(dsX, dsY, color='blue') #Plot the datapoints
# plt.show()

X, Y = shuffle(dsX, dsY) #

trainx, trainy = tf.cast(X[:trN], tf.float32),tf.cast(Y[:trN], tf.float32)
#print(trainx,trainy)
test_x, test_y = tf.cast(X[trN:], tf.float32),tf.cast(Y[trN:], tf.float32)

#print(test_x,test_y)

totalx = tf.cast(dsX, tf.float32)
epN = 6000
baN = trN//4
# the model: a simple imput + a hidden layer of sigmoid
model = tf.keras.models.Sequential([
 tf.keras.layers.Dense(10, activation=tf.nn.relu),
 tf.keras.layers.Dense(1)])
model.compile(optimizer='Adam', loss='mse')
model.fit(trainx, trainy, batch_size=baN, epochs=epN)
model.evaluate(test_x, test_y)
prY = model.predict(totalx)
model.summary()
plt.plot(dsX,prY,color='red', linewidth=3.0)
plt.show()