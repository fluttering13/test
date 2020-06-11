import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
df = pd.read_csv("CHD.csv", header=0)
print(df.describe())
fmean = df['age'].mean()
fdage = df['age'].max() - fmean
N = df["age"].count()
N = N - N//5
# parameters
train_epochs = 10
batchsize = N
normx = ( np.transpose([df['age']])- fmean )/ fdage
tensx = tf.cast(normx, tf.float32)
binay = tf.cast(df['chd'], tf.int64) # No need OHE in keras
trainx, trainy = tensx[:N], binay[:N]
test_x, test_y = tensx[N:], binay[N:]
# Create model with keras
model1 = tf.keras.models.Sequential(\
[tf.keras.layers.Dense(2,activation=tf.nn.softmax)])
optimiser = tf.keras.optimizers.Adam()
model1.compile (optimizer = optimiser,loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
model1.fit(trainx, trainy, batch_size=batchsize,epochs=train_epochs)
model1.evaluate(test_x, test_y)
model1.summary()
#Generate a new plot
graphnumber=321
plt.figure(1)

#Generate a new graph, and add it to the complete graph
trX = np.linspace( -fdage, fdage, 100).astype(float)
w = model1.get_weights()
w0, w1 = w[0][0][0]/fdage, w[0][0][1]/fdage
#print(w0,w1)
b0, b1 = w[1][0] , w[1][1]
z0, z1 = w0 * trX + b0 , w1 * trX + b1
w0, w1 = w[0][0][0]/fdage, w[0][0][1]/fdage
print(z0,z1)
# Generate the probabiliy function
trY = np.exp(z1) /( np.exp(z0) + np.exp(z1))
# Draw the samples & probability without the normalization
plt.subplot(graphnumber)
#Plot a scatter draw of the random datapoints
plt.scatter((df['age']),df['chd'])
plt.plot(trX+fmean, trY) #Plot a scatter of the datapoints
plt.grid(True)
plt.show()
#Plot the final graph
plt.savefig("test.svg")
