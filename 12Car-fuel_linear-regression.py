import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Read the original dataset
df = pd.read_csv("mpg.csv", header=0)
# Convert the displacement column as float
df['displacement']=df['displacement'].astype(float)
# We get data columns from the dataset
# First and last (mpg and car names) are ignored for X
X = df[df.columns[1:8]]
y = df['mpg']
plt.figure() # Create a new figure
f, ax1 = plt.subplots()
for i in range (1,8):
 number = 420 + i
 #ax1.locator_params(nbins=3) #adjust the limit of scale
 ax1 = plt.subplot(number)
 plt.title(list(df)[i])
 ax1.scatter(df[df.columns[i]],y) #Plot the datapoints
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
# Scale the data for convergency optimization
scaler = preprocessing.StandardScaler()
# Set the transform parameters
X = scaler.fit_transform(X)
print(X.shape)
# Split the datasets
trainx, test_x, trainy, test_y = train_test_split(X, y,test_size=1)
#parameters
epN = 3000
baN = trainy.shape[0]
# transform to tf tensors
trainx, trainy = tf.cast(trainx, tf.float32),tf.cast(trainy, tf.float32)
test_x, test_y = tf.cast(test_x, tf.float32),tf.cast(test_y, tf.float32)
#print(trainx.shape,trainy.shape,df.columns[i])



# Set the model with 2 hidden layers of relu activation
model = tf.keras.models.Sequential([
 tf.keras.layers.Dense(10, activation=tf.nn.relu),
 tf.keras.layers.Dense( 5, activation=tf.nn.relu),
 tf.keras.layers.Dense(1)])
model.compile(optimizer='Adam', loss='mse')
model.fit(trainx, trainy, batch_size=baN, epochs=epN)
model.evaluate(test_x, test_y)
model.summary()
for i in range (1,7):
 number = 420 + i
 #ax1.locator_params(nbins=3) #adjust the limit of scale
 ax1 = plt.subplot(number)
 plt.title(list(df)[i])
 ax1.scatter(test_x[:,i],test_y,color='blue') #Plot the datapoints
 plt.scatter(trainx[:,i],trainy,color='red')
plt.show()