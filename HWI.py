#Citation Request:
#This dataset is public available for research. The details are described in 
#[Moro et al., 2014].

#Data Set Information:
#he data is related with direct marketing campaigns of a Portuguese banking institution. 
#The marketing campaigns were based on phone calls. 
# Often, more than one contact to the same client was required, in order to access if the product.

#Attribute Information
#default: has credit in default? (categorical: 'no','yes')
#housing: has housing loan? (categorical: 'no','yes')
#loan: has personal loan? (categorical: 'no','yes')
#day:last contact day of the week
#duration: last contact duration
#campaign: number of contacts performed during this campaign and for this client
#pdays: number of days that passed by after the client was last contacted from a previous campaign
#previous: number of contacts performed before this campaign and for this client

#goal:
#determine how to integrate this data with your MLP implementation and write any code necessary to train and test the MLP on your data
#compare the result from your networks with 1, 2, and 3 hidden layers

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
# Read the original dataset
df = pd.read_csv("bank-full.csv", header=0)

X = df[df.columns[1:9]]
y = df['balance']
plt.figure() # Create a new figure
f, ax1 = plt.subplots()
#show the data innitally 
for i in range (1,8):
 number = 420 + i  # Integer subplot specification must be a three digit number (It is not necessary to plot them all)
 ax1 = plt.subplot(number)
 plt.title(list(df)[i])
 ax1.scatter(df[df.columns[i]],y) #Plot the datapoints
plt.show()
# Scale the data for convergency optimization
scaler = preprocessing.StandardScaler()
# Set the transform parameters
X = scaler.fit_transform(X)
# Split the datasets
trainx, test_x, trainy, test_y = train_test_split(X, y,test_size=1)


#parameters
epN = 3000
baN = trainy.shape[0]
# transform to tf tensors
trainx, trainy = tf.cast(trainx, tf.float32),tf.cast(trainy, tf.float32)
test_x, test_y = tf.cast(test_x, tf.float32),tf.cast(test_y, tf.float32)
# Set the model with 3 hidden layers of relu activation
model = tf.keras.models.Sequential([
 tf.keras.layers.Dense(512, activation=tf.nn.relu),
 tf.keras.layers.Dense(256, activation=tf.nn.relu),
 tf.keras.layers.Dense(256, activation=tf.nn.relu),
 tf.keras.layers.Dense(1)])
model.compile(optimizer='Adam', loss='mse') #mse,
model.fit(trainx, trainy, batch_size=baN, epochs=epN)
model.evaluate(test_x, test_y)
model.summary()

for i in range (1,8):
 number = 420 + i
 #ax1.locator_params(nbins=3) #adjust the limit of scale
 ax1 = plt.subplot(number)
 plt.title(list(df)[i])
 ax1.scatter(test_x[:,i],test_y,color='blue') #Plot the datapoints
 plt.scatter(trainx[:,i],trainy,color='red')
plt.show()

#use the configuration of neurons:
#one hidden layer (1024,1) ,epoch=3000 ;total parameters: 10241 ; the total loss remain:7507 
#two hidden layer (512,512,1),epoch=3000 ; total parameters: 267777 ;the total loss remain:114.2571
#three hidden layer (512,256,256,1),epoch=3000 :total parameters:201985 ;the total loss remain:11.5062
