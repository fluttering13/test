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
    ax1.locator_params(nbins=3)
    ax1 = plt.subplot(number)
    plt.title(list(df)[i])
    ax1.scatter(df[df.columns[i]],y) #Plot a scatter draw of the  datapoints

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# Scale the data for convergency optimization
scaler = preprocessing.StandardScaler()

# Set the transform parameters
X = scaler.fit_transform(X)

# Split the datasets
trainx, test_x, trainy, test_y = train_test_split(X, y, test_size=0.25)

#parameters
epN = 100
baN = trainy.shape[0]
# transform to tf tensors
trainx, trainy = tf.cast(trainx, tf.float32), tf.cast(trainy, tf.float32)
test_x, test_y = tf.cast(test_x, tf.float32), tf.cast(test_y, tf.float32)

# Set the model, a simple imput, a hidden layer of sigmoid activation
##rewite with 4 layers
model = tf.keras.models.Sequential([
         tf.keras.layers.Dense(128,activation=tf.nn.relu),
         tf.keras.layers.Dense( 64,activation=tf.nn.relu),
         tf.keras.layers.Dense( 32, activation=tf.nn.relu),
         tf.keras.layers.Dense(1)])

model.compile(optimizer='Adam', loss='categorical_crossentropy') ##rewite with categorical_crossenrtopy
model.fit(trainx, trainy, batch_size=baN, epochs=epN)
model.evaluate(test_x, test_y)
model.summary()
