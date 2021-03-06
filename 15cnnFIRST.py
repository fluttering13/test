import tensorflow as tf
import matplotlib.pyplot as plt

# Acquire data
mnist = tf.keras.datasets.mnist
(trainx,trainy), (test_x, test_y) = mnist.load_data()
#Show the 1st training image
mnistmp = trainx[0].reshape((28, 28),order='C')
plt.imshow(mnistmp, cmap='Greys',
 interpolation='nearest')
# Normalise data
trainx, test_x = tf.cast(trainx/255.0, tf.float32),
 tf.cast(test_x/255.0, tf.float32)
trainy, test_y = tf.cast(trainy , tf.int64 ),
 tf.cast(test_y , tf.int64 )
#Expand one more dim for channel
trainx = tf.expand_dims(trainx,3)
test_x = tf.expand_dims(test_x,3)
# Parameters
baN = 200
epN = 10
# Network Parameters
nclass = 10 # 10 digit classes
rate = 0.2 # Dropout rate
# Sequential convolution net with keras
convnet = tf.keras.models.Sequential([
 tf.keras.layers.Conv2D(32, (5,5), strides=(1,1),
 padding='same', data_format='channels_last',
 activation='relu'),#the samllest size is 32
 tf.keras.layers.MaxPool2D(pool_size=(2,2),
 strides=(2,2), padding='same',
 data_format='channels_last'),
 tf.keras.layers.Conv2D(32, (3,3), strides=(1,1),
 padding='same', data_format='channels_last',
 activation='relu'),
 tf.keras.layers.MaxPool2D(pool_size=(2,2),
 strides=(2,2), padding='same',
 data_format='channels_last'),
 tf.keras.layers.Dropout(rate),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(128, activation='relu'),
 tf.keras.layers.Dense(nclass, activation='softmax')])
convnet.compile(optimizer= 'Adam',
 loss='sparse_categorical_crossentropy',
 metrics = ['accuracy'])
convnet.fit(trainx,trainy,batch_size=baN,epochs=epN)
convnet.evaluate(test_x, test_y)
convnet.summary()