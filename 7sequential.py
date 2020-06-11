#data generation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#catch data
mnist=tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y)=mnist.load_data()
#print(train_x,train_y)
batch_size=32
epochs=10
#normailzation
train_x,test_x=tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)
train_y,test_y=tf.cast(train_y,tf.int64),tf.cast(test_y,tf.int64)
#sequential model
model1=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(), #change the date to 1 dimesion
    tf.keras.layers.Dense(512,activation=tf.nn.relu),#iput=512
    tf.keras.layers.Dropout(rate=0.2),#randomly drop out the neurons
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)#last layer to normalize the result
]
)
optimiser=tf.keras.optimizers.Adam() #use the Adam algorithm to optimize(momentum+AdaGrad)
model1.compile(optimizer=optimiser,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model1.fit(train_x,train_y,batch_size=batch_size,epochs=epochs)
model1.evaluate(test_x,test_y)
model1.summary()