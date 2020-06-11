import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import pandas as pd
#load the data sheet
(xtrain,ytrain),(xtest,ytest)=fashion_mnist.load_data()
fashion_labels=["T-shirt/top","Trousers","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
batch_size=128
epochs=3
n_classes=10
width=28
height=28
#normalize the feature for better training
xtrain=xtrain.astype('float32')/255.0
xtest=xtest.astype('float32')/255.0
#flatten the features for use the training algorithm
xtrain=xtrain.reshape((60000,width*height))
xtest=xtest.reshape((10000,width*height))
#print(xtrain,xtest)
split=50000
#split feature training set into training and vakidation sets
(xtrain,xvalid)=xtrain[:split],xtrain[split:]
(ytrain,yvalid)=ytrain[:split],ytrain[split:]
ytrain_ohe=tf.one_hot(ytrain,depth=n_classes).numpy()
yvalid_ohe=tf.one_hot(yvalid,depth=n_classes).numpy()
ytest_ohe=tf.one_hot(ytest,depth=n_classes).numpy()
#plot images
_,image=plt.subplots(1,10,figsize=(8,1))
for i in range(10):
    image[i].imshow(np.reshape(xtrain[i],(width,height)),cmap="Greys")
    print(fashion_labels[ytrain[i]],sep=",end=")
plt.show()
#bulid the model 
model=tf.keras.models.Sequential(\
[tf.keras.layers.Dense(n_classes,activation='softmax')]    )
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#loss function=cross entropy (not sure )
#metrics accracy is normal standard for error rate
model.fit(xtrain,ytrain_ohe,batch_size=batch_size,epochs=epochs,validation_data=(xvalid,yvalid_ohe))
model.summary()


# evaluate the model on the test set
scores = model.evaluate(xtest, ytest_ohe, batch_size)
print("Final test loss and accuracy :", scores)
y_predictions = model.predict(xtest)
# example of one predicted versus one true fashion label
index = 42
index_predicted = np.argmax(y_predictions[index]) #model預測的丟進去
# largest label probability
index_true = np.argmax(ytest_ohe[index])
# pick out index of element with a 1 in it
print("When prediction is " , index_predicted)
print("ie. predicted label is",
fashion_labels[index_predicted])
print("True label is ", fashion_labels[index_true])
print ("\n\nPredicted V (True) fashion labels,\
green is correct, red is wrong")
size = 12 # 12 random numbers out of x_test.shape[0]
fig = plt.figure(figsize=(15,3))
rows = 3
cols = 4
for i, index in enumerate(np.random.choice(\
    xtest.shape[0], size = size, replace = False)):
    axis=fig.add_subplot(rows,cols,i+1)
 # position i+1 in grid with rows rows and cols columns
    axis.imshow(xtest[index].reshape(width,height),cmap="Greys")
    index_predicted = np.argmax(y_predictions[index])
    index_true = np.argmax(ytest_ohe[index])
    axis.set_title(("{} ({})").format(\
    fashion_labels[index_predicted],fashion_labels[index_true]),
    color=("green" if index_predicted == index_true else "red"))
plt.show()