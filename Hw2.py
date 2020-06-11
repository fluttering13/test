#data generation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
TrX1=np.linspace(-1,1,101) #切100格
TrX2=np.linspace(-1,1,101)
#mock data
trX1=TrX1+np.random.randn(*TrX1.shape)*0.1 #add the noise
trX2=TrX2+np.random.randn(*TrX2.shape)*0.8
trY=2*trX1+3*trX2+0.2 #weight1:2;wieght2:3 bias=0.2
#2*x1+3*x2+0.2=Y
#convert to tensor form
X1=tf.convert_to_tensor(trX1,dtype=tf.float32,name="x1")
X2=tf.convert_to_tensor(trX2,dtype=tf.float32,name="x2")
Y=tf.convert_to_tensor(trY,dtype=tf.float32,name="y")
#plt all the thing to show the mock date
plt.figure()
plt.scatter(X1,Y,c='red',label='x1')
plt.scatter(X2,Y,c='blue',label='x2')
plt.plot(TrX1, 0.2+ 2*TrX1+3*TrX2 )
plt.show()

#define the neuron
def model(X1,X2, w1 ,w2,b):
 return tf.multiply(X1,w1)+tf.multiply(X2,w2)+b # define line as X * w + b
#set up the innitial condiction
w1=tf.Variable(np.random.randn(), name="w1") # weight
w2=tf.Variable(np.random.randn(), name="w2")
b=tf.Variable(np.random.randn(), name="b") # bias
def loss(x1,x2,y,w1,w2,b):
    yhat=model(x1,x2,w1,w2,b)
    return tf.reduce_mean(tf.square(y-yhat))
#find the dereivative of loss with respect to weight and bias
def grad01(x1,x2,y,w1,w2,b):
    with tf.GradientTape() as g1:
        lossfn1=loss(x1,x2,y,w1,w2,b)
    return g1.gradient(lossfn1,[w1,b]) #desire to optimize
def grad02(x1,x2,y,w1,w2,b):
    with tf.GradientTape() as g2:
        lossfn2=loss(x1,x2,y,w1,w2,b)
    return g2.gradient(lossfn2,[w2,b]) #desire to optimize
# define training fn
training_step=150
display_step=5 #the interval which shows in display
learning_rate=0.5
# training
grad1=tf.function(grad01)
grad2=tf.function(grad02)
for step in range(training_step):
    sW1,sB1=grad01(X1,X2,Y,w1,w2,b)
    sW2,sB2=grad02(X1,X2,Y,w1,w2,b)
    dW1=sW1*learning_rate
    dW2=sW2*learning_rate
    dB1=sB1*learning_rate
    dB2=sB2*learning_rate
    w1.assign_sub(dW1) # subract dW from w
    w2.assign_sub(dW2)
    b.assign_sub((dB1+dB2)/2) # subract dB from b
    los1=loss(X1,X2,Y,w1,w2,b) #new loss function
    tf.summary.scalar("loss",los1,step=step)
    if step==0 or step % display_step==0:
        print("Loss at step {:02d}: {:.6f}".format(step,los1))
        plt.cla()
        plt.scatter(X1,Y,c='red',label='x1')
        plt.scatter(X2,Y,c='blue',label='x2')
        plt.plot(TrX1, w1*TrX1+w2*TrX2+0.2, 'r-', lw=5)
        plt.text(0.5, 0, 'Step=%d\n Loss=%.5f' %(step, los1), fontdict={'size': 20, 'color':  'green'})
        plt.pause(0.5)
tf.print(w1) # Should be around 2
tf.print(w2) # Should be around 3
tf.print(b) # Should be around 0.2


