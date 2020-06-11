#data generation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
trX=np.linspace(-1,1,101) #切100格
#mock data
trY=2*trX+np.random.randn(*trX.shape)*0.4+0.2 #weight 0.4 bias=0.2
#convert to tensor form
X=tf.convert_to_tensor(trX,dtype=tf.float32,name="x")
Y=tf.convert_to_tensor(trY,dtype=tf.float32,name="y")
#plt all the thing
plt.figure()
plt.scatter(trX,trY)
plt.plot(trX, 0.2+ 2*trX )
plt.show()

#define the neuron
def model(X, w, b):
 return tf.multiply(X,w)+b # define line as X * w + b
w=tf.Variable(np.random.randn(), name="w") # weight
b=tf.Variable(np.random.randn(), name="b") # bias
def loss(x,y,w,b):
    yhat=model(x,w,b)
    return tf.reduce_mean(tf.square(y-yhat))
#find the dereivative of loss with respect to weight and bias
def grad0(x,y,w,b):
    with tf.GradientTape() as g:
        lossfn=loss(x,y,w,b)
    return g.gradient(lossfn,[w,b]) #desire to optimize
# define training fn
training_step=100 
display_step=10 #the interval which shows in display
learning_rate=0.5
# training
grad=tf.function(grad0)
for step in range(training_step):
    sW,sB=grad(X,Y,w,b)
    dW=sW*learning_rate
    dB=sB*learning_rate
    w.assign_sub(dW) # subract dW from w
    b.assign_sub(dB) # subract dB from b
    los1=loss(X,Y,w,b) #new loss function
    tf.summary.scalar("loss",los1,step=step)
    if step==0 or step % display_step==0:
        print("Loss at step {:02d}: {:.6f}".format(step,los1))
        plt.plot (trX, b + trX * w)
        plt.show()
tf.print(w) # Should be around 2
tf.print(b) # Should be around 0.2