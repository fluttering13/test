import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles

N=210
# Max No. of iterations, if the conditions are not met
MAX_ITERS = 1000
#to spit the point of the test and training data
cut=int(N*0.7) 
start = time.time()
#scatter the ponints like the circle
data, features = make_circles(n_samples=N,shuffle=True, noise= 0.12, factor=0.4)
tr_data, tr_features = data[:cut], features[:cut]
te_data, te_features = data[cut:], features[cut:]
fig, ax = plt.subplots()
ax.scatter(tr_data.transpose()[0],tr_data.transpose([1],marker = 'o', s = 100, c = tr_features,cmap = plt.cm.coolwarm)
plt.plot(tr_data.transpose()[0],tr_data.transpose([1],marker = 'o', s = 100, c = tr_features,cmap = plt.cm.coolwarm)
plt.show()
test=[]
#test.append is how we can asign the artribution
for i in te_data:
 Distances = tf.math.reduce_sum(tf.square(tf.subtract(i, tr_data)), axis=1)
 neighbor = tf.argmin(distances,0)
 test.append(tr_features[neighbor])
fig, ax = plt.subplots()
ax.scatter(te_data.transpose()[0], te_data.transpose()[1],marker = 'o', s = 100, c = test, cmap=plt.cm.coolwarm )
plt.plot()
plt.show()
end = time.time()
print ("Found in %.2f seconds" % (end-start))
print ("Cluster assignments:", test)