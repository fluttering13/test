import random
import numpy
N0 = 3 # input layer size
N1 = 2 # output layer size 
w = numpy.random.uniform(low=-1,high=+1,size=(N1,N0))
print("w=",w)
b = numpy.random.uniform(low=-1,high=+1,size=N1) #bias is equal to output size
print("b=",b)
y_in = numpy.array([0.2, 0.4, -0.1]) # input values
z = numpy.dot(w, y_in) + b
print("z=",z)
y_out = 1/(1+numpy.exp(-z)) # the sigmoid function
print("y=",y_out)

