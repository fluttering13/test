import tensorflow as tf
import os #忽略CPU AVX2訊息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
a = tf.constant(2,dtype=tf.float32) #統一變數型態為float32
b = tf.constant(3,dtype=tf.float32)
c = tf.multiply(a,b)
x = np.sin(2*3,dtype=np.float32) #統一變數型態為float32
d = tf.constant(x,dtype=np.float32) #轉換X至tensorflow
e = tf.divide(b,d)
tf.print(a,b,c,d,e)