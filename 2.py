import tensorflow as tf
import os #忽略CPU AVX2訊息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
a=tf.zeros([2, 3], tf.int32)
b=tf.zeros_like(a, dtype=None, name=None) #我要一個零矩陣維度跟指定的變數一樣
c=tf.fill([2, 3], 8)
d=tf.linspace(10.0, 13.0, 10) #下限 上限 切幾塊
e=tf.range(1, limit=10, delta=1, dtype=None,name='range') #下陷 上限 一刀切多少
a1=tf.constant([[1,2,3],[4,5,6],[7,8,9]], tf.float32)
a2=tf.constant([[1,2,3],[4,5,6],[7,8,9]], tf.float32)
f=tf.matmul(a1,a2)
g=tf.linalg.det(f).numpy() #det必須要float32的格式

print(g)
