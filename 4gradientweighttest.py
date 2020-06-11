
#define the weight
# if you need to call tape.gradient() more than once
# use GradientTape(persistent=True)
weight1 = tf.Variable(2.0)
weight2 = tf.Variable(3.0)
weight3 = tf.Variable(5.0)
def weighted_sum(x1, x2, x3):
 return weight1*x1 + weight2*x2 + weight3*x3
with tf.GradientTape(persistent=True) as tape:
 sum = weighted_sum(7.,5.,6.)
[weight1_grad] = tape.gradient(sum, [weight1])
[weight2_grad] = tape.gradient(sum, [weight2])
[weight3_grad] = tape.gradient(sum, [weight3])
print(weight1_grad.numpy()) #7.0
print(weight2_grad.numpy()) #5.0
print(weight3_grad.numpy()) #6.0