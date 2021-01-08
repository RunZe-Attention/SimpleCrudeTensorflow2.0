# 自动求导机制
import tensorflow as tf

# 为什么要用变量而不是张量
# 因为变量能够TF自动求导机制求导，因此用来定义机器学习的模型参数
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    y = tf.square(x)
y_grad = tape.gradient(y,x)
print(y,y_grad)

# 多元函数 Z = X^2 + Y^2 多个未知数
X = tf.constant([[1.,2.],[3.,4.]])
y = tf.constant([[1.],[2.]])
w = tf.Variable(initial_value=[[1.],[2.]])
b = tf.Variable(initial_value=1.)
with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X,w)+b - y))

w_grad,b_grad = tape.gradient(L,[w,b])
print("Here")
print("L.numpy() : {}".format(L.numpy()))
print(w_grad.numpy())
print(b_grad.numpy())


