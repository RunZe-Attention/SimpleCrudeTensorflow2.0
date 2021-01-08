import tensorflow as tf
print(tf.__version__)
# 张量的重要属性就行形状 类型 值
random_float = tf.random.uniform(shape=())
zero_vector = tf.zeros(shape=(2))

A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])

if __name__ == '__main__':
    print(random_float)
    print(zero_vector)
    print(A.numpy())
    print(tf.add(A,B))
    print(tf.matmul(A,B))# 矩阵标准乘法
