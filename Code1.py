
import tensorflow as tf
print(tf.__version__)


if __name__ == '__main__':

    A = tf.constant([[1, 2], [3, 4]])
    B = tf.constant([[5, 6], [7, 8]])
    C = tf.matmul(A, B)

    print(C)
