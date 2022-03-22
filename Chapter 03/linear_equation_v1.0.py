import tensorflow as tf

# linear equation is represented by
# ax + b = y
# y - ax = b
# y/b - a/b(x) = 1
# x, y is known and can be represented as X matrix, goal is to find a and b
# therefore we can represent the above goal as AX = B where X is the input matrix, A is the unknown(a,b) and B is all ones

# example
# 3x+2y = 15
# 4x−y = 10

# equation 1
x1 = tf.constant(3, dtype=tf.float32)
y1 = tf.constant(2, dtype=tf.float32)
point1 = tf.stack([x1, y1])

# equation 2
x2 = tf.constant(4, dtype=tf.float32)
y2 = tf.constant(-1, dtype=tf.float32)
point2 = tf.stack([x2, y2])

# solve for AX=C
X = tf.transpose(tf.stack([point1, point2]))
C = tf.ones((1,2), dtype=tf.float32)

A = tf.matmul(C, tf.matrix_inverse(X))

with tf.Session() as sess:
    X = sess.run(X)
    print(X)

    A = sess.run(A)
    print(A)
