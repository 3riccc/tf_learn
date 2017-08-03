import tensorflow as tf
# import numpy as np

matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])

pdt = tf.matmul(matrix1,matrix2)

# sess = tf.Session()
# res = sess.run(pdt)
# print(res)
# sess.close()

with tf.Session() as sess:
	with tf.device("/gpu:0"):
		res = sess.run(pdt)
		print(res)