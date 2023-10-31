import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

ten=tf.constant([[[1,2],[2,3]],[[3,4],[5,6]]])

sess=tf.Session()

print(sess.run(ten)[1,0,0])

sess.close()