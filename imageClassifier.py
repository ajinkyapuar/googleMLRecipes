# https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc#0

# https://github.com/random-forests/tutorials/blob/master/ep7.ipynb

# http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf


############################################################################################################################################

# inception : one of googles best image classifiers

# we will be using inception and retraining it (transfer learning)

############################################################################################################################################

# curl -O http://download.tensorflow.org/example_images/flower_photos.tgz

# tar xzf flower_photos.tgz

# docker run -it gcr.io/tensorflow/tensorflow:latest-devel

# docker run -it -v ~/Documents/Python/googleMLRecipes/tf_files:/tf_files  gcr.io/tensorflow/tensorflow:latest-devel

# ls /tf_files/

# cd /tensorflow

# git pull

#  In Docker

# python tensorflow/examples/image_retraining/retrain.py \
# --bottleneck_dir=/tf_files/bottlenecks \
# --how_many_training_steps 500  #quick train
# --model_dir=/tf_files/inception \
# --output_graph=/tf_files/retrained_graph.pb \
# --output_labels=/tf_files/retrained_labels.txt \
# --image_dir /tf_files/flower_photos

# curl -L https://goo.gl/tx3dqg > ~/Documents/Python/googleMLRecipes/tf_files/label_image.py

############################################################################################################################################

# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))

############################################################################################################################################

