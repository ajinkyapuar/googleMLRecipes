# https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/?utm_campaign=chrome_series_machinelearning_063016&utm_source=gdev&utm_medium=yt-desc#0

# https://github.com/random-forests/tutorials/blob/master/ep7.ipynb

# http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf


############################################################################################################################################

# inception : one of googles best image classifiers

# we will be using inception and retraining it (transfer learning)

############################################################################################################################################

# import tensorflow as tf
# hello = tf.constant('Hello, TensorFlow!')
# sess = tf.Session()
# print(sess.run(hello))

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
#quick train
# --how_many_training_steps 500
#quick train
# --model_dir=/tf_files/inception \
# --output_graph=/tf_files/retrained_graph.pb \
# --output_labels=/tf_files/retrained_labels.txt \
# --image_dir /tf_files/flower_photos

# Exit Docker CLI
# curl -L https://goo.gl/tx3dqg > ~/Documents/Python/googleMLRecipes/tf_files/label_image.py

#  In Docker
# python /tf_files/label_image.py /tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg

############################################################################################################################################

# Test
# python /tf_files/label_image.py /tf_files/flower_photos/asdf.jpg
# tulips (score = 0.86611)
# sunflowers (score = 0.11194)
# roses (score = 0.01422)
# dandelion (score = 0.00414)
# daisy (score = 0.00359)

############################################################################################################################################

# 6. Optional Step: Trying Other Hyperparameters
#
# There are several other parameters you can try adjusting to see if they help your results. The --learning_rate controls the magnitude of the updates to the final layer during training. If this rate is smaller, the learning will take longer, but it can help the overall precision. That's not always the case, though, so you need to experiment carefully to see what works for your case.
#
# The --train_batch_size parameter controls the number of images that the script examines during one training step. Because the learning rate is applied per batch, you'll need to reduce this value if you have larger batches to get the same overall effect.

############################################################################################################################################

# 7. Optional Step: Training on Your Own Categories
#
# After you see the script working on the flower example images, you can start looking at teaching the network to recognize categories you care about instead.
#
# In theory, all you need to do is run the tool, specifying a particular set of sub-folders. Each sub-folder is named after one of your categories and contains only images from that category.
#
# If you complete this step and pass the root folder of the subdirectories as the argument for the --image_dir parameter, the script should train the images that you've provided, just like it did for the flowers.
#
# The classification script uses the folder names as label names, and the images inside each folder should be pictures that correspond to that label, as you can see in the flower archive:
#
#
#
# Collect as many pictures of each label as you can and try it out!

############################################################################################################################################

# Tensorflow
# fit
# predict
# evlauate

############################################################################################################################################
