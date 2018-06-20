#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from img2latex.constants import CLASSIFIER_INPUT_SHAPE, CHARS


MODEL_DIR = "model/classifier"
NO_FILTERS = 11
KERNEL_SIZE = (10, 10)
POOL_RADIUS = 2
BATCH_SIZE = 1


def model_fn(features, labels, mode):
	# input layer
	input_layer = tf.reshape(features["x"], [-1, *CLASSIFIER_INPUT_SHAPE, 3])

	# 1 convolutional
	convolutional_1 = tf.layers.conv2d(inputs = input_layer, filters = NO_FILTERS, kernel_size = KERNEL_SIZE)

	convolutional_1_out_shape = (
		-1,
		CLASSIFIER_INPUT_SHAPE[0] - KERNEL_SIZE[0] + 1,
		CLASSIFIER_INPUT_SHAPE[1] - KERNEL_SIZE[1] + 1,
		NO_FILTERS
	)

	assert(convolutional_1.shape[1:] == convolutional_1_out_shape[1:])

	# 2 pooling
	pooling_2 = tf.layers.max_pooling2d(inputs = convolutional_1, pool_size=[POOL_RADIUS]*2, strides=POOL_RADIUS)
	pooling_2_out_shape = (
		-1,
		int(np.floor(convolutional_1_out_shape[1]/POOL_RADIUS)),
		int(np.floor(convolutional_1_out_shape[2]/POOL_RADIUS)),
		NO_FILTERS
	)

	assert(pooling_2.shape[1:] == pooling_2_out_shape[1:])

	# 3 convolutional
	convolutional_3 = tf.layers.conv2d(inputs = pooling_2, filters = NO_FILTERS, kernel_size = KERNEL_SIZE)

	convolutional_3_out_shape = (
		-1,
		int(np.floor(pooling_2_out_shape[1] - KERNEL_SIZE[0] + 1)),
		int(np.floor(pooling_2_out_shape[2] - KERNEL_SIZE[1] + 1)),
		NO_FILTERS
	)

	assert(convolutional_3.shape[1:] == convolutional_3_out_shape[1:])

	# 4 flat
	flat_4 = tf.reshape(convolutional_3, shape=[-1, np.prod(convolutional_3_out_shape[1:])])

	# 5 dense
	dense_5 = tf.layers.dense(inputs = flat_4, units=100, activation=tf.nn.relu)

	# 6 dense
	dense_6 = tf.layers.dense(inputs = dense_5, units=len(CHARS), activation=tf.nn.sigmoid)
	out = dense_6

	assert(out.shape[1:] == (len(CHARS),))
	if labels is not None:
		assert(labels.shape[1:] == (len(CHARS),))

	predictions = {
		"classes": tf.argmax(input=out, axis=1),
		"probabilities": tf.nn.softmax(out, name="softmax_tensor"),
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	loss = tf.reduce_sum(tf.square(out - labels))

	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0002)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step()
		)
		return tf.estimator.EstimatorSpec(mode=mode, train_op=train_op, loss=loss)
	elif mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss
		)


est = tf.estimator.Estimator(model_fn, model_dir=MODEL_DIR)


def predict(xdata):
	"""
	:param xdata: np.ndarray of shape (number_of_samples, height, width, depth)
			where number_of_samples is the number of images
			where height/width is the height/width of the images
			where depth is the colordepth of the image
	:type xdata: np.ndarray
	:return: np.ndarray of shape (number_of_samples,), named ydata
		for n in range(number_of_samples): CHARS[ydata[n]] == label of xdata[n]
	:rtype: np.ndarray
	"""
	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": xdata},
		shuffle=False
	)

	return est.predict(
		input_fn=predict_input_fn,
	)


def train(xdata, ydata):
	"""
	:param xdata: np.ndarray of shape (number_of_samples, height, width, depth)
		where number_of_samples is the number of images
		where height/width is the height/width of the images
		where depth is the colordepth of the image
	:type xdata: np.ndarray
	:param ydata: np.ndarray of shape (number_of_samples,)
		for n in range(number_of_samples): CHARS[ydata[n]] == label of xdata[n]
	:type ydata: np.ndarray
	"""

	y = np.zeros((len(ydata), len(CHARS)), dtype=np.float32)
	for i in range(len(ydata)):
		y[i,ydata[i]] = 1

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": xdata},
		y=y,
		batch_size=BATCH_SIZE,
		num_epochs=None,
		shuffle=True
	)

	est.train(
		input_fn=train_input_fn,
		steps=2000
	)
