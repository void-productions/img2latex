#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from constants import CLASSIFIER_INPUT_SHAPE, CHARS

MODEL_DIR = "model/classifier"
NO_FILTERS = 11
KERNEL_SIZE = (10, 10)
POOL_RADIUS = 2
BATCH_SIZE = 1

def model_fn(features, labels, mode):
	l1 = tf.reshape(features["x"], [-1, *CLASSIFIER_INPUT_SHAPE, 3])
	l2 = tf.layers.conv2d(inputs = l1, filters = NO_FILTERS, kernel_size = KERNEL_SIZE)
	l2_out_shape = (
		CLASSIFIER_INPUT_SHAPE[0] - KERNEL_SIZE[0] + 1,
		CLASSIFIER_INPUT_SHAPE[1] - KERNEL_SIZE[1] + 1,
	)
	assert(l2.shape == (BATCH_SIZE, *l2_out_shape, NO_FILTERS))
	l3 = tf.layers.max_pooling2d(inputs = l2, pool_size=[POOL_RADIUS]*2, strides=POOL_RADIUS)
	l3_out_shape = (
		np.floor(l2_out_shape[0]/POOL_RADIUS),
		np.floor(l2_out_shape[1]/POOL_RADIUS)
	)
	assert(l3.shape == (BATCH_SIZE, *l3_out_shape, NO_FILTERS))
	l4 = tf.layers.conv2d(inputs = l3, filters = NO_FILTERS, kernel_size = KERNEL_SIZE)
	l5 = tf.reshape(l4, shape=[BATCH_SIZE, -1])
	l6 = tf.layers.dense(inputs = l5, units=100, activation=tf.nn.relu)
	l7 = tf.layers.dense(inputs = l6, units=len(CHARS), activation=tf.nn.relu)
	out = l7

	assert(out.shape == (BATCH_SIZE, len(CHARS)))
	assert(labels.shape == (BATCH_SIZE, len(CHARS)))

	predictions = {
		"classes": tf.argmax(input=out, axis=1),
		"probabilities": tf.nn.softmax(out, name="softmax_tensor"),
	}

	loss = tf.reduce_sum(tf.square(out - labels))

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	elif mode == tf.estimator.ModeKeys.TRAIN:
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
	assert(xdata.shape == (BATCH_SIZE, *CLASSIFIER_INPUT_SHAPE))

	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": xdata}
	)

	return est.predict(
		input_fn=predict_input_fn,
	)

def train(xdata, ydata):
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
