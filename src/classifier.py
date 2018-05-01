#!/usr/bin/env python3

import tensorflow as tf

MODEL_DIR = "model/classifier"
NO_FILTERS = 11
INPUT_SHAPE = (40, 40)
KERNEL_SIZE = (10, 10)
POOL_RADIUS = 2
NO_LABELS = 2
BATCH_SIZE = 1
IMAGE_SIZE = (50, 50)

def model_fn(features, labels, mode):
	l1 = tf.reshape(features["x"], [-1, INPUT_SHAPE[0], INPUT_SHAPE[1], 3])
	l2 = tf.layers.conv2d(inputs = l1, filters = NO_FILTERS, kernel_size = KERNEL_SIZE)
	l3 = tf.layers.max_pooling2d(inputs = l2, poolsize=[POOL_RADIUS]*2, strides=POOL_RADIUS)
	l4 = tf.layers.conv2d(inputs = l3, filters = NO_FILTERS, kernel_size = KERNEL_SIZE)
	l5 = tf.layers.dense(inputs = l4, units=100, activation=tf.nn.relu)
	l6 = tf.layers.dense(inputs = l5, units=1, activation=tf.nn.relu)
	out = l6

	assert(out.shape == (BATCH_SIZE, NO_LABELS))
	assert(labels.shape == (BATCH_SIZE,))

	predictions = {
		"classes": tf.argmax(input=out, axis=1),
		"probabilities": tf.nn.softmax(out, name="softmax_tensor"),
	}

	loss = tf.reduce_sum(tf.square(out - tf.eye(NO_LABELS)[labels]))

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
	assert(xdata.shape == (BATCH_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1]))

	predict_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": xdata}
	)

	return est.predict(
		input_fn=predict_input_fn,
	)

def train(xdata, ydata):
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": xdata},
		y=ydata,
		batch_size=BATCH_SIZE,
		num_epochs=None,
		shuffle=True
	)

	est.train(
		input_fn=train_input_fn,
		steps=2000
	)
