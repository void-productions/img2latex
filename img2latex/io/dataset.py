import abc
import sys

class Dataset:
	"""
	A container for data and labels.
	"""

	def __init__(self, data, labels):
		"""
		Initiates a Dataset.
		data.shape[0] == labels.shape[0]

		:param data: A numpy array.
		:type data: numpy.ndarray
		:param labels: A numpy array.
		:type labels: numpy.ndarray
		"""

		self._data = data
		self._labels = labels

	def get_data(self):
		"""
		Returns the data of this dataset.

		:return: The data of this dataset
		:rtype: numpy.ndarray
		"""

		return self._data

	def get_labels(self):
		"""
		Returns the labels of this dataset.

		:return: The labels of this dataset
		:rtype: numpy.ndarray
		"""

		return self._labels

	def validate(data, labels):
		"""
		Throws an exception, if the dataset is not valid
		:return:
		"""




class ClassifierDataset(Dataset):
	"""

	"""
