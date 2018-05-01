import os

def rootdir():
	return os.path.realpath(
		os.path.join(
			os.path.dirname(__file__),
			os.path.pardir
		)
	)

def resdir():
	return os.path.join(rootdir(), "res")
