import sys

class Archive:
    """
    A class to read and write training data in archives.
    """

    def __init__(self, path):
        """
        Creates a new ArchiveHandler

        :param path: The path to the archive directory.
        :type path: str
        """

        self._path = path


    def read(self):
        """
        Reads an archive and returns the data.

        :return: A Dataset object.
        :rtype: io.Dataset
        """
