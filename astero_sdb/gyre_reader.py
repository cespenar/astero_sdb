import numpy as np

class GyreData():

    """Structure containing data from a GYRE output file.

        Assumes the following structure of a file:

        line 1: blank
        line 2: header numbers
        line 3: header names
        line 4: header data
        line 5: body numbers
        line 6: body names
        lines 7-: body data

    Inspired by Bill Wolf's PyMesaReader:
    https://github.com/wmwolf/py_mesa_reader

    Parameters
    ----------
    file_name : str
        The name of GYRE output file to be read in.

    Attributes
    ----------
    file_name : str
        Path to the GYRE output file.
    bulk_data : numpy.ndarray
        The main data in the structured array format.
    bulk_names : tuple of str
        List of all available data column names.
    header_data : dict
        Header data in dict format.
    header_names : list of str
        List of all available header names.
    """

    header_names_line = 3
    body_names_line = 6

    @classmethod
    def set_header_names_line(cls, name_line=2):
        cls.header_names_line = name_line

    @classmethod
    def set_body_names_line(cls, name_line=6):
        cls.bulk_names_line = name_line

    def __init__(self, file_name):
        """Make a GyreData object from a GYRE output file.

        Assumes the following structure of a file:

        line 1: blank
        line 2: header numbers
        line 3: header names
        line 4: header data
        line 5: body numbers
        line 6: body names
        lines 7-: body data

        This structure can be altered by the class methods
        'GyreData.set_header_names_line' and 'GyreData.set_body_names_line'.

        Parameters
        ----------
        file_name : str
            The name of GYRE output file to be read in.
        """

        self.file_name = file_name
        self.header_names = None
        self.header_data = None
        self.body_names = None
        self.body_data = None
        self.read_gyre()

    def read_gyre(self):
        """Reads data from a GYRE output file.
        
        Parameters
        ----------

        Returns
        ----------
        
        """

        self.body_data = np.genfromtxt(
            self.file_name, skip_header=GyreData.body_names_line - 1,
            names=True, dtype=None)

        self.body_names = self.body_data.dtype.names

        header_data = []
        with open(self.file_name) as f:
            for i, line in enumerate(f):
                if i == GyreData.header_names_line - 1:
                    self.header_names = line.split()
                elif i == GyreData.header_names_line:
                    header_data = [float(v) for v in line.split()]
                elif i > GyreData.header_names_line:
                    break
        self.header_data = dict(zip(self.header_names, header_data))

    def is_in_header(self, key):
        """Determine if 'key' exists in header data.

        Parameters
        ----------
        key : str
            The string to test if it is in header names.

        Returns
        ----------
        bool
            True if 'key' is in header names, otherwise False.
        """
        return key in self.header_names

    def is_in_data(self, key):
        """Determine if 'key' exists in body data.

        Parameters
        ----------
        key : str
            The string to test if it is a valid column name.

        Returns
        -------
        bool
            True if 'key' is a valid column name, otherwise False.
        """
        return key in self.body_names

    def header(self, key):
        """Returns a header value for 'key'.

        Parameters
        ----------
        key : str
            The name of in header.

        Returns
        ----------
        numpy.ndarray
            A value for the name 'key'.

        Raises
        ----------
        KeyError
            If 'key' is an invalid key.
        """

        if self.is_in_header(key):
            return self.header_data[key]
        else:
            raise KeyError(f"{key:s} is not a valid data type")

    def data(self, key):
        """Returns numpy array with the data column for 'key'.

        Parameters
        ----------
        key : str
            The name of data column.

        Returns
        ----------
        numpy.ndarray
            An array with data correspoding to the name 'key'.

        Raises
        ----------
        KeyError
            If 'key' is an invalid key.
        """

        if self.is_in_data(key):
            return self.body_data[key]
        else:
            raise KeyError(f"{key:s} is not a valid data type")
