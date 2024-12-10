import csv
import math


"""
Abstraction for the DataFrame. Implements all functionality to work with dataframes.
"""
class DataFrame:

    def __init__(self, name, path, dtypes):
        self._name = name
        self._path = path
        self._dtypes = dtypes

        self._read_dataframe()
        self._change_dtypes()
    
    def _read_dataframe(self):
        """
        Reads data from file to dataframe.
        """
        with open(self._path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')
            self._df = [line for line in reader]
    
    def _change_dtypes(self):
        """
        Infers and converts columns in dataframe to correct data type.
        """
        for ri, row in enumerate(self._df[1:]):
            for ci, col in enumerate(row):
                try:
                    col = self._dtypes[ci](col)
                except ValueError as e:
                    col = self._dtypes[ci](0)
                self._df[ri + 1][ci] = col

    def print_columns(self):
        """
        Prints columns of dataframe to the screen.
        """
        print(', '.join(self._df[0]))
    
    def columns(self):
        """
        Retrieves columns of dataframe.

        Returns:
        --------
        columns : list
        """
        return self._df[0]
    
    def info(self):
        """
        Prints the following information about dataset to the screen:
            - Name of dataset
            - Count of columns and observations
            - Column names
        """
        print(f'Dataset: {self._name}\n')
        print(f'Statistics:\n  Columns:\t\t{len(self._df[0])}\n  Observations:\t{len(self._df[1:])}\n')
        print(f'Column names:')
        for t, c in zip(self._dtypes, self._df[0]):
            print(f'  ({t.__name__}) \t{c}')
        print()
    
    def describe_column(self, name=None, index=None):
        """
        Describes a provided column using information specific to its data type and prints this to the screen.
        
        Supported column types:
            - numeric: Number of observations, minimum and maximum value, mean, variance, and standard deviation.
            - boolean: Number of observations, and number of True and False values.
            - string: Number of observations and count of unique values.

        Parameters:
        -----------
        name : str
            Column name in the dataframe. (optional)
        index : int
            Position of the index of the column in the dataframe. (optional)

        Either the name or index must be provided.
        """
        if name:
            try:
                index = self._df[0].index(name)
            except ValueError as e:
                raise Exception(f'Column {name} was not found.')
        
        if index:
            try:
                self._df[0][index]
            except IndexError as e:
                raise Exception(f'Column at index {index} was not found.')
        
        column = self._df[0][index]
        print(f'Column: {column}\n')

        vals = [v[index] for v in self._df[1:]]
        dtype = self._dtypes[index]
        if dtype in (int, float):
            n = len(vals)
            minimum = min(vals)
            maximum = max(vals)
            mean = sum(vals) / n
            variance = sum([math.pow((v - mean), 2) for v in vals]) / n
            sd = math.sqrt(variance)
            print(f'N:\t\t{n}\nMin:\t{minimum}\nMax:\t{maximum}\nAvg.:\t{mean:.02f}\nVar.\t{variance:.02f}\nSD:\t\t{sd:.02f}')

        elif dtype == bool:
            n = len(vals)
            true = sum([v for v in vals if v == 1])
            false = sum([v for v in vals if v == 0])
            print(f'N:\t{n}\nN True:\t{true}\nN False:\t{false}')

        elif dtype == str:
            n = len(vals)
            unique = len(set(vals))
            print(f'N:\t\t\t{n}\nN Unique:\t{unique}')
            
    def describe_columns(self):
        """
        Describes all columns in the dataframe and prints the information to the screen.
        """
        for i, _ in enumerate(self._df[0]):
            self.describe_column(index=i)
            print('\n\n')


"""
Abstraction for the WHO COVID-19 dataframe.
"""
class Covid(DataFrame):
    def __init__(self):
        super().__init__(
            'WHO COVID-19 data',
            'COVID.csv',
            [str, str, int, float, int, int, int, float, int, int, str]
        )


"""
Abstraction for the World Happiness 2019 dataframe.
"""
class WorldHappiness2019(DataFrame):
    def __init__(self):
        super().__init__(
            'World Happiness 2019',
            'happiness.csv',
            [int, str, float, float, float, float, float, float, float]
        )


if __name__ == '__main__':
    ##### INSTRUCTIONS #####
    # Run the program and the following code will test/demonstrate inheritance and polymorphism. Encapsulation was implemented
    # Pythonic way, using leading underscores, as Python does not natively support encapsulation.
    # Test Covid 19 dataframe.
    df = Covid()
    df.print_columns()
    assert len(df.columns()) == 11
    df.info()
    df.describe_column(name='WHO Region')
    df.describe_column(name='Cases - newly reported in last 7 days')
    df.describe_column(index=3)  # Note: uses 0-indexing!
    df.describe_columns()

    # Test World Happiness 2019 dataframe.
    df = WorldHappiness2019()
    df.print_columns()
    assert len(df.columns()) == 9
    df.info()
    df.describe_column(name='Country or region')
    df.describe_column(name='Score')
    df.describe_column(index=0)  # Note: uses 0-indexing!
    df.describe_columns()
