from time import time
import pandas as pd

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        print(f'Time taken to run {func.__name__}: {end-start} seconds')
    return wrapper

def printShape(func):
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        print(f'Shape of {func.__name__} output: ', end=' ')
        if type(df) is not pd.DataFrame:
            for output in df:
                print(output.shape, end=' ')
            print()
        else:
            print(df.shape)
        return df
    return wrapper
