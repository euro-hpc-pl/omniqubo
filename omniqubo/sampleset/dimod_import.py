from dimod import SampleSet
from pandas import DataFrame


def dimod_import(data: SampleSet) -> DataFrame:
    """Transforms SampleSet of dimod into DataFrame

    The DataFrame also consist of extra column "feasible", set to True for all
    samples.

    :param data: optimization samples from dimod
    :return: the same optimization samples in DataFrame
    """
    df = data.to_pandas_dataframe()
    return df
