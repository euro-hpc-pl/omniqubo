from dimod import SampleSet
from pandas import DataFrame


def dimod_import(data: SampleSet) -> DataFrame:
    df = data.to_pandas_dataframe()
    df["feasible"] = True
    return df
