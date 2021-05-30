from decimal import Decimal

from hana_ml import DataFrame


def mse_score(algo=None, df: DataFrame = None, target=None, ftr: list = None, id: str = None):
    if algo is not None:
        res = algo.predict(df, id, ftr)
        if type(res) == tuple:
            res = res[0]
        res = res.select([res.columns[0], res.columns[1]]).join(
            df.select([id, target]).rename_columns(["ID_TEMP", target]),
            "ID_TEMP=" + id).deselect("ID_TEMP")
        res = res.cast(res.columns[1], "DOUBLE")
        pandas = res.collect()
        cols = res.columns
        pandas['mse_coef'] = pandas.apply(lambda row: (row[cols[2]] - row[cols[1]]) ** 2, axis=1)
        return pandas['mse_coef'].mean()
    else:
        pandas = df.collect()
        cols = df.columns
        pandas['mse_coef'] = pandas.apply(lambda row: (Decimal(row[cols[0]]) - Decimal(row[cols[1]])) ** 2, axis=1)
        return pandas['mse_coef'].mean()
