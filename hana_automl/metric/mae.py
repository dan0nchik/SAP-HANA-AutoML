from decimal import Decimal

from hana_ml import DataFrame


def mae_score(algo=None, df: DataFrame = None, target=None, ftr: list = None, id: str = None):
    if algo is not None:
        res = algo.predict(df, id, ftr)
        if type(res) == tuple:
            res = res[0]
        res = res.select([res.columns[0], res.columns[1]]).join(
            df.select([id, target]).rename_columns(["ID_TEMP", target]),
            "ID_TEMP=" + id).deselect("ID_TEMP")
        pandas = res.collect()
        cols = res.columns
        pandas['mae_coef'] = pandas.apply(lambda row: abs(Decimal(row[cols[2]]) - Decimal(row[cols[1]])), axis=1)
        return pandas['mae_coef'].mean()
    else:
        pandas = df.collect()
        cols = df.columns
        pandas['mae_coef'] = pandas.apply(lambda row: abs(Decimal(row[cols[0]]) - Decimal(row[cols[1]])), axis=1)
        return pandas['mae_coef'].mean()
