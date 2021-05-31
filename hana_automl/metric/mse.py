from decimal import Decimal

from hana_ml import DataFrame

# locale.setlocale(locale.LC_ALL, "USA")


def mse_score(
    algo=None, df: DataFrame = None, target=None, ftr: list = None, id: str = None
):
    if algo is not None:
        res = algo.predict(df, id, ftr)
        if type(res) == tuple:
            res = res[0]
        if (
            str(algo).split(" ")[0]
            == "<hana_ml.algorithms.pal.neural_network.MLPRegressor"
        ):
            id_val = 2
        else:
            id_val = 1
        res = (
            res.select([res.columns[0], res.columns[id_val]])
            .join(
                df.select([id, target]).rename_columns(["ID_TEMP", target]),
                "ID_TEMP=" + id,
            )
            .deselect("ID_TEMP")
        )
        pandas = res.collect()
        cols = res.columns
        pandas["mse_coef"] = pandas.apply(
            lambda row: val(row[cols[2]], row[cols[1]]), axis=1
        )
        return pandas["mse_coef"].mean()
    else:
        pandas = df.collect()
        cols = df.columns
        pandas["mse_coef"] = pandas.apply(
            lambda row: (Decimal(row[cols[0]]) - Decimal(row[cols[1]])) ** 2, axis=1
        )
        return pandas["mse_coef"].mean()


def val(a, b):
    if type(a) is not Decimal:
        a = Decimal(a)
    if type(b) is not Decimal:
        if type(b) is not str:
            b = str(b)
        b = Decimal(b)
    return (a - b) ** 2
