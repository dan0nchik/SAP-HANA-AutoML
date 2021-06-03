from hana_ml import DataFrame


def mse_score(
    algo=None,
    df: DataFrame = None,
    target: str = None,
    ftr: list = None,
    id: str = None,
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
        res = res.rename_columns(["ID", "PREDICTION", "REAL"]).select(
            ("AVG((REAL - PREDICTION)*(REAL - PREDICTION))", "VAL")
        )
        pandas = res.collect()
        return pandas.VAL[0]
    else:
        res = df.rename_columns(["REAL", "PREDICTION"]).select(
            ("AVG((REAL - PREDICTION)*(REAL - PREDICTION))", "VAL")
        )
        pandas = res.collect()
        return pandas.VAL[0]
