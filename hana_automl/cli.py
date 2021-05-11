import click
from hana_automl.automl import AutoML
from hana_automl.pipeline.input import Input
import numpy as np


@click.command()
@click.option("-i", help="Path or URL of file to be processed.")
@click.option("--target", help="Column or variable to be predicted")
@click.option("--table", default=None, help="Name of existing table created in HANA")
@click.option("--rm_columns", help="Columns in the dataframe to be removed")
@click.option("--categorical", help="Categorical features or columns in the dataframe")
@click.option("--steps", default=10, help="Specify the number of iterations")
@click.option("--id_column", default=None, help="ID column in table")
@click.option(
    "--optimizer",
    default="BayesianOptimizer",
    help="Optimizer that will find the best algorithm",
)
@click.option("--wizard", default=False, help="Interactive mode. Best for beginners")
def start(
    i, target, rm_columns, categorical, id_column, optimizer, steps, wizard, table
):
    if wizard:
        wizard_mode()
        return
    automl = AutoML()
    automl.fit(
        file_path=i,
        target=target,
        columns_to_remove=rm_columns,
        categorical_features=categorical,
        id_column=id_column,
        optimizer=optimizer,
        steps=steps,
        table_name=table,
    )


def wizard_mode():
    file_path = input(
        "Welcome to the wizard mode! It will guide you through the whole AutoML process. Let's start with an input "
        "file. It can be a URl or a path. Press [Enter] to proceed with default file: "
    )
    df = Input.download_data("data/train.csv" if file_path == "" else file_path)

    print(f"Here's your dataframe: \n{df.head()}")
    print(f"Its columns:")
    col_index = 0
    col_list = None
    for col in df.columns:
        print(f"[{col_index}]", col)
        col_index += 1
    target = df.columns[
        int(
            input(
                "Enter number of column(s) that you want to predict. (Example: 3,4 or 5): "
            )
        )
    ]

    print("HANA needs ID column to work properly")
    col_index = 0
    for col in df.columns:
        print(f"[{col_index}]", col)
        col_index += 1
    id_column = df.columns[
        int(input("Enter number of column to set as ID column. (Example: 3,4 or 5): "))
    ]

    rm_col = input("Do you want to remove any columns? y|n: ")
    if rm_col == "yes" or rm_col == "y":
        print("Here are the columns:")
        col_index = 0
        for col in df.columns:
            print(f"[{col_index}]", col)
            col_index += 1
        col = input("Enter number of columns to delete. (Example: 1,4,5): ")
        col_list = []
        for i in col.split(","):
            col_list.append(df.columns[int(i)])
        print(f"OK. Columns {col_list} will be deleted later.")

    print(
        "We've automatically detected categorical (string, date, object, etc) columns:"
    )
    col_index = 0
    for col in df.columns:
        print(
            f"[{col_index}]",
            "categ."
            if df[col].dtype == str or df[col].dtype == np.object_
            else "normal",
            f"dtype: {df[col].dtype}",
            "name: ",
            col,
        )
        col_index += 1

    cat_col = input("Enter categorical columns to confirm. (Example: 3,4,5): ")
    cat_list = []
    for i in cat_col.split(","):
        cat_list.append(df.columns[int(i)])

    table = input(
        "Great. Now we need to load your data to HANA. Enter name if you have existing table ([Enter] for "
        "none): "
    )

    print("Starting automated machine learning...")
    automl = AutoML()
    automl.fit(
        df,
        target=target,
        table_name=table,
        columns_to_remove=col_list,
        categorical_features=cat_list,
        id_column=id_column,
    )


if __name__ == "__main__":
    start()
