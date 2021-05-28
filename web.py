import sys
from contextlib import contextmanager
from io import StringIO
from threading import current_thread

import hdbcli
import optuna.visualization as optuna_vs
import pandas as pd
import streamlit as st
from hana_ml.dataframe import ConnectionContext, create_dataframe_from_pandas
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME

from hana_automl.automl import AutoML
from hana_automl.pipeline.data import Data
from hana_automl.storage import Storage
from web.session import session_state


# from https://discuss.streamlit.io/t/cannot-print-the-terminal-output-in-streamlit/6602/2
@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield


st.title("Welcome to SAP HANA AutoML!")

st.sidebar.title("1. Enter your HANA database credentials:")
user = st.sidebar.text_input(label="Username", value="DEVELOPER")
password = st.sidebar.text_input(
    label="Password", type="password", value="8wGGdQhjwxJtKCYhO5cI3"
)
host = st.sidebar.text_input(label="Host", value="localhost")
port = st.sidebar.text_input(label="Port", value="39015")

df = None
automl = None
columns = []
id_col = None
test_id_col = None
test_df = None
no_id_msg = "I don't have it"


@st.cache
def get_database_connection():
    return host, int(port), user, password


@st.cache(
    allow_output_mutation=True, hash_funcs={hdbcli.dbapi.Connection: lambda _: None}
)
def cache_automl():
    return automl


if st.sidebar.button(label="Submit"):
    if user != "" and password != "" and host != "" and port != "":
        get_database_connection()
        st.success("Successfully connected to the database!")

st.sidebar.markdown("# 2. Load data")
st.sidebar.markdown("## From file:")
uploaded_file = st.sidebar.file_uploader(label="", type=["csv", "xlsx"])
st.sidebar.write("*(Optional)* provide table name to load dataset there:")
table_name = st.sidebar.text_input(label="", value=None)
if table_name == "None" or table_name == "":
    table_name = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.text("Here is the head of your dataset:")
    st.dataframe(df.head(10))

st.sidebar.title("3. Choose task:")
chosen_task = st.sidebar.selectbox(
    "", ["Determine task for me", "Classification", "Regression"]
)
task = None
if chosen_task == "Classification":
    task = "cls"
if chosen_task == "Regression":
    task = "reg"

st.sidebar.title("4. Select columns to remove:")
if df is not None:
    columns_to_rm = st.sidebar.multiselect("", df.columns)
else:
    st.sidebar.write("Load dataset first")

st.sidebar.title("5. Select categorical features:")
if df is not None:
    categorical = st.sidebar.multiselect(
        "",
        df.columns,
        key="ftr",
    )
else:
    st.sidebar.write("Load dataset first")

st.sidebar.title("6. Select target variable:")
st.sidebar.write("*It is a column to predict*")
if df is not None:
    target = st.sidebar.selectbox("", df.columns)
else:
    st.sidebar.write("Load dataset first")

st.sidebar.title("(Optional) Select ID column:")
if df is not None:
    columns = list(df.columns)
    columns.append(no_id_msg)
    id_col = st.sidebar.selectbox("", columns, key="id")
else:
    st.sidebar.write("Load dataset first")

st.sidebar.title("7. How many steps?")
steps = st.sidebar.slider("", min_value=1, max_value=100, step=1)

st.sidebar.title("8. How much time?")
time = st.sidebar.number_input("In seconds", 10, 86400)

st.sidebar.title("9. Optional settings:")
ensemble = st.sidebar.checkbox("Use ensemble")
leaderboard = st.sidebar.checkbox("Show leaderboard", value=True)
optimizer = st.sidebar.selectbox("Optimizer", ["OptunaSearch", "BayesianOptimizer"])
verbosity = st.sidebar.selectbox("Verbosity level", [1, 2], key="verbose")

start_training = st.sidebar.button("Start training!")
CONN = get_database_connection()

if start_training:
    session_state.show_results = False
    if id_col == no_id_msg:
        id_col = None
    with st.spinner("Magic is happening (well, just tuning the models)..."):
        with st.beta_expander("Show output"):
            with st_stdout("text"):
                session_state.automl = AutoML(
                    ConnectionContext(CONN[0], CONN[1], CONN[2], CONN[3])
                )
                session_state.automl.fit(
                    df=df,
                    task=task,
                    steps=int(steps),
                    target=target,
                    table_name=table_name,
                    columns_to_remove=columns_to_rm,
                    categorical_features=categorical,
                    id_column=id_col,
                    optimizer=optimizer,
                    time_limit=int(time),
                    ensemble=ensemble,
                    output_leaderboard=leaderboard,
                    verbosity=verbosity,
                )
                session_state.show_results = True

if session_state.show_results:
    st.markdown("## Success!, here is best model's params:")

    st.write(session_state.automl.opt.get_tuned_params())
    if (
        optimizer == "OptunaSearch"
        and session_state.automl.opt.study.trials_dataframe().shape[0] >= 2
    ):
        st.markdown("## Some cool statistics")
        plot1 = optuna_vs.plot_optimization_history(session_state.automl.opt.study)
        plot2 = optuna_vs.plot_param_importances(session_state.automl.opt.study)
        st.plotly_chart(plot1)
        st.plotly_chart(plot2)

    left_column, right_column = st.beta_columns(2)

    left_column.markdown("## Save model")
    model_name = left_column.text_input(label="Enter model name:")
    schema = left_column.text_input(label="Enter schema:")
    if left_column.button("Save"):
        if model_name != "" and schema != "":
            storage = Storage(
                ConnectionContext(CONN[0], CONN[1], CONN[2], CONN[3]),
                schema,
            )
            session_state.automl.model.name = model_name
            storage.save_model(session_state.automl)
            left_column.success("Saved!")
            left_column.dataframe(storage.list_models())

    right_column.markdown("## Test/predict with model")
    predict_file = right_column.file_uploader(
        label="File to predict:", type=["csv", "xlsx"]
    )
    if predict_file is not None:
        predict_df = pd.read_csv(predict_file)
    if predict_file is not None:
        predict_id_column = right_column.selectbox(
            "Select ID column", predict_df.columns, key="predict_id"
        )
        if right_column.button("Predict"):
            right_column.write(
                session_state.automl.predict(df=predict_df, id_column=predict_id_column)
            )

    test_file = right_column.file_uploader(label="File to test:", type=["csv", "xlsx"])
    if test_file is not None:
        test_df = pd.read_csv(test_file)
    if test_file is not None:
        test_columns = list(test_df.columns)
        test_columns.append(no_id_msg)
        test_id = right_column.selectbox(
            "Select ID column", test_columns, key="id_test"
        )
        test_target = right_column.selectbox(
            "Select target column", test_columns, key="test_t"
        )

        if right_column.button("Test"):
            hana_test_df = create_dataframe_from_pandas(
                ConnectionContext(CONN[0], CONN[1], CONN[2], CONN[3]),
                test_df,
                "AUTOML_TEST",
                force=True,
                drop_exist_tab=True,
            )
            if test_id == no_id_msg:
                hana_test_df = hana_test_df.add_id()
                test_id = "ID"
            if test_target == no_id_msg:
                test_target = None
            print(hana_test_df.columns)
            # hana_test_df.declare_lttab_usage(True)
            right_column.write(
                session_state.automl.score(hana_test_df, target=test_target, id_column=test_id)
            )
