import hdbcli
import optuna.visualization as optuna_vs
import pandas as pd
import streamlit as st
from hana_ml.dataframe import ConnectionContext

from hana_automl.automl import AutoML
from hana_automl.storage import Storage
from web.session import session_state
from web.utility import st_stdout, get_table_download_link, get_types

st.title("Welcome to SAP HANA AutoML!")
st.write(
    "Useful links: [repository](https://github.com/dan0nchik/SAP-HANA-AutoML), [documentation]("
    "https://sap-hana-automl.readthedocs.io/en/latest/index.html)"
)

if not session_state.show_results and session_state.cc is not None:
    st.write("## 👈 Complete all steps to start training!")
if not session_state.show_results and session_state.cc is None:
    st.write("## What's this? 🤔")
    st.write(
        "This app is visualizing the work of [hana_automl package]("
        "https://github.com/dan0nchik/SAP-HANA-AutoML) - open-source **Automated Machine Learning library**"
    )
    st.write(
        "Here you can train a quite complex machine learning model in a few button clicks!"
    )
    st.write(
        "If you have an SAP HANA instance and have some *tabular* data to train model on, welcome on board!"
    )
    st.markdown("We currently support: **Regression** and **Classification** tasks.")
    st.markdown(
        "Head to our [documentation](https://sap-hana-automl.readthedocs.io/en/latest/index.html) to learn more"
    )
    st.write("👈 Complete all steps to start training!")

st.sidebar.title("1. Enter your HANA database credentials:")
user = st.sidebar.text_input(label="Username")
password = st.sidebar.text_input(label="Password", type="password")
host = st.sidebar.text_input(label="Host")
port = st.sidebar.text_input(label="Port")

df = None
automl = None
columns = []
id_col = None
test_id_col = None
test_df = None
no_id_msg = "I don't have it"
predict_df = None
CONN = None
existing_table = None
metric = None
predict_slot = None


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
        try:
            host, port, user, password = get_database_connection()
            session_state.cc = ConnectionContext(host, port, user, password, sslValidateCertificate = 'false', encrypt=True)
            session_state.show_results = False
            st.success("Successfully connected to the database! 😎")
        except Exception as ex:
            st.error(ex)

st.sidebar.markdown("# 2. Load data")
st.sidebar.markdown("## From file:")
uploaded_file = st.sidebar.file_uploader(label="", type=["csv", "xlsx"])
table_name = st.sidebar.text_input(
    label="OPTIONAL provide table name to load dataset there:", value=None
)
if table_name == "None" or table_name == "":
    table_name = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.text("Here is the head of your dataset:")
    st.dataframe(df.head(10))

st.sidebar.markdown("## Or from HANA database:")
schema = st.sidebar.text_input(label="Enter schema", value="")
if (
    schema != "" or schema != "None" or schema is not None
) and session_state.cc is not None:
    tables = session_state.cc.sql(
        f"SELECT * FROM TABLES WHERE SCHEMA_NAME='{schema}'"
    ).collect()
    existing_table = st.sidebar.selectbox(
        "Select table with data", tables["TABLE_NAME"]
    )

if existing_table is not None and uploaded_file is None:
    df = session_state.cc.table(existing_table, schema).collect()
    st.text(f"Here is the head of dataset from HANA table {existing_table}")
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
st.sidebar.write(
    "Confused what is it? Read more [here](https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63)"
)
if df is not None:
    categorical = st.sidebar.multiselect(
        "We have chosen them for you, please check",
        list(df.columns),
        default=get_types(df),
        key="ftr",
    )
else:
    st.sidebar.write("Load dataset first")

st.sidebar.title("6. Select target variable:")
st.sidebar.write("*It is a column to predict*")
if df is not None:
    target = st.sidebar.selectbox("", df.columns, index=len(df.columns) - 1)
else:
    st.sidebar.write("Load dataset first")

st.sidebar.title("(Optional) Select ID column:")
if df is not None:
    columns = list(df.columns)
    columns.append(no_id_msg)
    id_col = st.sidebar.selectbox("", columns, key="id", index=len(columns) - 1)
else:
    st.sidebar.write("Load dataset first")

st.sidebar.title("7. How many steps?")
steps = st.sidebar.slider("", min_value=1, max_value=100, step=1, value=3)

st.sidebar.title("8. How much time?")
time = st.sidebar.number_input("In seconds", 10, 86400, value=60)
st.sidebar.write(f"Total training time will be ~{(time * 2) // 60} min")

st.sidebar.title("9. Advanced settings:")
ensemble = st.sidebar.checkbox("Use ensemble")
leaderboard = st.sidebar.checkbox("Show leaderboard", value=True)
optimizer = st.sidebar.selectbox(
    "Hyperparameter optimizer", ["OptunaSearch", "BayesianOptimizer"]
)
verbose = st.sidebar.selectbox("Level of output", [0, 1, 2], key="verbose", index=2)

if task == "reg":
    metric = st.sidebar.selectbox("Metric", ["mae", "mse", "rmse"])

start_training = st.sidebar.button("Start training!")

if start_training:
    session_state.show_results = False
    if id_col == no_id_msg:
        id_col = None
    with st.spinner(
        "Please wait, magic is happening (well, just tuning the models)..."
    ):
        with st.beta_expander("Show output"):
            with st_stdout("text"):
                session_state.automl = AutoML(session_state.cc)
                if existing_table is not None and uploaded_file is None:
                    df_to_fit = existing_table
                else:
                    df_to_fit = df
                session_state.automl.fit(
                    df=df_to_fit,
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
                    verbose=verbose,
                    tuning_metric=metric,
                )
                session_state.show_results = True

if session_state.show_results:
    st.markdown("## Success!, here is best model's params:")
    st.write(session_state.automl.opt.get_tuned_params())
    if optimizer == "OptunaSearch":
        if session_state.automl.opt.study.trials_dataframe().shape[0] >= 2:
            st.markdown("## Some cool statistics")
            plot1 = optuna_vs.plot_optimization_history(session_state.automl.opt.study)
            plot2 = optuna_vs.plot_param_importances(session_state.automl.opt.study)
            st.plotly_chart(plot1)
            st.plotly_chart(plot2)

    left_column, right_column = st.beta_columns(2)
    with left_column:
        st.markdown("## Save model")
        model_name = st.text_input(label="Enter model name:")
        schema = st.text_input(label="Enter schema:", value=user)
        if st.button("Save"):
            if model_name != "" and schema != "" and schema is not None:
                storage = Storage(
                    session_state.cc,
                    schema,
                )
                session_state.automl.model.name = model_name
                storage.save_model(session_state.automl)
                st.success("Saved!")
                st.dataframe(storage.list_models())
            else:
                st.error("Please provide valid schema and name!")

    with right_column:
        st.markdown("## Test/predict with model")

        predict_file = st.file_uploader(label="File to predict:", type=["csv", "xlsx"])
        if predict_file is not None:
            predict_df = pd.read_csv(predict_file)
            predict_slot = st.empty()
            predict_slot.write(predict_df.head(5))
        if predict_df is not None:
            predict_columns = list(predict_df.columns)
            predict_columns.append(no_id_msg)
            predict_id_column = st.selectbox(
                "Select ID column", predict_columns, key="predict_id"
            )
            if predict_id_column == no_id_msg:
                predict_df["ID"] = range(0, len(predict_df))
                predict_id_column = "ID"

            if right_column.button("Predict"):
                predicted = session_state.automl.predict(
                    df=predict_df, id_column=predict_id_column
                )
                st.write(predicted)
                st.write(
                    get_table_download_link(predicted, "predictions"),
                    unsafe_allow_html=True,
                )

        test_file = st.file_uploader(label="File to test:", type=["csv", "xlsx"])

        if test_file is not None:
            test_df = pd.read_csv(test_file)

        if test_df is not None:
            test_slot = st.empty()
            test_slot.write(test_df.head(5))
            test_columns = list(test_df.columns)
            test_columns.append(no_id_msg)
            test_id = st.selectbox("Select ID column", test_columns, key="id_test")
            test_target = st.selectbox(
                "Select target column", test_columns, key="test_t"
            )
            if test_id == no_id_msg:
                test_df["ID"] = range(0, len(test_df))
                test_id = "ID"
            if test_target == no_id_msg:
                test_target = None
            if st.button("Test"):
                st.write(
                    "Model score: "
                    + str(
                        session_state.automl.score(
                            test_df, target=test_target, id_column=test_id
                        )
                    )
                )
    if st.button("Reset app (start again)"):
        session_state.show_results = False
        session_state.cc = None
        session_state.automl = AutoML(None)
