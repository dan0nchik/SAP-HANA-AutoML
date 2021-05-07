import pandas as pd
import streamlit as st
from hana_ml.dataframe import ConnectionContext
from hana_automl.storage import Storage
from hana_automl.automl import AutoML
import optuna.visualization as optuna_vs
from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import sys


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


st.title('Welcome to SAP HANA AutoML!')

st.sidebar.title('1. Enter your HANA database credentials:')
user = st.sidebar.text_input(label='Username')
password = st.sidebar.text_input(label='Password', type='password')
host = st.sidebar.text_input(label='Host')
port = st.sidebar.text_input(label='Port', value='39015')


@st.cache
def get_database_connection():
    return host, int(port), user, password


@st.cache
def start_train(start: bool = None):
    return start


@st.cache
def show_results(show: bool = None):
    return show


if st.sidebar.button(label='Submit'):
    if user != '' and password != '' and host != '' and port != '':
        get_database_connection()
        st.success('Successfully connected to the database!')

st.sidebar.markdown('# 2. Load data')
st.sidebar.markdown('## From file:')
uploaded_file = st.sidebar.file_uploader(label='', type=['csv', 'xlsx'])
st.sidebar.write('*(Optional)* provide table name to load dataset there:')
table_name = st.sidebar.text_input(label='', value=None)
if table_name == 'None' or table_name == '':
    table_name = None

df = None
automl = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.text('Here is the head of your dataset:')
    st.write(df.head(10))

st.sidebar.title('3. Choose task:')
chosen_task = st.sidebar.selectbox('', ['Determine task for me', 'Classification', 'Regression'])
task = None
if chosen_task == 'Classification':
    task = 'cls'
if chosen_task == 'Regression':
    task = 'reg'

st.sidebar.title('4. Select columns to remove:')
if df is not None:
    columns_to_rm = st.sidebar.multiselect(
        '',
        df.columns)
else:
    st.sidebar.write('Load dataset first')

st.sidebar.title('5. Select categorical features:')
if df is not None:
    categorical = st.sidebar.multiselect(
        '',
        df.columns,
        key='ftr',
    )
else:
    st.sidebar.write('Load dataset first')

st.sidebar.title('6. Select target variable:')
st.sidebar.write('*It is a column to predict*')
if df is not None:
    target = st.sidebar.selectbox('', df.columns)
else:
    st.sidebar.write('Load dataset first')

st.sidebar.title('(Optional) Select ID column:')
if df is not None:
    id_col = st.sidebar.selectbox('', df.columns, key='id')
else:
    st.sidebar.write('Load dataset first')

st.sidebar.title('7. How many steps?')
steps = st.sidebar.slider('', min_value=1, max_value=100, step=1)

st.sidebar.title('8. How much time?')
time = st.sidebar.number_input('In seconds', 1, 86400)

st.sidebar.title('9. Optional settings:')
ensemble = st.sidebar.checkbox('Use ensemble')
leaderboard = st.sidebar.checkbox('Show leaderboard', value=True)
optimizer = st.sidebar.selectbox('Optimizer', ['OptunaSearch', 'BayesianOptimizer'])


if st.sidebar.button('Start training!'):
    CONN = get_database_connection()
    automl = AutoML(ConnectionContext(CONN[0], CONN[1], CONN[2], CONN[3]))
    with st.spinner('Magic is happening (well, just tuning the models)...'):
        with st.beta_expander('Show output'):
            with st_stdout("text"):
                automl.fit(df=df, task=task, steps=int(steps), target=target, table_name=table_name,
                           columns_to_remove=columns_to_rm,
                           categorical_features=categorical, id_column=id_col, optimizer=optimizer,
                           time_limit=int(time),
                           ensemble=ensemble,
                           output_leaderboard=leaderboard)
                show_results(True)
else:
    st.markdown("## 👈 Complete all steps to setup the AutoML process")


if show_results():
    st.markdown("## Success!, here is best model's params:")
    st.write(automl.best_params)
    if optimizer == "OptunaSearch" and steps >= 2:
        st.markdown("## Some cool statistics")
        plot1 = optuna_vs.plot_optimization_history(automl.opt.study)
        plot2 = optuna_vs.plot_param_importances(automl.opt.study)
        st.plotly_chart(plot1)
        st.plotly_chart(plot2)

    left_column, right_column = st.beta_columns(2)

    left_column.markdown('## Save model')
    model_name = left_column.text_input(label="Enter model name:")
    schema = left_column.text_input(label="Enter schema:")
    if left_column.button('Save'):
        if model_name != '' and schema != '':
            storage = Storage(CONN[0], CONN[1], CONN[2], CONN[3], ConnectionContext(CONN[0], CONN[1], CONN[2], CONN[3]),
                              schema)
            storage.save_model(automl)
            left_column.success('Saved!')
            left_column.write(storage.list_models())

    right_column.markdown('## Test/predict with model')
