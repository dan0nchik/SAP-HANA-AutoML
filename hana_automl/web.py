import time

import pandas as pd
import streamlit as st
from hana_ml.dataframe import ConnectionContext

from hana_automl.automl import AutoML

st.title('Welcome to SAP HANA AutoML!')
st.sidebar.title('1. Enter your HANA database credentials:')
user = st.sidebar.text_input(label='Username')
password = st.sidebar.text_input(label='Password', type='password')
host = st.sidebar.text_input(label='Host')
port = st.sidebar.text_input(label='Port', value='39015')


@st.cache
def get_database_connection():
    return host, int(port), user, password


if st.sidebar.button(label='Submit'):
    if user != '' and password != '' and host != '' and port != '':
        get_database_connection()
        st.success('Successfully connected to the database!')

st.sidebar.markdown('# 2. Load data')
st.sidebar.markdown('## From file:')
uploaded_file = st.sidebar.file_uploader(label='', type=['csv', 'xlsx'])
st.sidebar.write('*(Optional)* provide table name to load dataset there:')
table_name = st.sidebar.text_input(label='', value=None)

df = None
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
    st.sidebar.write('No dataset')

st.sidebar.title('5. Select categorical features:')
if df is not None:
    categorical = st.sidebar.multiselect(
        '',
        df.columns,
        key='ftr',
    )
else:
    st.sidebar.write('No dataset')

st.sidebar.title('6. Select target variable:')
st.sidebar.write('It is a column to predict')
if df is not None:
    target = st.sidebar.selectbox('', df.columns)
else:
    st.sidebar.write('No dataset')

st.sidebar.title('(Optional) Select ID column:')
if df is not None:
    id_col = st.sidebar.selectbox('', df.columns, key='id')
else:
    st.sidebar.write('No dataset')

st.sidebar.title('7. How many steps?')
steps = st.sidebar.slider('', min_value=1, max_value=100, step=1)

st.sidebar.title('8. How much time?')
time = st.sidebar.slider('Time limit in seconds', min_value=1, max_value=3600, step=1)

st.sidebar.title('9. Optional settings:')
ensemble = st.sidebar.checkbox('Use ensemble')
leaderboard = st.sidebar.checkbox('Show leaderboard', value=True)
optimizer = st.sidebar.selectbox('Optimizer', ['OptunaSearch', 'BayesianOptimizer'])
if st.sidebar.button('Start training!'):
    CONN = get_database_connection()
    automl = AutoML(ConnectionContext(CONN[0], CONN[1], CONN[2], CONN[3]))
    with st.spinner('The model is tuning...'):
        automl.fit(df=df, task=task, steps=int(steps), target=target, table_name=table_name,
                   columns_to_remove=columns_to_rm,
                   categorical_features=categorical, id_column=id_col, optimizer=optimizer, time_limit=int(time),
                   ensemble=ensemble,
                   output_leaderboard=leaderboard)
    st.write("Success!, here is best model's params:")
    st.write(automl.best_params)


