import time

import pandas as pd
import streamlit as st
from hana_ml.dataframe import ConnectionContext

st.title('Welcome to SAP HANA AutoML!')
st.sidebar.title('1. Enter your HANA database credentials:')
user = st.sidebar.text_input(label='Username')
password = st.sidebar.text_input(label='Password', type='password')
host = st.sidebar.text_input(label='Host')
port = st.sidebar.text_input(label='Port', value='39015')

if st.sidebar.button(label='submit'):
    if user != '' and password != '' and host != '' and port != '':
        with st.spinner('Please wait...'):
            connection_context = ConnectionContext(address=host, port=int(port), user=user, password=password)
        st.success('Successfully connected to the database!')

st.sidebar.markdown('# 2. Load data')
st.sidebar.markdown('## From file:')
uploaded_file = st.sidebar.file_uploader(label='', type=['csv', 'xlsx'])
st.sidebar.write('*(Optional)* provide table name to load dataset there:')

# expander = st.sidebar.beta_expander("Help") expander.write("If 'Table name' is filled, the file will be loaded to
# this table. Leave this field empty to create new " "table from file")
table_name = st.sidebar.text_input(label='')

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.text('Here is the head of your dataset:')
    st.write(df.head(10))

st.sidebar.title('3. Choose task:')
task = st.sidebar.selectbox('', ['Determine task for me', 'Classification', 'Regression'])

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


