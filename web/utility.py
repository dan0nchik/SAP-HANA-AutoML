import base64
import sys
from contextlib import contextmanager
from io import StringIO
from threading import current_thread
from typing import Union

import hana_ml.dataframe
import pandas
import streamlit as st
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME


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


def get_table_download_link(df, file_name):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe, file name
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">Download file</a>'


def get_types(df: pandas.DataFrame):
    categorical = []
    for column in df.columns:
        if (
            df[column].dtype in ["object", "bool"]
            or df[column].nunique() < df.shape[0] / 100 * 30
        ):
            categorical.append(column)
    return categorical if len(categorical) > 0 else None
