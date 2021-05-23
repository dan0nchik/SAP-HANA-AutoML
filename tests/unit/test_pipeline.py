from unittest import mock

import pytest

from hana_automl.pipeline.data import Data
from hana_automl.pipeline.input import Input
from hana_automl.pipeline.pipeline import Pipeline
from hana_automl.utils.error import InputError, PipelineError


def test_input():
    input = Input()
    with pytest.raises(InputError, match="No data provided"):
        input.load_data()
    with pytest.raises(InputError, match="Please provide valid file path or url"):
        input.download_data("")
    with pytest.raises(InputError, match="The file format is missing or not supported"):
        input.download_data("/home/user/downloads/data.lalalala")


@mock.patch("hana_automl.pipeline.data.Data")
def test_pipe(data):
    with pytest.raises(PipelineError, match="Optimizer not found!"):
        pipe = Pipeline(data, 0, "reg")
        pipe.train()
