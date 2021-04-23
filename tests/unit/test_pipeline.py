from hana_automl.pipeline.pipeline import Pipeline
from unittest import mock
import pytest
from hana_automl.utils.error import PipelineError


@mock.patch("pipeline.data.Data")
def test_invalid_steps(data):
    with pytest.raises(PipelineError) as execinfo:
        pipe = Pipeline(data=data, steps=-1)
    assert execinfo.value.args[0] == "Steps < 1!"
