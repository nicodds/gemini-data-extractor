import pytest
import os
from argparse import Namespace, ArgumentParser
from gemini_data_extractor import _get_variables, _setup_argparser


@pytest.fixture
def example_prompt_file() -> str:
    return 'examples/prompt_001.txt'


@pytest.fixture
def example_data_file() -> str:
    return 'examples/example_001.pdf'


@pytest.fixture
def mock_args_direct_prompt(example_data_file) -> Namespace:
    args = Namespace()
    args.prompt_text = "test prompt"
    args.prompt_file = None
    args.path = example_data_file
    args.api_key = None
    args.project_id = None
    args.location = None
    return args


def test__get_variables_no_value(mock_args_direct_prompt):
    os.environ['GEMINI_API_KEY'] = "api_key"
    os.environ['GCP_PROJECT_ID'] = "project_id"
    os.environ['GCP_LOCATION'] = "location"

    api_key, project_id, location = _get_variables(mock_args_direct_prompt)
    assert api_key == "api_key"
    assert project_id == "project_id"
    assert location == "location"


def test__get_variables_with_value(mock_args_direct_prompt):
    mock_args_direct_prompt.api_key = "api_key"
    mock_args_direct_prompt.project_id = "project_id"
    mock_args_direct_prompt.location = "location"

    api_key, project_id, location = _get_variables(mock_args_direct_prompt)
    assert api_key == "api_key"
    assert project_id == "project_id"
    assert location == "location"


def test__setup_argparser():
    parser = _setup_argparser()

    assert isinstance(parser, ArgumentParser)
    assert parser.prog == "gemini-data-extractor"
    assert parser.description == \
        "Extract data from files using Gemini AI models"
    assert len(parser._mutually_exclusive_groups) == 2
    assert parser._mutually_exclusive_groups[0].required is False
    assert parser._mutually_exclusive_groups[1].required is True
