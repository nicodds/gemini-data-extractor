import pytest
from gemini_data_extractor.api_caller import ApiCaller


@pytest.fixture
def example_prompt_file() -> str:
    return 'examples/prompt_001.txt'


@pytest.fixture
def example_data_file() -> str:
    return 'examples/example_001.pdf'


def test_init_with_api_key():
    api_key = "test_api_key"
    caller = ApiCaller(api_key=api_key)
    assert caller._api_key == api_key
    assert caller._project_id is None
    assert caller._genai_client is not None


def test_init_with_project_id():
    project_id = "test_project_id"
    caller = ApiCaller(project_id=project_id)
    assert caller._project_id == project_id
    assert caller._api_key is None
    assert caller._genai_client is not None


def test_init_with_both_api_key_and_project_id():
    api_key = "test_api_key"
    project_id = "test_project_id"
    with pytest.raises(ValueError,
                       match="Both project_id and api_key cannot be provided"):
        ApiCaller(api_key=api_key, project_id=project_id)


def test_init_without_api_key_or_project_id():
    with pytest.raises(ValueError,
                       match="Either project_id or api_key must be provided"):
        ApiCaller()


def test_init_with_location():
    api_key = "test_api_key"
    location = "test_location"
    caller = ApiCaller(api_key=api_key, location=location)
    assert caller._location == location


def test_init_without_location():
    api_key = "test_api_key"
    caller = ApiCaller(api_key=api_key)
    assert caller._location == 'us-central1'


def test_extract_with_prompt_text(mocker, example_data_file):
    api_key = "test_api_key"
    caller = ApiCaller(api_key=api_key)
    mocker.patch.object(
        caller._genai_client.models,  # type: ignore
        'generate_content',
        return_value=mocker.Mock(text="response_text"))
    response = caller.extract(
        prompt_text="test_prompt",
        prompt_file=None,
        path=example_data_file)
    assert response == "response_text"


def test_extract_with_prompt_file(mocker, example_prompt_file,
                                  example_data_file):
    api_key = "test_api_key"
    caller = ApiCaller(api_key=api_key)
    mocker.patch.object(
        caller._genai_client.models,  # type: ignore
        'generate_content',
        return_value=mocker.Mock(text="response_text"))
    response = caller.extract(
        prompt_text=None,
        prompt_file=example_prompt_file,
        path=example_data_file)
    assert response == "response_text"


def test_extract_without_prompt_text_or_file():
    api_key = "test_api_key"
    caller = ApiCaller(api_key=api_key)
    with pytest.raises(
            ValueError,
            match="Either prompt_text or prompt_file must be provided"):
        caller.extract(prompt_text=None, prompt_file=None, path="test_path")


def test_get_file_part(mocker, example_data_file):
    api_key = "test_api_key"
    caller = ApiCaller(api_key=api_key)
    part = caller._get_file_part(example_data_file)
    assert part.inline_data.mime_type == 'application/pdf'  # type: ignore


def test_get_prompt_part_with_text():
    api_key = "test_api_key"
    caller = ApiCaller(api_key=api_key)
    part = caller._get_prompt_part(prompt_text="test_prompt", prompt_file=None)
    assert part.text == "test_prompt"


def test_get_prompt_part_with_file(tmp_path, example_prompt_file):
    api_key = "test_api_key"
    caller = ApiCaller(api_key=api_key)
    part = caller._get_prompt_part(
        prompt_text=None,
        prompt_file=example_prompt_file)
    assert len(str(part.text)) > 100


def test_get_prompt_part_without_text_or_file():
    api_key = "test_api_key"
    caller = ApiCaller(api_key=api_key)
    with pytest.raises(
            ValueError,
            match="Either prompt_text or prompt_file must be provided"):
        caller._get_prompt_part(prompt_text=None, prompt_file=None)
