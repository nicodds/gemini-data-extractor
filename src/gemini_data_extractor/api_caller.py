from typing import Optional
from mimetypes import MimeTypes
from google import genai
from google.genai.types import Part

__all__ = ["ApiCaller"]

MODEL_ID = 'gemini-2.0-flash-001'
LOCATION = 'us-central1'


class ApiCaller:
    def __init__(self, api_key: Optional[str] = None,
                 project_id: Optional[str] = None,
                 location: Optional[str] = None) -> None:
        self._api_key = api_key
        self._project_id = project_id
        self._location = location or LOCATION
        self._genai_client = None

        if self._project_id is not None and self._api_key is not None:
            raise ValueError("Both project_id and api_key cannot be provided")
        elif self._project_id is not None:
            self._genai_client = genai.Client(vertexai=True,
                                              project=self._project_id,
                                              location=self._location)
        elif self._api_key is not None:
            self._genai_client = genai.Client(api_key=self._api_key)
        else:
            raise ValueError("Either project_id or api_key must be provided")

    def extract(self, prompt_text: Optional[str],
                prompt_file: Optional[str],
                path: str) -> Optional[str]:
        prompt = self._get_prompt_part(prompt_text, prompt_file)
        file = self._get_file_part(path)

        response = self._genai_client.models.generate_content(  # type: ignore
            model=MODEL_ID, contents=[prompt, file]
        )

        return response.text

    def _get_file_part(self, path: str) -> Part:
        mime_type = MimeTypes().guess_type(path)[0] or \
            'application/octet-stream'

        with open(path, 'rb') as f:
            return Part.from_bytes(data=f.read(),
                                   mime_type=mime_type)

    def _get_prompt_part(self, prompt_text: Optional[str],
                         prompt_file: Optional[str]) -> Part:
        if (prompt_text is None and prompt_file is None) or \
                (prompt_text is not None and prompt_file is not None):
            raise ValueError("Either prompt_text or prompt_file must " +
                             "be provided")
        if prompt_file is not None:
            with open(prompt_file, 'r') as f:
                prompt_text = f.read()

        return Part.from_text(text=prompt_text)  # type: ignore
