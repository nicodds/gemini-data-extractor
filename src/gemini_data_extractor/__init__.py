from argparse import ArgumentParser, Namespace
import os
from typing import Optional, Tuple
import warnings

from .api_caller import ApiCaller


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    args = _setup_argparser().parse_args()
    api_key, project_id, location = _get_variables(args)

    caller = ApiCaller(
        api_key=api_key,
        project_id=project_id,
        location=location
    )

    data = caller.extract(
        prompt_text=args.prompt_text,
        prompt_file=args.prompt_file,
        path=args.path
    )

    print(data)


def _get_variables(args: Namespace) -> Tuple[
                Optional[str], Optional[str], Optional[str]]:

    api_key = args.api_key or os.getenv('GEMINI_API_KEY', None)
    project_id = args.project_id or os.getenv('GCP_PROJECT_ID', None)
    location = args.location or os.getenv('GCP_LOCATION', None)

    return (api_key, project_id, location)


def _setup_argparser() -> ArgumentParser:
    argparser = ArgumentParser(
        prog="gemini-data-extractor",
        description="Extract data from files using Gemini AI models"
    )
    auth_group = argparser.add_mutually_exclusive_group(required=False)
    prompt_group = argparser.add_mutually_exclusive_group(required=True)

    auth_group.add_argument(
        "--api-key",
        help="API key for Gemini AI (can be provided as " +
             "GEMINI_API_KEY env var)",
        required=False
    )
    auth_group.add_argument(
        "--project-id",
        help="Google Cloud Project ID for Vertex AI endpoints (can " +
             "be provided as GCP_PROJECT_ID env var)",
        required=False
    )

    prompt_group.add_argument(
        "--prompt-text",
        help="Text to use as prompt",
        required=False
    )
    prompt_group.add_argument(
        "--prompt-file",
        help="Path to file containing text to use as prompt",
        required=False
    )

    argparser.add_argument(
        "--path",
        help="Path to file to extract data from",
        required=True
    )

    argparser.add_argument(
        "--location",
        help="Google Cloud location for Vertex AI endpoints",
        required=False
    )

    return argparser
