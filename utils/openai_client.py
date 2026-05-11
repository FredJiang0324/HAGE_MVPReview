"""
OpenAI Client Wrapper
Supports both regular OpenAI and Azure OpenAI APIs
"""

import os
from typing import Optional
from openai import OpenAI, AzureOpenAI
import dotenv

dotenv.load_dotenv()


def get_openai_client(api_key: Optional[str] = None):
    """
    Get an OpenAI client instance.
    Automatically detects whether to use regular OpenAI or Azure OpenAI based on environment variables.

    Args:
        api_key: Optional API key. If not provided, will use environment variables.

    Returns:
        OpenAI or AzureOpenAI client instance
    """
    use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

    if use_azure:
        # Use Azure OpenAI
        azure_api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.cognitiveservices.azure.com/")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

        if not azure_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set")

        return AzureOpenAI(
            api_key=azure_api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
    else:
        # Use regular OpenAI
        openai_api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        return OpenAI(api_key=openai_api_key)


def get_model_name(default: str = "gpt-4.1-mini") -> str:
    """
    Get the model name to use.
    For Azure, this returns the deployment name.

    Args:
        default: Default model name if not specified in environment

    Returns:
        Model or deployment name
    """
    use_azure = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"

    if use_azure:
        # For Azure, use the deployment name
        return os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", os.getenv("DEFAULT_MODEL", default))
    else:
        # For regular OpenAI, use the model name
        return os.getenv("DEFAULT_MODEL", default)