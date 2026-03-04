import os
from typing import Union
from langfuse import Langfuse


from rtk.utils import get_logger

logger = get_logger("rtk.langfuse")


def create_langfuse_client(**kwargs) -> Langfuse:
    if not kwargs:
        return Langfuse()
    return Langfuse(**kwargs)


def load_prompt_from_langfuse(ls: Langfuse = create_langfuse_client(), **kwargs) -> str:
    """Load a prompt from Langfuse using the provided prompt ID.

    Args:
        ls (Langfuse): An instance of the Langfuse client.
        **kwargs: Additional keyword arguments to pass to the `get_prompt` method."""
    try:
        logger.debug(f"Attempting to load prompt from Langfuse with parameters: {kwargs}")
        prompt_template = ls.get_prompt(**kwargs)
    except Exception as e:
        logger.error(f"Failed to load prompt with name '{kwargs.get('name', 'unknown')}': {e}")
        raise
    return prompt_template
