import os
import threading

import httpx
import openai
from dotenv import load_dotenv

from agency_swarm.util.tracking.tracker_factory import get_tracker_by_name

load_dotenv()

client_lock = threading.Lock()
client = None
_tracker = get_tracker_by_name("local")


def set_tracker(name: str):
    """Set the global usage tracker.

    Args:
        name: The name of the usage tracking mechanism to use.
    """
    global _tracker, client
    with client_lock:
        _tracker = get_tracker_by_name(name)
    client = get_openai_client()


def get_tracker():
    """Get the current usage tracker instance.

    Returns:
        Tracker: The current usage tracker instance.
    """
    return _tracker


def get_openai_client():
    global client
    with client_lock:
        if client is None:
            # Check if the API key is set
            api_key = openai.api_key or os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenAI API key is not set. Please set it using set_openai_key."
                )

            client = openai.OpenAI(
                api_key=api_key,
                timeout=httpx.Timeout(60.0, read=40, connect=5.0),
                max_retries=10,
                default_headers={"OpenAI-Beta": "assistants=v2"},
            )
    return client


def set_openai_client(new_client):
    global client
    with client_lock:
        client = new_client


def set_openai_key(key: str):
    if not key:
        raise ValueError("Invalid API key. The API key cannot be empty.")

    openai.api_key = key

    global client
    with client_lock:
        client = None
