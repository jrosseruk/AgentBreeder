import json
import backoff
import openai
from icecream import ic
import logging
from dotenv import load_dotenv
import os
import asyncio
from .anthropic_api import ANTHROPIC_PORT
from .openai_api import OPENAI_PORT

load_dotenv(override=True)
import httpx


client = httpx.AsyncClient()
OPENAI_URL = f"http://localhost:{OPENAI_PORT}/json"
CLAUDE_URL = f"http://localhost:{ANTHROPIC_PORT}/json"

# https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py


async def get_json_completion(
    messages, response_format, model="gpt-4o-mini", temperature=0.5, retry=0
) -> dict:
    payload = {
        "messages": messages,
        "response_format": response_format,
        "model": model,
        "temperature": temperature,
    }
    if "claude" in model:
        url = CLAUDE_URL
    else:
        url = OPENAI_URL

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=None)
        data = response.json().get("result")
        return data
    except httpx.RequestError as e:
        logging.error(f"An error occurred while requesting {e.request.url!r}.")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return {}
