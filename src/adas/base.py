import string
import random
import copy
from collections import namedtuple

Info = namedtuple("Info", ["name", "author", "content", "iteration_idx"])
SEARCHING_MODE = False

import openai
from openai import AsyncOpenAI
import json
import backoff
import logging
from dotenv import load_dotenv
import asyncio

#  Disable logging for httpx
logging.getLogger("httpx").disabled = True
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

load_dotenv(override=True)
client = AsyncOpenAI()
MAX_REQUESTS_PER_MINUTE = 29000  # adjust as needed
N = MAX_REQUESTS_PER_MINUTE / 60 / 3  # We'll dequeue one item every 1/N seconds


# @backoff.on_exception(backoff.expo, openai.RateLimitError)
async def get_json_response_from_gpt(msg, model, system_message, temperature=0.5):
    await asyncio.sleep(1.0 / N)
    # print("Request:", msg)
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature,
        max_tokens=4096,
        stop=None,
        response_format={"type": "json_object"},
    )
    # print("Response:", response)
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict


class LLMAgentBase:
    """
    Attributes:
    """

    def __init__(
        self,
        output_fields: list,
        agent_name: str,
        role="helpful assistant",
        model="gpt-4o-mini",
        temperature=0.5,
    ) -> None:
        self.output_fields = output_fields
        self.agent_name = agent_name

        self.role = role
        self.model = model
        self.temperature = temperature

        # give each instance a unique id
        self.id = self.random_id()

    def random_id(self, length=4):
        characters = (
            string.ascii_letters + string.digits
        )  # includes both upper/lower case letters and numbers
        random_id = "".join(random.choices(characters, k=length))
        return random_id

    def generate_prompt(self, input_infos, instruction) -> str:
        # construct system prompt
        output_fields_and_description = {
            key: (
                f"Your {key}."
                if not "answer" in key
                else f"Your {key}. Directly answer the question. Keep it very concise."
            )
            for key in self.output_fields
        }
        ROLE_DESC = lambda role: f"You are a {role}."
        FORMAT_INST = (
            lambda request_keys: f"""Reply EXACTLY with the following JSON format.{str(request_keys)}DO NOT MISS ANY REQUEST FIELDS and ensure that your response is a well-formed JSON object!"""
        )
        system_prompt = (
            ROLE_DESC(self.role) + "" + FORMAT_INST(output_fields_and_description)
        )

        # construct input infos text
        input_infos_text = ""
        for input_info in input_infos:
            if isinstance(input_info, Info):
                (field_name, author, content, iteration_idx) = input_info
            else:
                continue
            if author == self.__repr__():
                author += " (yourself)"
            if field_name == "task":
                input_infos_text += f"# Your Task:{content}"
            elif iteration_idx != -1:
                input_infos_text += (
                    f"### {field_name} #{iteration_idx + 1} by {author}:{content}"
                )
            else:
                input_infos_text += f"### {field_name} by {author}:{content}"

        prompt = input_infos_text + instruction
        return system_prompt, prompt

    async def query(self, input_infos: list, instruction, iteration_idx=-1) -> dict:
        system_prompt, prompt = self.generate_prompt(input_infos, instruction)
        try:
            response_json = {}
            response_json = await get_json_response_from_gpt(
                prompt, self.model, system_prompt, self.temperature
            )
            assert len(response_json) == len(
                self.output_fields
            ), "not returning enough fields"
        except Exception as e:
            # print(e)
            if "maximum context length" in str(e) and SEARCHING_MODE:
                raise AssertionError(
                    "The context is too long. Please try to design the agent to have shorter context."
                )
            # try to fill in the missing field
            for key in self.output_fields:
                if not key in response_json and len(response_json) < len(
                    self.output_fields
                ):
                    response_json[key] = ""
            for key in copy.deepcopy(list(response_json.keys())):
                if (
                    len(response_json) > len(self.output_fields)
                    and not key in self.output_fields
                ):
                    del response_json[key]
        output_infos = []
        for key, value in response_json.items():
            info = Info(key, self.__repr__(), value, iteration_idx)
            output_infos.append(info)
        return output_infos

    def __repr__(self):
        return f"{self.agent_name} {self.id}"

    async def __call__(self, input_infos: list, instruction, iteration_idx=-1):
        return await self.query(input_infos, instruction, iteration_idx=iteration_idx)
