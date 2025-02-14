import random
from api import get_json_completion
from discover.meta_agent_prompts import REFLECTION_PROMPT_1
import os
import uuid
from evals import AgentScaffoldException
import logging
import json
import re
import asyncio
from evals import Benchmark


class Evolve:

    def __init__(
        self,
        args,
        mutation_operators,
        evaluator,
        base_prompt,
        base_prompt_response_format,
        debug_sample,
    ) -> None:
        """
        Initializes the Evolve class.

        Args:
            args: Arguments object containing configurations for the mutator, such
            as debugging limits and model settings.

            population: The population of scaffolds for mutation.
            mutation_operators: A list of mutation operator strings to apply.
            evaluator: An evaluator object for validating and testing evolved scaffolds.
        """

        self.mutation_operators = mutation_operators

        self.args = args
        self.evaluator = evaluator

        self.base_prompt = base_prompt
        self.base_prompt_response_format = base_prompt_response_format

        self.debug_sample = debug_sample

    async def evolve(self, parents: list[dict]) -> dict:
        """
        Applies a mutation to a given scaffold.

        Args:
            scaffold (Scaffold): The scaffold object to mutate.

        Returns:
            Scaffold: The mutated scaffold object. Returns None if the mutation fails.
        """

        mutated_scaffold = None
        scaffold_response = {"code": None}
        i = 0
        while (not mutated_scaffold or not scaffold_response.get("code")) and i < 3:
            i += 1
            try:
                (
                    scaffold_response,
                    messages,
                    reflexion_response_format,
                    parent_scaffold_ids,
                    sampled_mutation,
                ) = await random.choice([self._mutate, self._mutate, self._crossover])(
                    parents
                )

                scaffold_response = await self._debug(
                    messages, scaffold_response, reflexion_response_format
                )

                mutated_scaffold = {
                    "scaffold_name": scaffold_response["name"],
                    "scaffold_code": scaffold_response["code"],
                    "scaffold_first_parent_id": str(parent_scaffold_ids[0]),
                    "scaffold_second_parent_id": str(parent_scaffold_ids[1]),
                    "scaffold_thought_process": scaffold_response["thought"],
                    "scaffold_mutation_prompt": (
                        sampled_mutation if sampled_mutation else ""
                    ),
                }
            except Exception as e:

                print(f"Error evolving scaffold: {e}")
                mutated_scaffold = None

        return mutated_scaffold

    async def _mutate(self, parents: list[dict]):
        """
        Applies a sampled mutation to a scaffold and refines it using reflexion-based prompts.

        Args:
            None

        Returns:
            tuple: A tuple containing the next_response (dict), the updated messages (list),
                and the reflexion_response_format (str).
        """
        scaffold = parents[0]
        logging.info(f"Mutating {scaffold.get('scaffold_name')} scaffold...")
        print(f"Mutating {scaffold.get('scaffold_name')} scaffold...")

        sampled_mutation = random.choice(self.mutation_operators)

        messages = [
            {
                "role": "user",
                "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object.""",
            },
            {
                "role": "user",
                "content": f"""
                {self.base_prompt}
             
                Here is the multi-agent scaffold I would like you to mutate:

                ---------------
                Scaffold: {scaffold.get('scaffold_name')}
                {scaffold.get("scaffold_thought_process")}
                ---------------
                {scaffold.get("scaffold_code")}

                The mutation I would like to apply is:
                {sampled_mutation}

                
                IMPORTANT:
                In general, the new scaffold will perform better with more detailed prompts for the agents, more planning steps,
                encouringing them to think longer and harder. It may be worth adding a final agent to the scaffold to help
                transform the output of the final agent into the desired output format for the task as the scaffold will be scored
                very lowley if the output is not in the correct format, even if the thinking was sound.

                Ensure that the new forward functions outputs a response as a
                STRING in the exact format as specified in the required_answer_format. This could be
                either a single letter (e.g. A, B, C, D) or a word or phrase, or a 
                short piece of code.
                
                """.strip(),
            },
        ]

        return await self._evolve(
            messages, [parents[0].get("scaffold_id"), None], sampled_mutation
        )

    async def _crossover(self, parents: list[dict]):
        """
        Applies crossover to two scaffolds and refines the result using reflexion-based prompts.

        Args:
            None

        Returns:
            tuple: A tuple containing the next_response (dict), the updated messages (list),
                and the reflexion_response_format (str).
        """
        scaffold_1 = parents[0]
        scaffold_2 = parents[1]
        logging.info(
            f"Crossing over {scaffold_1.get('scaffold_name')} and {scaffold_2.get('scaffold_name')} scaffolds..."
        )
        print(
            f"Crossing over {scaffold_1.get('scaffold_name')} and {scaffold_2.get('scaffold_name')} scaffolds..."
        )

        messages = [
            {
                "role": "user",
                "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object.""",
            },
            {
                "role": "user",
                "content": f"""
                {self.base_prompt}
             
                Here are the two scaffolds I'd like you to crossover/combine into a novel new scaffold:

                ---------------
                Scaffold 1: {scaffold_1.get('scaffold_name')}
                {scaffold_1.get("scaffold_thought_process")}
                ---------------
                {scaffold_1.get("scaffold_code")}

                ---------------
                Scaffold 2: {scaffold_2.get('scaffold_name')}
                {scaffold_2.get("scaffold_thought_process")}
                ---------------
                {scaffold_2.get("scaffold_code")}   

                Ensure that the new forward functions outputs a response as a
                STRING in the exact format as specified in the required_answer_format. This could be
                either a single letter (e.g. A, B, C, D) or a word or phrase, or a 
                short piece of code.             
                """.strip(),
            },
        ]

        return await self._evolve(
            messages,
            [parents[0].get("scaffold_id"), parents[1].get("scaffold_id")],
            None,
        )

    async def _base(self, parents: list[dict]):
        """
        Applies a sampled mutation to a scaffold and refines it using reflexion-based prompts.

        Args:
            None

        Returns:
            tuple: A tuple containing the next_response (dict), the updated messages (list),
                and the reflexion_response_format (str).
        """
        scaffold = parents[0]
        logging.info(f"Mutating {scaffold.get('scaffold_name')} scaffold...")
        print(f"Mutating {scaffold.get('scaffold_name')} scaffold...")

        sampled_mutation = random.choice(self.mutation_operators)

        messages = [
            {
                "role": "user",
                "content": """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object.""",
            },
            {
                "role": "user",
                "content": f"""
                {self.base_prompt}

                Please generate a new multi-agent scaffold from scratch. Use the multi-agent structure
                provided e.g. Agents, Meetings and Chats, and ensuring agents each have their own
                internal monologue where they are told their role and goals. Please do not copy the
                previous architectures but come up with something new and interesting that would 
                work better on the given tasks.

                Ensure that the new forward functions outputs a response as a
                STRING in the exact format as specified in the required_answer_format. This could be
                either a single letter (e.g. A, B, C, D) or a word or phrase, or a 
                short piece of code.
                
                """.strip(),
            },
        ]

        return await self._evolve(messages, [parents[0].get("scaffold_id"), None], None)

    async def _evolve(self, messages, parent_scaffold_ids, sampled_mutation):

        # Generate new solution and do reflection
        try:

            next_response: dict[str, str] = await get_json_completion(
                messages,
                self.base_prompt_response_format,
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                temperature=0.5,
                retry=0,
            )
            # print(next_response)

            # Reflexion 1
            REFLECTION_PROMPT_1, reflexion_response_format = (
                self._get_reflexion_prompt_1(next_response)
            )

            messages.append({"role": "assistant", "content": str(next_response)})
            messages.append({"role": "user", "content": REFLECTION_PROMPT_1})
            next_response = await get_json_completion(
                messages,
                reflexion_response_format,
                model="gpt-4o",
                temperature=0.5,
                retry=0,
            )
            # Reflexion 2
            Reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section,
            revise the code further. Put your new reflection thinking in "reflection". Repeat the
            previous "thought" and "name", and update the corrected version of the code in "code".
            """
            messages.append({"role": "assistant", "content": str(next_response)})
            messages.append({"role": "user", "content": Reflexion_prompt_2})
            next_response = await get_json_completion(
                messages,
                reflexion_response_format,
                model="gpt-4o",
                temperature=0.5,
                retry=0,
            )

        except Exception as e:
            print("During LLM generate new solution:")
            print(e)

            return None

        # Clean up the scaffold to only allow numbers, letters, hyphens and underscores
        next_response["name"] = re.sub(
            r"[^A-Za-z0-9 \-\u2013\u2014]+", "", next_response["name"]
        )

        return (
            next_response,
            messages,
            reflexion_response_format,
            parent_scaffold_ids,
            sampled_mutation,
        )

    def _get_reflexion_prompt_1(self, prev_example):

        prev_example_str = (
            "Here is the previous agent you tried:\n"
            + json.dumps(prev_example)
            + "\n\n"
        )
        r1 = (
            REFLECTION_PROMPT_1.replace("[EXAMPLE]", prev_example_str)
            if prev_example
            else REFLECTION_PROMPT_1.replace("[EXAMPLE]", "")
        )
        reflexion_response_format = {
            "reflection": """
                Provide your thoughts on the interestingness of the architecture,
                identify any mistakes in the implementation, and suggest improvements.
            """,
            "thought": """
                Revise your previous proposal or propose a new architecture if necessary,
                using the same format as the example response.
            """,
            "name": """
                Provide a name for the revised or new architecture. (Don't put words like
                'new' or 'improved' in the name.)
            """,
            "code": """
                Provide the corrected code or an improved implementation. Make sure you
                actually implement your fix and improvement in this code.
            """,
        }

        return r1, reflexion_response_format

    async def _debug(self, messages, next_response, reflexion_response_format):
        """
        Handles debugging for a given response during mutation.

        Args:
            messages: List of messages exchanged during the mutation process.
            next_response: The generated response containing the multi-agent scaffold code and metadata.
            reflexion_response_format: The response format for reflection.

        Returns:
            dict: The updated next_response after debugging attempts.
        """

        agent_scaffold, temp_file = Benchmark.get_callable(
            str(uuid.uuid4()), next_response["name"], next_response["code"]
        )

        for d in range(self.args.debug_max):

            try:

                if "return self.forward" in next_response["code"]:
                    raise AgentScaffoldException(
                        """The output of the forward function must not be the forward function
                        itself, as it will recurse infinitely."""
                    )

                if "return await self.forward" in next_response["code"]:
                    raise Exception("Infinite loop detected")
                agentScaffold = agent_scaffold()
                # Set a timeout of 3 minutes (180 seconds)
                try:
                    print(self.debug_sample)
                    input = self.debug_sample.input
                    format = self.debug_sample.metadata["format"]

                    print("Input", input)
                    print("Format", format)
                    output = await asyncio.wait_for(
                        agentScaffold.forward(input, format), timeout=720
                    )
                except asyncio.TimeoutError:
                    next_response["code"] = None
                    break
                    # raise AgentScaffoldException(
                    #     """The forward function took too long to execute. Make sure your code
                    #     is efficient and doesn't have any infinite loops."""
                    # )
                except Exception as e:
                    raise AgentScaffoldException(e)

                if output.lower().startswith("error"):
                    raise AgentScaffoldException(output)
                print("Debug successful")
                break

            except AgentScaffoldException as e:
                logging.info(f"Debugging meta agent's code: {e}")
                messages.append({"role": "assistant", "content": str(next_response)})
                messages.append(
                    {
                        "role": "user",
                        "content": f"""Error during evaluation:\n{e}\nCarefully consider where
                        you went wrong in your latest implementation. Using insights from
                        previous attempts, try to debug the current code to implement the
                        same thought. Repeat your previous thought in 'thought', and put
                        your thinking for debugging in 'debug_thought'""",
                    }
                )
                try:
                    next_response = await get_json_completion(
                        messages,
                        reflexion_response_format,
                        model=self.args.model,
                        temperature=0.5,
                        retry=0,
                    )
                except Exception as e:
                    print(f"Error during debugging: {e}")
                    next_response["code"] = None

        os.remove(temp_file)

        return next_response
