import sys
import asyncio

sys.path.append("src")
sys.path.append("src/adas")

from adas.base import LLMAgentBase, Info


class MMLUAgentScaffold:
    def __init__(self) -> None:
        pass

    async def forward(self, input_text, required_answer_format=""):

        taskInfo = Info("task", "User", input_text + required_answer_format, -1)

        # Domain-specific expert analysis instruction
        domain_analysis_instruction = (
            "Please analyze the given problem in detail and provide your reasoning."
        )
        # Reflection and feedback instruction
        reflection_instruction = "Reflect on the previous reasoning and feedback. Request specific additional information if needed, and refine your solution."
        # Final decision instruction
        final_decision_instruction = "Given all the intermediate results and reflections, reason over them carefully and provide a final answer."
        # Initialize domain-specific expert agents
        domain_agents = {
            "physics": LLMAgentBase(["reasoning"], "Physics Expert"),
            "history": LLMAgentBase(["reasoning"], "History Expert"),
            "chemistry": LLMAgentBase(["reasoning"], "Chemistry Expert"),
            "general": LLMAgentBase(["reasoning"], "General Expert"),
        }
        # Domain identification instruction
        domain_identification_instruction = 'Please identify the domain of the given question. Return the domain in the "domain" field.'
        domain_identification_agent = LLMAgentBase(
            ["domain"], "Domain Identification Agent"
        )
        # Initialize reflection agent
        reflection_agent = LLMAgentBase(["reflection", "request"], "Reflection Agent")
        # Initialize final decision agent
        final_decision_agent = LLMAgentBase(
            ["final_answer"], "Final Decision Agent", temperature=0.1
        )
        # Number of maximum iterations
        N_max = 5
        # Identify the domain of the task
        domain_info = await domain_identification_agent(
            [taskInfo], domain_identification_instruction
        )
        domain_info = domain_info[0]
        domain = domain_info.content.lower()
        domain_agent = domain_agents.get(domain, domain_agents["general"])
        # Domain-specific expert analysis
        expert_analysis = await domain_agent([taskInfo], domain_analysis_instruction)
        intermediate_results = expert_analysis
        for i in range(N_max):
            # Reflection and feedback loop
            reflection_results = await reflection_agent(
                [taskInfo] + intermediate_results, reflection_instruction
            )
            reflection, request = reflection_results[0], reflection_results[1]
            intermediate_results.extend(reflection_results)
            # Use the requested information to refine the solution
            if "additional information" in request.content.lower():
                additional_info = await domain_agent(
                    [taskInfo] + intermediate_results, domain_analysis_instruction
                )
                intermediate_results.extend(additional_info)
        # Make the final decision based on all intermediate results
        final_results = await final_decision_agent(
            [taskInfo] + intermediate_results, final_decision_instruction
        )
        final_answer = [info for info in final_results if info.name == "final_answer"][
            0
        ]
        return final_answer.content


if __name__ == "__main__":
    MMLUAgentScaffold = MMLUAgentScaffold()
    # Set a timeout of 3 minutes (180 seconds)
    input = "What is the capital of france?"
    output = asyncio.run(MMLUAgentScaffold.forward(input))
    output = str(output)
    print(output)
