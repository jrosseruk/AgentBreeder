import sys
import asyncio

sys.path.append("src")
sys.path.append("src/adas")

from adas.base import LLMAgentBase, Info


class GPQAAgentSystem:
    def __init__(self) -> None:
        pass

    async def forward(self, input_text, required_answer_format=""):

        taskInfo = Info("task", "User", input_text + required_answer_format, -1)

        # Define initial instruction for expert agents to apply heuristics
        initial_instruction = "Apply domain-specific heuristics to provide your initial thoughts and answers."

        # Initialize domain-specific Expert Agents
        expert_roles = [
            "Physics Heuristic Expert",
            "Chemistry Heuristic Expert",
            "Biology Heuristic Expert",
        ]
        expert_agents = [
            LLMAgentBase(["thinking", "answer"], role + " Agent", role=role)
            for role in expert_roles
        ]
        # Initial thoughts and answers from all experts
        all_thoughts_answers = [
            await agent([taskInfo], initial_instruction) for agent in expert_agents
        ]
        all_thinking = [ta[0] for ta in all_thoughts_answers]
        all_answers = [ta[1] for ta in all_thoughts_answers]

        # Initialize Adaptive Agent for dynamic refinement
        adaptive_instruction = "Refine the answers dynamically based on performance feedback and confidence scores."
        adaptive_agent = LLMAgentBase(["thinking", "answer"], "Adaptive Agent")
        # Instruction for generating confidence scores
        confidence_instruction = (
            "On a scale from 0 to 1, how confident are you in your previous answer?"
        )
        confidence_agent = LLMAgentBase(["confidence"], "Confidence Agent")
        # Define threshold and maximum iterations
        threshold = 0.8
        iteration = 0
        max_iterations = 5
        while iteration < max_iterations:
            # Assess confidence levels of initial answers
            confidence_levels = []
            for i in range(len(expert_agents)):
                confidence_response = await confidence_agent(
                    [taskInfo, all_thinking[i], all_answers[i]], confidence_instruction
                )
                try:
                    confidence_level = float(confidence_response[0].content)
                    confidence_levels.append(confidence_level)
                except ValueError:
                    confidence_levels.append(
                        0.0
                    )  # Default to low confidence if conversion fails
            avg_confidence = sum(confidence_levels) / len(confidence_levels)
            # If the average confidence level is above the threshold, break the loop
            if avg_confidence >= threshold:
                break
            # Refine answers based on performance feedback
            for i in range(len(expert_agents)):
                refined_response = await adaptive_agent(
                    [taskInfo, all_thinking[i], all_answers[i]], adaptive_instruction
                )
                all_thinking[i], all_answers[i] = refined_response
            iteration += 1
        # Instruction for synthesizing the refined insights into a final decision
        consensus_instruction = "Given all the refined insights, carefully reason over them and provide a final answer to the task."
        consensus_agent = LLMAgentBase(
            ["thinking", "answer"], "Consensus Agent", temperature=0.1
        )
        # Generate the final answer
        final_response = await consensus_agent(
            [taskInfo] + all_thinking + all_answers, consensus_instruction
        )
        thinking_final, answer_final = final_response
        return answer_final.content


if __name__ == "__main__":
    GPQAAgentSystem = GPQAAgentSystem()
    # Set a timeout of 3 minutes (180 seconds)
    input = "What is the capital of france?"
    output = asyncio.run(GPQAAgentSystem.forward(input))
    output = str(output)
    print(output)
