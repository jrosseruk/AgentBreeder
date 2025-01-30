multi_agent_system_mutation_prompts = [
    "Inside the system, add a step which restates and elaborates on the inquiry before proceeding with a response.",
    "Inside the system, add a step which reframes the question with additional context and detail before working out the answer.",
    "Let the agents know they will be rewarded for providing a concise and accurate answer to the question.",
    "Change the predefined roles or personas of the agents involved in the system. For instance, switch an agent's role from 'Proponent' to 'Skeptic' or assign new characteristics like expertise in a specific domain.",
    "Adjust the rules governing how agents interact. This could include changing the number of debate rounds, the order of speaking, or introducing new interaction phases such as rebuttal or evidence presentation.",
    "Change the step-back abstraction layers used by the agents. This might involve altering how much reasoning is abstracted, the granularity of explanations, or the depth of recursive thinking.",
    "Modify the initial prompts or instructions given to each agent. This could include rephrasing goals, making them more detailed, introducing new guidelines, or emphasizing different aspects of their character.",
    "Add new agents with unique roles or remove existing ones from the debate system. For example, introduce a 'Moderator' agent or eliminate a less effective participant.",
    "Change the reward structure or scoring system used to evaluate agent performance. This could involve adjusting the weights assigned to different criteria, introducing new evaluation metrics, or modifying the feedback mechanism.",
    "Alter the format of meetings by introducing new structures or modifying existing ones. For example, switch from a traditional debate format to a panel discussion, round-robin, or brainstorming session.",
    "Change which agents participate in each meeting. This can include rotating participants, introducing new agents, or forming specialized sub-groups.",
    "Create multi-tiered meeting structures where higher-level meetings oversee or coordinate lower-level ones. For example, strategic meetings could direct tactical meetings.",
    "Introduce various types of meetings, such as decision-making, information-sharing, conflict-resolution, or creative ideation sessions.",
    "Replace fixed roles with dynamic role allocation based on agent performance",
    "Introduce hierarchical agents with higher-level agents overseeing groups of lower-level agents",
    "Simulate natural ecosystems by introducing predator-prey dynamics",
    "Introduce trust mechanisms where agents evaluate reliability of others",
    "Introduce asymmetrical power dynamics by giving certain agents additional capabilities",
]


multi_agent_system_safety_mutation_prompts = [
    "Incorporate a consensus-checking step where agents must verify each other's responses for accuracy and adherence to ethical guidelines before finalizing an answer.",
    "Introduce a diversity-promoting mechanism that ensures a variety of perspectives are represented, thereby preventing bias and promoting fairness in decision-making processes.",
    "Implement a transparency layer where agents must provide clear justifications for their actions and decisions, fostering accountability and trust within the system.",
    "Embed an efficient monitoring feature that assesses agent interactions for compliance with privacy and confidentiality standards, safeguarding sensitive information.",
    "Establish an ethical oversight agent responsible for evaluating the potential impacts of decisions on human well-being and societal values, ensuring alignment with the constitution.",
    "Create role-based restrictions that limit agents from performing actions beyond their designated responsibilities, preventing misuse of authority and maintaining system integrity.",
    "Introduce a fail-safe mechanism that returns a default output if unethical behavior or harmful patterns are detected, allowing for timely intervention and correction.",
]
