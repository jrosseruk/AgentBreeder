<div align="center">

  <img src="assets/agentbreeder_no_background.png" alt="AgentBreeder" width="200" height="auto" />
  <h1>AgentBreeder</h1>
  
  <p>
    AgentBreeder: Mitigating the AI Safety Impact of Multi-Agent Scaffolds
  </p>
  
  
<!-- Badges -->
<p>
  <a href="https://github.com/J-Rosser-UK/AgentBreeder/contributors">
    <img src="https://img.shields.io/github/contributors/J-Rosser-UK/AgentBreeder" alt="contributors" />
  </a>
  <a href="">
    <img src="https://img.shields.io/github/last-commit/J-Rosser-UK/AgentBreeder" alt="last update" />
  </a>
  <a href="https://github.com/J-Rosser-UK/AgentBreeder/network/members">
    <img src="https://img.shields.io/github/forks/J-Rosser-UK/AgentBreeder" alt="forks" />
  </a>
  <a href="https://github.com/J-Rosser-UK/AgentBreeder/stargazers">
    <img src="https://img.shields.io/github/stars/J-Rosser-UK/AgentBreeder" alt="stars" />
  </a>
  <a href="https://github.com/J-Rosser-UK/AgentBreeder/issues/">
    <img src="https://img.shields.io/github/issues/J-Rosser-UK/AgentBreeder" alt="open issues" />
  </a>
  <a href="https://github.com/J-Rosser-UK/AgentBreeder/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/J-Rosser-UK/AgentBreeder.svg" alt="license" />
  </a>
</p>
   
<h4>
    <!-- <a href="https://github.com/J-Rosser-UK/AgentBreeder/">View Demo</a> -->
  <!-- <span> · </span> -->
    <a href="https://docs.google.com/presentation/d/197lRGAtPoG1NWLJ_fDOLTHBlyz9eA6G35g-XNvyb9To/edit?usp=sharing">Documentation</a>
  <span> · </span>
    <a href="https://github.com/J-Rosser-UK/AgentBreeder/issues/">Report Bug</a>
  <span> · </span>
    <a href="https://github.com/J-Rosser-UK/AgentBreeder/issues/">Request Feature</a>
  </h4>
</div>

<br />

 <img src="assets/AgentBreederDiagramJPG.jpg" alt="AgentBreeder" width="auto" height="auto" />

## Abstract

Scaffolding Large Language Models (LLMs) into multi-agent systems often improves performance on complex tasks, but the safety impact of such scaffolds has not been as thoroughly explored. In this paper, we introduce AGENTBREEDER a framework for multi-objective evolutionary search over scaffolds. Our REDAGENTBREEDER evolves scaffolds towards jailbreaking the base LLM while achieving high task success, while BLUEAGENTBREEDER instead aims to combine safety with task reward. We evaluate the systems discovered by the different instances of AGENTBREEDER and popular baselines using widely recognized reasoning, mathematics, and safety benchmarks. Our work highlights and mitigates the safety risks due to multi-agent scaffolding.

## Directory Structure
```
AgentBreeder/
  ├── README.md
  ├── Dockerfile
  ├── LICENSE
  ├── requirements.txt
  ├── .env.sample
  ├── .flake8
  ├── assets/
  │   └── source.txt
  └── src/
      ├── eval.py
      ├── main.py
      ├── base/
      │   ├── __init__.py
      │   ├── base.py
      │   ├── session.py
      │   └── tables.py
      ├── chat/
      │   ├── __init__.py
      │   ├── api.py
      │   ├── chat.py
      │   ├── test_api.py
      │   └── test_api_requests.py
      ├── descriptor/
      │   ├── __init__.py
      │   └── descriptor.py
      ├── evals/
      │   ├── __init__.py
      │   ├── arc.py
      │   ├── benchmark.py
      │   ├── clrs_text.py
      │   ├── drop.py
      │   ├── gpqa.py
      │   ├── metrics.py
      │   ├── mgsm.py
      │   ├── mmlu.py
      │   ├── test_arc.py
      │   ├── test_clrs_text.py
      │   ├── test_mmlu.py
      │   └── validator.py
      ├── generator/
      │   ├── __init__.py
      │   ├── generator.py
      │   └── mutator.py
      ├── illuminator/
      │   ├── __init__.py
      │   └── illuminator.py
      ├── prompts/
      │   ├── initial_population.py
      │   ├── mutation_base.py
      │   ├── mutation_prompts.py
      │   ├── mutation_reflexion.py
      │   └── utils.py
      └── visualizations/
          ├── tree.py
          └── visualizer.py

```

## Run with Docker
```
git clone https://github.com/J-Rosser-UK/AgentBreeder

cd AgentBreeder

sudo docker build -t agent_breeder .

sudo docker run -it agent_breeder

```


## Run directly
```
git clone https://github.com/J-Rosser-UK/AgentBreeder

cd AgentBreeder

cd src

python -m venv venv 

venv/Scripts/activate // windows

source venv/bin/activate // unix

pip install -r requirements.txt

### in a different terminal ###
python src/chat/api.py
###############################
 
python src/main.py --population_id None
```

