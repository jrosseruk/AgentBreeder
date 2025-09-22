<div align="center">

  <img src="assets/agentbreeder_no_background.png" alt="AgentBreeder" width="200" height="auto" />
  <h1>AgentBreeder</h1>

  <p>
    <strong>Mitigating the AI Safety Impact of Multi-Agent Scaffolds via Self-Improvement</strong>
  </p>

  <!-- NeurIPS Badge -->
  <p>
    <a href="https://openreview.net/forum?id=10342">
      <img src="https://img.shields.io/badge/NeurIPS%202025-Spotlight-red?style=for-the-badge" alt="NeurIPS 2025 Spotlight" />
    </a>
  </p>

  <!-- Standard Badges -->
  <p>
    <a href="https://github.com/jrosseruk/AgentBreeder/contributors">
      <img src="https://img.shields.io/github/contributors/jrosseruk/AgentBreeder" alt="contributors" />
    </a>
    <a href="">
      <img src="https://img.shields.io/github/last-commit/jrosseruk/AgentBreeder" alt="last update" />
    </a>
    <a href="https://github.com/jrosseruk/AgentBreeder/network/members">
      <img src="https://img.shields.io/github/forks/jrosseruk/AgentBreeder" alt="forks" />
    </a>
    <a href="https://github.com/jrosseruk/AgentBreeder/stargazers">
      <img src="https://img.shields.io/github/stars/jrosseruk/AgentBreeder" alt="stars" />
    </a>
    <a href="https://github.com/jrosseruk/AgentBreeder/issues/">
      <img src="https://img.shields.io/github/issues/jrosseruk/AgentBreeder" alt="open issues" />
    </a>
    <a href="https://github.com/jrosseruk/AgentBreeder/blob/master/LICENSE">
      <img src="https://img.shields.io/github/license/jrosseruk/AgentBreeder.svg" alt="license" />
    </a>
  </p>

  <h4>
    <a href="https://jrosser.co.uk/AgentBreeder/">🌐 Website</a>
    <span> · </span>
    <a href="https://iclr.cc/media/iclr-2025/Slides/10000479.pdf">📖 Documentation</a>
    <span> · </span>
    <a href="https://arxiv.org/abs/2502.00757">📄 Paper</a>
    <span> · </span>
    <a href="https://github.com/jrosseruk/AgentBreeder/issues/">🐛 Report Bug</a>
    <span> · </span>
    <a href="https://github.com/jrosseruk/AgentBreeder/issues/">✨ Request Feature</a>
  </h4>
</div>

<br />

<div align="center">
  <img src="assets/AgentBreederDiagramJPG.jpg" alt="AgentBreeder Framework" width="100%" height="auto" />
</div>

## 🤝 Get in Touch

**Interested in collaborating on AI safety research?** We're actively looking for collaborators to extend AgentBreeder and explore new frontiers in multi-agent safety. Whether you're working on:

- 🔬 **Safety evaluation benchmarks** - Help us develop better safety metrics
- 🏗️ **Multi-agent architectures** - Explore novel scaffold designs
- 🎯 **Red/blue team methodologies** - Advance adversarial testing approaches
- 📊 **Evaluation frameworks** - Improve our assessment capabilities

**Reach out to us:** [jrosser.co.uk](https://jrosser.co.uk)


## 🏆 NeurIPS 2025 Spotlight

**AgentBreeder** was accepted as a **spotlight paper** at NeurIPS 2025 and previouly an Oral at the ICLR 2025 SSI-FM Workshop!

## 🚀 Key Achievements

- **🎯 79.4% average uplift** in safety benchmark performance while maintaining capability
- **🔬 First framework** for multi-objective evolutionary search over multi-agent scaffolds
- **⚖️ Balanced optimization** of safety and capability through Pareto optimization
- **🔴 Red-team insights** revealing how capable scaffolds can become highly vulnerable
- **📊 Comprehensive evaluation** across DROP, MMLU, GPQA, and SaladData benchmarks

## 📋 Abstract

Scaffolding Large Language Models (LLMs) into multi-agent systems often improves performance on complex tasks, but the safety impact of such scaffolds has not been thoroughly explored. We introduce **AgentBreeder**, a framework for multi-objective self-improving evolutionary search over scaffolds, specifically targeting scaffolds' safety impact on large language models in multi-agent systems.

We evaluate discovered scaffolds on widely recognized reasoning, mathematics, and safety benchmarks. In **'blue' mode**, we see a **79.4% average uplift** in safety benchmark performance while maintaining or improving capability scores. In **'red' mode**, we find adversarially weak scaffolds emerging concurrently with capability optimization. Our work demonstrates the risks of multi-agent scaffolding and provides a framework for mitigating them.

## 🎭 Three Operational Modes

AgentBreeder operates in three distinct modes, each serving different research and deployment needs:

### 🔵 BlueAgentBreeder (Defense)
- **Objective**: Maximize both safety and capability
- **Use case**: Develop robust, safe multi-agent systems
- **Result**: 79.4% average safety improvement while maintaining capability

### 🔴 RedAgentBreeder (Attack)
- **Objective**: Maximize capability while minimizing safety
- **Use case**: Red-team testing and vulnerability discovery
- **Result**: Reveals how scaffolding can inadvertently expose safety weaknesses

### 🎯 CapableAgentBreeder (Capability)
- **Objective**: Maximize capability only
- **Use case**: Baseline comparison and pure performance optimization
- **Result**: Competitive performance with existing approaches

## 🛠️ Installation

### Docker Installation (Recommended)
