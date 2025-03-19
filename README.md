# Awesome-Math-LLM 

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of resources dedicated to Large Language Models (LLMs) for mathematics, mathematical reasoning, and mathematical problem-solving.

<!-- <p align="center">
  <img src="imgs/math-llm-banner.png" width="800px"></img>
</p> -->


> [!NOTE]
> This repository is being developed incrementally. See our [Incremental Development Plan](INCREMENTAL_DEVELOPMENT_PLAN.md) for details on our phased approach.
> 
> We welcome contributions! Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

## Table of Contents

- [1. Surveys](#1-surveys)
- [2. Base Models](#2-base-models)
- [3. Mathematical Reasoning Methods](#3-mathematical-reasoning-methods)
  - [3.1 Reasoning Paradigms](#31-reasoning-paradigms)
  - [3.2 Training Methodologies](#32-training-methodologies)
  - [3.3 Technical Implementations](#33-technical-implementations)
- [4. Mathematical Domains and Applications](#4-mathematical-domains-and-applications)
  - [4.1 Word Problems](#41-word-problems)
  - [4.2 Formal Mathematics and Theorem Proving](#42-formal-mathematics-and-theorem-proving)
  - [4.3 Geometry Problems](#43-geometry-problems)
  - [4.4 Educational Mathematics](#44-educational-mathematics)
  - [4.5 Applied Mathematics](#45-applied-mathematics)
- [5. Evaluation, Benchmarks and Datasets](#5-evaluation-benchmarks-and-datasets)
  - [5.1 General Mathematical Benchmarks](#51-general-mathematical-benchmarks)
  - [5.2 Domain-Specific Benchmarks](#52-domain-specific-benchmarks)
  - [5.3 Multimodal Mathematics Datasets](#53-multimodal-mathematics-datasets)
  - [5.4 Instruction Tuning Datasets](#54-instruction-tuning-datasets)
- [6. Tools and Libraries](#6-tools-and-libraries)
- [7. Analysis, Limitations and Future Directions](#7-analysis-limitations-and-future-directions)
  - [7.1 Capability Analysis](#71-capability-analysis)
  - [7.2 Known Limitations](#72-known-limitations)
  - [7.3 Emerging Research Directions](#73-emerging-research-directions)

<!-- ## Current Development Phase

We are currently in **Phase 1: Foundation** of our development plan, focusing on:
1. Foundational LLMs with Mathematical Capabilities
2. Primary Benchmarks and Datasets
3. Basic Mathematical Reasoning Approaches -->

### Recent Highlights

[2025-03]

- **ERNIE**: "Baidu Unveils ERNIE 4.5 and Reasoning Model ERNIE X1" [[website](https://yiyan.baidu.com/)]
- **QaQ**: "QwQ-32B: Embracing the Power of Reinforcement Learning" [[blog](https://qwenlm.github.io/blog/qwq-32b/)] [[repo](https://github.com/QwenLM/QwQ)]


## 1. Surveys

*Meta-analyses and survey papers about LLMs for mathematics*

- MLLM Survey: "A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges" [2024-12] [paper]
- DL4Math: "A Survey of Deep Learning for Mathematical Reasoning" [2022-12] [paper]
- LLM4Math: "Large Language Models for Mathematical Reasoning: Progresses and Challenges" [2024-02] [paper]

<!-- For detailed summaries of these surveys, see our [Survey Summaries document](docs/survey_summaries.md). -->

## 2. Base Models
This section contains a list of base LLMs that are designed for general-purpose tasks, but also have demonstrated mathematical capabilities. 

### 2.1 Publicly Available Models

**LLAMA**

- **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models" [2023-02] [[paper](https://arxiv.org/abs/2302.13971)]
- **LLaMA 2**: "Llama 2: Open Foundation and Fine-Tuned Chat Models" [2023-07] [[paper](https://arxiv.org/abs/2307.09288)] [[repo](https://github.com/facebookresearch/llama)]
- **LLaMA 3**: "The Llama 3 Herd of Models" [2024-04] [[blog](https://ai.meta.com/blog/meta-llama-3/)] [[repo](https://github.com/meta-llama/llama3)] [[paper](https://arxiv.org/abs/2407.21783)]

**Qwen**

- **Qwen**: "Qwen Technical Report" [2023-09] [[paper](https://arxiv.org/abs/2309.16609)] [[repo](https://github.com/QwenLM/Qwen)]
- **Qwen 2**: "Qwen2: A Family of Open-Source LLMs with 14B-128B Parameters" [2024-09] [[paper](https://arxiv.org/abs/2409.12488)] [[repo](https://github.com/QwenLM/Qwen)]
- **Qwen2.5**: "Qwen2.5 Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.15115)]    

**DeepSeek**

- **DeepSeek**: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism" [2024-01] [[paper](https://arxiv.org/abs/2401.02954)] [[repo](https://github.com/deepseek-ai/DeepSeek-LLM)]

- **DeepSeekMoE**: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" [2024-01] [[paper](https://arxiv.org/abs/2401.12246)] [[repo](https://github.com/deepseek-ai/DeepSeek-MoE)]

- **DeepSeek-V2**: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" [2024-05] [[paper](https://arxiv.org/abs/2405.04434)] [[repo](https://github.com/deepseek-ai/DeepSeek-V2)]

- **DeepSeek-V3**: "DeepSeek-V3 Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.19437)]

**Gemma**

- **Gemma**: "Gemma: Open Models Based on Gemini Research and Technology" [2024-02] [[paper](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)] [[blog](https://blog.google/technology/developers/gemma-open-models/)]

- **Gemma 2**: "Gemma 2: Improving Open Language Models at a Practical Size" [2024-06] [[paper](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)]

**Mistral**

- **Mistral**: "Mistral 7B" [2023-10] [[paper](https://arxiv.org/abs/2310.06825)] [[repo](https://github.com/mistralai/mistral-src)]

**Phi**

- **Phi-1.5**: "Textbooks Are All You Need II: phi-1.5 technical report" [2023-09] [[paper](https://arxiv.org/abs/2309.05463)] [[model](https://huggingface.co/microsoft/phi-1_5)]

- **Phi-2**: "Phi-2: The surprising power of small language models" [2023-12] [[blog](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)]

- **Phi-3**: "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone" [2024-04] [[paper](https://arxiv.org/abs/2404.14219)]

- **Phi-4**: "Phi-4 Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.08905)]

- **Phi-4-Mini**: "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs" [2025-03] [[paper](https://arxiv.org/abs/2503.01743)]

**Baichuan**

- **Baichuan 2**: "Baichuan 2: Open Large-scale Language Models" [2023-09] [[paper](https://arxiv.org/abs/2309.10305)] [[repo](https://github.com/baichuan-inc/Baichuan2)]


**Others**

- **GPT-NeoX**: "GPT-NeoX-20B: An Open-Source Autoregressive Language Model" [2022-04] [ACL 2022 Workshop on Challenges & Perspectives in Creating LLMs] [[paper](https://arxiv.org/abs/2204.06745)] [[repo](https://github.com/EleutherAI/gpt-neox)]

- **BLOOM**: "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model" [2022-11] [[paper](https://arxiv.org/abs/2211.05100)] [[model](https://huggingface.co/models?search=bigscience/bloom)]

- **YAYI2**: "YAYI 2: Multilingual Open-Source Large Language Models" [2023-12] [[paper](https://arxiv.org/abs/2312.14862)] [[repo](https://github.com/wenge-research/YAYI2)]

- **Mixtral**: "Mixtral of Experts" [2024-01] [[paper](https://arxiv.org/abs/2401.04088)] [[blog](https://mistral.ai/news/mixtral-of-experts/)]

- **Orion**: "Orion-14B: Open-source Multilingual Large Language Models" [2024-01] [[paper](https://arxiv.org/abs/2401.06066)] [[repo](https://github.com/OrionStarAI/Orion)] 

- **OLMo**: "OLMo: Accelerating the Science of Language Models" [2024-02] [[paper](https://arxiv.org/abs/2402.00838)] [[repo](https://github.com/allenai/OLMo)]

- **Yi**: "Yi: Open Foundation Models by 01.AI" [2024-03] [[paper](https://arxiv.org/abs/2403.04652)] [[repo](https://github.com/01-ai/Yi)]

- **OLMoE**: "OLMoE: Open Mixture-of-Experts Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.02060)][[repo](https://github.com/allenai/OLMoE?tab=readme-ov-file#pretraining)]

- **Yi-Lightning**: "Yi-Lightning Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.01253)

- **YuLan-Mini**: "YuLan-Mini: An Open Data-efficient Language Model" [2024-12] [[paper](https://arxiv.org/abs/2412.17743)]

- **OLMo 2**: "2 OLMo 2 Furious" [2024-12] [[paper](https://arxiv.org/abs/2501.00656)]

- **SmolLM2**: "SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model" [2025-02] [[paper](https://arxiv.org/abs/2502.02737)]

### 2.2 Closed-Source (Or Commercial) Models

**GPT** [[link](https://openai.com/)]
- **GPT-3**: "Language Models are Few-Shot Learners" [2020-05] [[paper](https://arxiv.org/abs/2005.14165)]

- **GPT-4**: "GPT-4 Technical Report" [2023-03] [[paper](https://arxiv.org/abs/2303.08774)]

- **GPT-4o**: "GPT-4o System Card" [2024-10] [[paper](https://arxiv.org/abs/2410.21276)]

- **GPT-4.5**: "Introducing GPT-4.5" [2025-02] [[blog](https://openai.com/index/introducing-gpt-4-5/)]

**Claude** [[link](https://www.anthropic.com/)]

- **Claude 3**: "The Claude 3 Model Family: Opus, Sonnet, Haiku" [2024-03] [[paper](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)] [[blog](https://www.anthropic.com/news/claude-3-family)]

- **Claude 3.5**: "Claude 3.5 Sonnet Model Card Addendum" [2024-10] [[paper](https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/Model_Card_Claude_3_Addendum.pdf)] [[blog](https://www.anthropic.com/claude/haiku)]

- **Claude 3.7**: "Claude 3.7 Sonnet System Card" [2025-03] [[paper](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.7_Sonnet.pdf)] [[blog](https://www.anthropic.com/news/claude-3-7-sonnet)]

**Gemini** [[link](https://gemini.google.com/)]

- **Gemini 1.5 Pro**: "Gemini 1.5 Unlocking multimodal understanding across millions of tokens of context" [2024-05] [[paper](https://arxiv.org/abs/2403.05530)] [[blog](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#gemini-15)]

- **Gemini 2**: "Gemini 1.5 Flash: Fast and Efficient Multimodal Reasoning" [2024-05] [[blog](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/)]

**Grok** [[link](https://grok.com/)]

- **Grok 1**: "Grok 1: Open Release of Grok-1" [2024-03] [[blog](https://x.ai/news/grok-os)]

- **Grok 2**: "Grok-2 Beta Release" [2024-08] [[blog](https://x.ai/news/grok-2)]

- **Grok 3**: "Grok 3 Beta — The Age of Reasoning Agents" [2025-02] [[blog](https://x.ai/news/grok-3)]

### 2.3 Reasoning Models

- **DeepSeek-R1**: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" [2025-01] [[paper](https://arxiv.org/abs/2501.12948)] [[repo](https://github.com/deepseek-ai/DeepSeek-R1)] 

- **QaQ**: "QwQ-32B: Embracing the Power of Reinforcement Learning" [2025-03] [[blog](https://qwenlm.github.io/blog/qwq-32b/)] [[repo](https://github.com/QwenLM/QwQ)]

- **ERNIE X1**: "Baidu Unveils ERNIE 4.5 and Reasoning Model ERNIE X1" [2025-03] [[website](https://yiyan.baidu.com/)]

- **Gemini 2.0 Flash Thinking**: "Gemini 2.0 Flash Thinking" [2025-03] [[blog](https://deepmind.google/technologies/gemini/flash-thinking/)]


### 2.4 Multimodal Models

### 2.5 Key LLM Techniques and Methods

- **LoRA**: "LoRA: Low-Rank Adaptation of Large Language Models" [2021-06] [[paper](https://arxiv.org/abs/2106.09685)]

- **MoE**: "Mixture of Experts" [2023-11] [[blog](https://huggingface.co/blog/moe)]

- **COT**: "Chain of Thought Prompting Elicits Reasoning in Large Language Models" [2022-01] [[paper](https://arxiv.org/abs/2201.11903)]

- **Self-Consistency**: "Self-Consistency Improves Chain of Thought Reasoning in Language Models" [2022-05] [[paper](https://arxiv.org/abs/2203.11171)]

- **DPO**: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" [2023-05] [[paper](https://arxiv.org/abs/2305.18290)]

- **RLHF**: "Training language models to follow instructions with human feedback" [2022-05] [[paper](https://arxiv.org/abs/2203.02155)]

- **GRPO**: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" [2024-02] [[paper](https://arxiv.org/abs/2402.03300)]


## 3. Mathematical Reasoning Methods

*Core approaches and methodologies for enhancing mathematical reasoning in Large Language Models*

### 3.1 Reasoning Paradigms

*Fundamental techniques for mathematical problem-solving with LLMs*

- **Chain-of-Thought**: "Chain of Thought Prompting Elicits Reasoning in Large Language Models" [2022-01] [[paper](https://arxiv.org/abs/2201.11903)]

- **Self-Consistency**: "Self-Consistency Improves Chain of Thought Reasoning in Language Models" [2022-05] [[paper](https://arxiv.org/abs/2203.11171)]

- **Tree of Thoughts**: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" [2023-05] [[paper](https://arxiv.org/abs/2305.10601)]

- **Tool-Augmented Reasoning**: "ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving" [2023-11] [[paper](https://arxiv.org/abs/2309.17452)]

- **Program-Based Methods**: "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks" [2022-11] [[paper](https://arxiv.org/abs/2211.12588)]

- **Search-Based Methods**: "Enhancing LLM Reasoning with Reward-guided Tree Search" [2023-10] [[paper](https://arxiv.org/abs/2310.09177)]

- **Verification Methods**: "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations" [2023-11] [[paper](https://arxiv.org/abs/2308.07615)]

- **Multi-Turn Reasoning**: "MathChat: Benchmarking Mathematical Reasoning and Instruction Following in Multi-Turn Interactions" [2023-11] [[paper](https://arxiv.org/abs/2306.01337)]

### 3.2 Training Methodologies

*Methods for training and fine-tuning LLMs for improved mathematical reasoning*

- **RLHF**: "Training language models to follow instructions with human feedback" [2022-05] [[paper](https://arxiv.org/abs/2203.02155)]

- **DPO**: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" [2023-05] [[paper](https://arxiv.org/abs/2305.18290)]

- **Step-DPO**: "Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs" [2023-11] [[paper](https://arxiv.org/abs/2310.06671)]

- **GRPO**: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" [2024-02] [[paper](https://arxiv.org/abs/2402.03300)]

- **Process Supervision**: "AlphaMath Almost Zero: Process Supervision without Process" [2023-12] [[paper](https://arxiv.org/abs/2312.00481)]

- **Specialized Pre-training**: "Rho-1: Not All Tokens Are What You Need" [2023-10] [[paper](https://arxiv.org/abs/2310.11715)]

- **Data Generation**: "MathScale: Scaling Instruction Tuning for Mathematical Reasoning" [2023-12] [[paper](https://arxiv.org/abs/2312.17120)]

- **Knowledge Distillation**: "JiuZhang3.0: Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models" [2023-10] [[paper](https://arxiv.org/abs/2310.06692)]

### 3.3 Technical Implementations

*Implementation details and techniques that impact mathematical reasoning performance*

- **Iterative Improvement**: "Building Math Agents with Multi-Turn Iterative Preference Learning" [2023-10] [[paper](https://arxiv.org/abs/2310.03731)]

- **Test-Time Scaling**: "Mathematical Discoveries from Program Search with Large Language Models" [2023-11] [[paper](https://arxiv.org/abs/2311.13444)]

- **Monte Carlo Methods**: "Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning" [2023-10] [[paper](https://arxiv.org/abs/2310.14197)]

- **Output Space Shaping**: "ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving" [2023-11] [[paper](https://arxiv.org/abs/2309.17452)]

- **Selective Loss Functions**: "Rho-1: Not All Tokens Are What You Need" [2023-10] [[paper](https://arxiv.org/abs/2310.11715)]

- **Code Interpreters**: "Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-Based Self-Verification" [2023-08] [[paper](https://arxiv.org/abs/2308.07921)]

## 4. Mathematical Domains and Applications

*Different mathematical domains and applications where LLMs are applied*

### 4.1 Word Problems

- **GSM8K Solving**: "Training Verifiers to Solve Math Word Problems" [2021-10] [[paper](https://arxiv.org/abs/2110.14168)]

- **Scratchpad Reasoning**: "Show Your Work: Scratchpads for Intermediate Computation with Language Models" [2021-12] [[paper](https://arxiv.org/abs/2112.00114)]

- **Pseudo-Dual Learning**: "Pseudo-Dual Learning: Solving Math Word Problems with Reexamination" [2023-10] [[paper](https://arxiv.org/abs/2310.04292)]

- **Symbolic Reasoning**: "Reasoning in Large Language Models Through Symbolic Math Word Problems" [2023-05] [[paper](https://arxiv.org/abs/2305.20050)]

### 4.2 Formal Mathematics and Theorem Proving

- **GPT-f**: "GPT-f: Generative Language Modeling for Automated Theorem Proving" [2021-05] [[paper](https://arxiv.org/abs/2009.03393)]

- **miniF2F**: "miniF2F: A Cross-System Benchmark for Formal Olympiad-Level Mathematics" [2022-09] [[paper](https://arxiv.org/abs/2109.00110)]

- **COPRA**: "COPRA: COllaborative PRoof Assistant with GPT-4" [2024-02] [[paper](https://arxiv.org/abs/2402.10108)]

- **Formal Mathematics**: "Formal Mathematical Reasoning: A New Frontier in AI" [2023-11] [[paper](https://arxiv.org/abs/2306.03544)]

- **Llemma**: "Llemma: An Open Language Model For Mathematics" [2023-10] [[paper](https://arxiv.org/abs/2310.10631)]

### 4.3 Geometry Problems

- **AlphaGeometry**: "Solving Olympiad Geometry without Human Demonstrations" [2024-01] [[paper](https://www.nature.com/articles/s41586-023-06747-5)]

- **UniGeo**: "Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression" [2022-10] [[paper](https://arxiv.org/abs/2210.01196)]

- **Inter-GPS**: "Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning" [2021-06] [[paper](https://arxiv.org/abs/2105.04166)]

### 4.4 Educational Mathematics

- **Early Education**: "CMATH: Can Your Language Model Pass Chinese Elementary School Math Test?" [2023-06] [[paper](https://arxiv.org/abs/2306.16636)]

- **Educational Applications**: "Three Questions Concerning the Use of Large Language Models to Facilitate Mathematics Learning" [2023-10] [[paper](https://arxiv.org/abs/2310.03078)]

- **Pre-service Teachers**: "Exploring Pre-service Teachers' Perceptions of Large Language Models-Generated Hints in Online Mathematics Learning" [2023-06] [[paper](https://arxiv.org/abs/2306.17129)]

### 4.5 Applied Mathematics

- **Multidisciplinary Reasoning**: "Solving Quantitative Reasoning Problems with Language Models" [2023-05] [[paper](https://arxiv.org/abs/2206.14858)]

- **Competition-Level Problems**: "OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems" [2023-11] [[paper](https://arxiv.org/abs/2311.07575)]

## 5. Evaluation, Benchmarks and Datasets

*Benchmarks, datasets, and evaluation methodologies for mathematical LLMs*

### 5.1 General Mathematical Benchmarks

- **MATH Dataset**: "Measuring Mathematical Problem Solving With the MATH Dataset" [2021-03] [[paper](https://arxiv.org/abs/2103.03874)]

- **GSM8K Dataset**: "Training Verifiers to Solve Math Word Problems" [2021-10] [[paper](https://arxiv.org/abs/2110.14168)]

- **LILA**: "LILA: A Unified Benchmark for Mathematical Reasoning" [2022-10] [[paper](https://arxiv.org/abs/2210.17517)]

- **CHAMP**: "CHAMP: Mathematical Reasoning with Chain of Thought in Large Language Models" [2024-05] [[paper](https://arxiv.org/abs/2405.19254)]

### 5.2 Domain-Specific Benchmarks

- **MathQA**: "MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms" [2022-05] [[paper](https://aclanthology.org/N19-1245/)]

- **GeomVerse**: "GeomVerse: A Systematic Evaluation of Large Models for Geometric Reasoning" [2023-12] [[paper](https://arxiv.org/abs/2312.12241)]

- **TABMWP**: "Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning" [2023-01] [[paper](https://arxiv.org/abs/2209.14610)]

- **AMPS**: "Measuring Mathematical Problem Solving With the MATH Dataset" [2021-03] [[paper](https://arxiv.org/abs/2103.03874)]

- **ROBUSTMATH**: "MATHATTACK: Attacking Large Language Models Towards Math Solving Ability" [2023-09] [[paper](https://arxiv.org/abs/2309.05690)]

### 5.3 Multimodal Mathematics Datasets

- **MATHVISTA**: "MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts" [2023-10] [[paper](https://arxiv.org/abs/2310.02255)]

- **MathVerse**: "MathVerse: Assessing Visual Mathematical Understanding in Multimodal LLMs" [2024-04] [[paper](https://arxiv.org/abs/2404.13834)]

- **ErrorRadar**: "ErrorRadar: Evaluating the Multimodal Error Detection of LLMs in Educational Settings" [2024-03] [[paper](https://arxiv.org/abs/2403.03894)]

- **MathV360K**: "Math-LLaVA: A Multimodal Math QA Dataset with 360K Instances" [2024-04] [[paper](https://arxiv.org/abs/2404.02693)]

### 5.4 Instruction Tuning Datasets

- **OpenMathInstruct-1**: "OpenMathInstruct-1: A 1.8M Math Instruction Tuning Dataset" [2024-03] [[paper](https://arxiv.org/abs/2402.10176)]

- **OpenMathInstruct-2**: "OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data" [2023-10] [[paper](https://arxiv.org/abs/2310.12591)]

- **MATH-Instruct**: "Mammoth: Building Math Generalist Models Through Hybrid Instruction Tuning" [2023-09] [[paper](https://arxiv.org/abs/2309.05653)]

- **MAVIS-Instruct**: "MAVIS: Multimodal Automatic Visual Instruction Synthesis for Math Problem Solving" [2024-04] [[paper](https://arxiv.org/abs/2404.04473)]

## 6. Tools and Libraries

*Software tools and frameworks for mathematical LLMs*

- **LPML**: "LPML: LLM-Prompting Markup Language for Mathematical Reasoning" [2023-09] [[paper](https://arxiv.org/abs/2309.04269)]

- **Math-CodeInterpreter**: "Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-Based Self-Verification" [2023-08] [[paper](https://arxiv.org/abs/2308.07921)]

- **MathPrompter**: "MathPrompter: Mathematical Reasoning Using Large Language Models" [2023-07] [[paper](https://arxiv.org/abs/2303.05398)]

- **BEATS**: "BEATS: Optimizing LLM Mathematical Capabilities with BackVerify and Adaptive Disambiguate based Efficient Tree Search" [2023-10] [[paper](https://arxiv.org/abs/2310.04344)]

- **rStar-Math**: "rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking" [2023-10] [[paper](https://arxiv.org/abs/2310.14570)]

- **BoostStep**: "BoostStep: Boosting mathematical capability of Large Language Models via improved single-step reasoning" [2023-10] [[paper](https://arxiv.org/abs/2310.08573)]

## 7. Analysis, Limitations and Future Directions

*Research on capabilities, limitations, and future directions of LLMs for mathematics*

### 7.1 Capability Analysis

- **Model Analysis**: "Common 7B Language Models Already Possess Strong Math Capabilities" [2023-10] [[paper](https://arxiv.org/abs/2310.04560)]

- **Scaling Relationships**: "Scaling Relationship on Learning Mathematical Reasoning with Large Language Models" [2023-10] [[paper](https://arxiv.org/abs/2310.07177)]

- **Tokenization Impact**: "How Well Do Large Language Models Perform in Arithmetic Tasks?" [2023-04] [[paper](https://arxiv.org/abs/2304.02015)]

### 7.2 Known Limitations

- **Robustness Analysis**: "A Causal Framework to Quantify the Robustness of Mathematical Reasoning with Language Models" [2023-05] [[paper](https://arxiv.org/abs/2305.14291)]

- **Error Analysis**: "Learning from Mistakes Makes LLM Better Reasoner" [2023-10] [[paper](https://arxiv.org/abs/2310.13522)]

- **Attacks on Math Ability**: "MATHATTACK: Attacking Large Language Models Towards Math Solving Ability" [2023-09] [[paper](https://arxiv.org/abs/2309.05690)]

### 7.3 Emerging Research Directions

- **Cross-Modal Reasoning**: "Math-PUMA: Progressive Upward Multimodal Alignment for Math Reasoning Enhancement" [2024-04] [[paper](https://arxiv.org/abs/2404.01166)]

- **Human-Centric Design**: "Exploring Pre-service Teachers' Perceptions of Large Language Models-Generated Hints in Online Mathematics Learning" [2023-06] [[paper](https://arxiv.org/abs/2306.17129)]

- **Continual Learning**: "Learning from Mistakes Makes LLM Better Reasoner" [2023-10] [[paper](https://arxiv.org/abs/2310.13522)]

- **Program Search**: "Mathematical Discoveries from Program Search with Large Language Models" [2023-11] [[paper](https://arxiv.org/abs/2311.13444)]

## How to Contribute

We are looking for contributors to help build this resource. If you know of relevant papers, datasets, tools, or other resources, please consider contributing by:

1. Checking our [Incremental Development Plan](INCREMENTAL_DEVELOPMENT_PLAN.md) to see what phase we're in
2. Following our [contribution guidelines](CONTRIBUTING.md)
3. Submitting a pull request with your additions -->

## Citation

If you find this repository useful, please consider citing:

```bibtex
@misc{awesome-math-llm,
  author = {doublelei},
  title = {Awesome-Math-LLM},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/doublelei/Awesome-Math-LLM}}
}
```

## License

This repository is licensed under the [MIT License](LICENSE).