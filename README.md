# Awesome-Math-LLM

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of resources dedicated to Large Language Models (LLMs) for mathematics, mathematical reasoning, and mathematical problem-solving.

> We welcome contributions! Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

## Table of Contents

- [1. 📚 Surveys & Overviews](#1--surveys--overviews)
  - [1.1 Related Awesome Lists](#11-related-awesome-lists)
- [2. 🧮 Mathematical Tasks & Foundational Capabilities](#2--mathematical-tasks--foundational-capabilities)
  - [2.1 Fundamental Calculation & Representation](#21-fundamental-calculation--representation)
  - [2.2 Arithmetic & Word Problems](#22-arithmetic--word-problems)
  - [2.3 Algebra, Geometry, Calculus, etc.](#23-algebra-geometry-calculus-etc)
  - [2.4 Competition Math](#24-competition-math)
  - [2.5 Formal Theorem Proving](#25-formal-theorem-proving)
- [3. 🧠 Core Reasoning & Problem-Solving Techniques](#3--core-reasoning--problem-solving-techniques)
  - [3.1 Chain-of-Thought & Prompting Strategies](#31-chain-of-thought--prompting-strategies)
  - [3.2 Search & Planning](#32-search--planning)
  - [3.3 Reinforcement Learning & Reward Modeling](#33-reinforcement-learning--reward-modeling)
  - [3.4 Self-Improvement & Self-Training](#34-self-improvement--self-training)
  - [3.5 Tool Use & Augmentation](#35-tool-use--augmentation)
  - [3.6 Neurosymbolic Methods & Solver Integration](#36-neurosymbolic-methods--solver-integration)
- [4. 👁️ Multimodal Mathematical Reasoning](#4-%EF%B8%8F-multimodal-mathematical-reasoning)
- [5. 🤖 Models](#5--models)
  - [5.1 Math-Specialized LLMs](#51-math-specialized-llms)
  - [5.2 Reasoning-Focused LLMs](#52-reasoning-focused-llms)
  - [5.3 Leading General LLMs](#53-leading-general-llms)
- [6. 📊 Datasets & Benchmarks](#6--datasets--benchmarks)
  - [6.1 Problem Solving Benchmarks](#61-problem-solving-benchmarks)
  - [6.2 Theorem Proving Benchmarks](#62-theorem-proving-benchmarks)
  - [6.3 Multimodal Benchmarks](#63-multimodal-benchmarks)
  - [6.4 Training Datasets](#64-training-datasets)
  - [6.5 Augmented / Synthetic Datasets](#65-augmented--synthetic-datasets)
- [7. 🛠️ Tools & Libraries](#7-%EF%B8%8F-tools--libraries)
- [8. 🤝 Contributing](#8--contributing)
- [9. 📄 Citation](#9--citation)
- [10. ⚖️ License](#10-%EF%B8%8F-license)

---

### Recent Highlights *(Note: Dates appear to be futuristic)*

* **[2025-03] Survey (Math Reasoning & Optimization):** "A Survey on Mathematical Reasoning and Optimization with Large Language Models" ([Paper](https://arxiv.org/abs/2503.17726)) - **Key resource for this list!**
* **[2025-03] ERNIE:** "Baidu Unveils ERNIE 4.5 and Reasoning Model ERNIE X1" ([Website](https://yiyan.baidu.com/))
* **[2025-03] QaQ:** "QwQ-32B: Embracing the Power of Reinforcement Learning" ([Blog](https://qwenlm.github.io/blog/qwq-32b/)) [Repo](https://github.com/QwenLM/QwQ)

---

## 1. 📚 Surveys & Overviews

*Meta-analyses and survey papers about LLMs for mathematics.*

* "A Survey on Mathematical Reasoning and Optimization with Large Language Models" ([Paper](https://arxiv.org/abs/2503.17726)) - *(March 2025)*
* "A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics" ([Paper](https://arxiv.org/abs/2502.14333)) - *(February 2025)*
* "From System 1 to System 2: A Survey of Reasoning Large Language Models" ([Paper](https://arxiv.org/abs/2502.17419)) - *(February 2025)*
* **Survey (Multimodal):** "A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model: Benchmark, Method & Challenges" ([Paper](https://arxiv.org/abs/2412.11936)) - *(December 2024)*
* "Large Language Models for Mathematical Reasoning: Progresses and Challenges" ([Paper](https://arxiv.org/abs/2402.00157)) - *(February 2024)*
* **Survey (Formal Math):** "Formal Mathematical Reasoning: A New Frontier in AI" ([Paper](https://arxiv.org/abs/2306.03544)) - *(June 2023)*
* "A Survey of Deep Learning for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2212.10535)) - *(December 2022)*

### 1.1 Related Awesome Lists

*Other curated lists focusing on relevant areas.*

* "Awesome LLM Reasoning" ([GitHub](https://github.com/atfortes/Awesome-LLM-Reasoning))
* "Awesome System 2 Reasoning LLM" ([GitHub](https://github.com/zzli2022/Awesome-System2-Reasoning-LLM))
* "Awesome Multimodal LLM for Math/STEM" ([GitHub](https://github.com/InfiMM/Awesome-Multimodal-LLM-for-Math-STEM))

## 2. 🧮 Mathematical Tasks & Foundational Capabilities

> This section outlines the fundamental capabilities LLMs need for mathematics (Calculation & Representation) and the major mathematical reasoning domains they are applied to. Resources are often categorized by the primary domain addressed.

### 2.1 Fundamental Calculation & Representation

*Focuses on how LLMs process, represent, and compute basic numerical operations. Challenges here underpin performance on more complex tasks.*

* **FoNE:** "FoNE: Precise Single-Token Number Embeddings via Fourier Features" ([Paper](https://arxiv.org/abs/2502.09741)) - *(February 2025)*
* "Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling" ([Paper](https://arxiv.org/abs/2501.16975)) - *(January 2025)*
* "Arithmetic Transformers Can Length-Generalize in Both Operand Length and Count" ([Paper](https://arxiv.org/abs/2410.15787)) - *(October 2024)*
* "Language Models Encode Numbers Using Digit Representations in Base 10" ([Paper](https://arxiv.org/abs/2410.11781)) - *(October 2024)*
* **MathGLM (RevOrder):** "RevOrder: A Novel Method for Enhanced Arithmetic in Language Models" ([Paper](https://arxiv.org/abs/2402.03822)) - *(February 2024)*
* "Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs" ([Paper](https://arxiv.org/abs/2402.14903)) - *(February 2024)*
* "Length Generalization in Arithmetic Transformers" ([Paper](https://arxiv.org/abs/2306.15400)) - *(June 2023)*
* **GOAT:** "Goat: Fine-tuned LLaMA Outperforms GPT-4 on Arithmetic Tasks" ([Paper](https://arxiv.org/abs/2305.14201)) - *(May 2023)*
* "How well do large language models perform in arithmetic tasks?" ([Paper](https://arxiv.org/abs/2304.02015)) - *(April 2023)*
* "Teaching algorithmic reasoning via in-context learning" ([Paper](https://arxiv.org/abs/2211.09066)) - *(November 2022)*
* **Scratchpad:** "Show Your Work: Scratchpads for Intermediate Computation with Language Models" ([Paper](https://arxiv.org/abs/2112.00114)) - *(December 2021)*

### 2.2 Arithmetic & Word Problems

*Solving grade-school to high-school level math word problems, requiring understanding context and applying arithmetic/algebraic steps.*

* *Key Benchmarks:* GSM8K, SVAMP, AddSub/ASDiv, MultiArith, Math23k, TabMWP, MR-GSM8K (See Section 6.1 for details)

* **UPFT:** "The First Few Tokens Are All You Need: An Efficient and Effective Unsupervised Prefix Fine-Tuning Method for Reasoning Models" ([Paper](https://arxiv.org/abs/2503.02875)) - *(March 2025)*
* **ArithmAttack:** "ArithmAttack: Evaluating Robustness of LLMs to Noisy Context in Math Problem Solving" ([Paper](https://arxiv.org/abs/2501.08203)) - *(January 2025)*
* **MetaMath:** "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models" ([Paper](https://arxiv.org/abs/2309.12284)) ([Code](https://github.com/meta-math/MetaMath)) - *(September 2023)*
* **WizardMath:** "WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct" ([Paper](https://arxiv.org/abs/2308.09583)) ([HF Models](https://huggingface.co/WizardLM/WizardMath-70B-V1.0)) - *(August 2023)*
* "Let's Verify Step by Step" ([Paper](https://arxiv.org/abs/2305.20050)) - *(May 2023)*
* **MathPrompter:** "MathPrompter: Mathematical Reasoning using Large Language Models" ([Paper](https://arxiv.org/abs/2303.05398)) - *(March 2023)*

### 2.3 Algebra, Geometry, Calculus, etc.

*Problems spanning standard high school and undergraduate curricula in core mathematical subjects.*

* *Key Benchmarks:* MATH, SciBench, MMLU (Math Subsets) (See Section 6.1 for details)

* **AlphaGeometry:** "AlphaGeometry: An Olympiad-level AI system for geometry" ([Blog Post](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/)) - *(January 2024)*
* **Llemma:** "Llemma: An Open Language Model For Mathematics" ([Paper](https://arxiv.org/abs/2310.10631)) ([HF Models](https://huggingface.co/EleutherAI/llemma_7b)) - *(October 2023)*
* **UniGeo:** "Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression" ([Paper](https://arxiv.org/abs/2210.01196)) - *(October 2022)*
* **Inter-GPS:** "Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning" ([Paper](https://arxiv.org/abs/2105.04165)) - *(May 2021)*

### 2.4 Competition Math

*Challenging problems from competitions like AMC, AIME, IMO, Olympiads, often requiring creative reasoning.*

* *Key Benchmarks:* MATH (Competition subset), AIME, OlympiadBench, miniF2F (Formal) (See Section 6.1, 6.2 for details)

* "Brains vs. Bytes: Evaluating LLM Proficiency in Olympiad Mathematics" ([Paper](https://arxiv.org/abs/2504.01995)) - *(April 2025)*
* "Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad" ([Paper](https://arxiv.org/abs/2503.21934)) - *(March 2025)*
* **AlphaGeometry2:** "Gold-medalist Performance in Solving Olympiad Geometry with AlphaGeometry2" ([Paper](https://arxiv.org/abs/2502.03544)) - *(February 2025)*
* **AlphaGeometry:** "AlphaGeometry: An Olympiad-level AI system for geometry" ([Blog Post](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/)) - *(January 2024)*

### 2.5 Formal Theorem Proving

*Generating and verifying formal mathematical proofs using Interactive Theorem Provers (ITPs).*

* *Key Benchmarks:* miniF2F, ProofNet, NaturalProofs, HolStep, CoqGym, LeanStep, INT, FOLIO, MathConstruct (See Section 6.2 for details)

* **LeanNavigator:** "Generating Millions Of Lean Theorems With Proofs By Exploring State Transition Graphs" ([Paper](https://arxiv.org/abs/2503.04772)) - *(March 2025)*
* **MathConstruct Benchmark:** "MathConstruct: Challenging LLM Reasoning with Constructive Proofs" ([Paper](https://arxiv.org/abs/2502.10197)) - *(February 2025)*
* **BFS-Prover:** "BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving" ([Paper](https://arxiv.org/abs/2502.03438)) - *(February 2025)*
* **Llemma:** "Llemma: An Open Language Model For Mathematics" ([Paper](https://arxiv.org/abs/2310.10631)) ([HF Models](https://huggingface.co/EleutherAI/llemma_7b)) - *(October 2023)*
* "Draft, sketch, and prove: Guiding formal theorem provers with informal proofs" ([Paper](https://arxiv.org/abs/2210.12283)) - *(October 2022)*
* **GPT-f:** "Generative Language Modeling for Automated Theorem Proving" ([Paper](https://arxiv.org/abs/2009.03393)) - *(September 2020)*
* **CoqGym:** "Learning to Prove Theorems via Interacting with Proof Assistants" ([Paper](https://arxiv.org/abs/1905.09381)) - *(May 2019)*
* **DeepMath:** "DeepMath - Deep Sequence Models for Premise Selection" ([Paper](https://arxiv.org/abs/1606.04442)) - *(June 2016)*

## 3. 🧠 Core Reasoning & Problem-Solving Techniques

> This section details the core techniques and methodologies used by LLMs to reason and solve mathematical problems ('HOW' problems are solved), often applicable across multiple domains.

### 3.1 Chain-of-Thought & Prompting Strategies

*Techniques involving generating step-by-step reasoning, structuring prompts effectively, and iterative refinement/correction within the generation process.*

* **BoostStep:** "Boosting mathematical capability of Large Language Models via improved single-step reasoning" ([Paper](https://arxiv.org/abs/2501.03226)) - *(January 2025)*
* **Rank-verifier:** "Rank-verifier Agreement Disentangles Large Language Model Capabilities" ([Paper](https://arxiv.org/abs/2405.01316)) - *(May 2024)*
* **MCR:** "Improving Mathematical Reasoning with Multi-agent Consensus" ([Paper](https://arxiv.org/abs/2404.06174)) - *(April 2024)*
* **ISR-LLM:** "ISR-LLM: Iterative Self-Refinement with Large Language Models for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2402.00801)) - *(February 2024)*
* **GRACE:** "GRACE: Generate, Reason, Act, Check - A Framework for Improving Large Language Model Reasoning" ([Paper](https://arxiv.org/abs/2402.12728)) - *(February 2024)*
* **BoostedPrompt:** "BoostedPrompt: A Boosting Method for Few-Shot Prompting" ([Paper](https://arxiv.org/abs/2311.14029)) - *(November 2023)*
* **PromptPG-CoT:** "PromptPG: Prompt Engineering via Policy Gradient for Math Reasoning with Large Language Models" ([Paper](https://arxiv.org/abs/2311.07310)) - *(November 2023)*
* **CR (Conditional Rationale):** "CR: Improving LLM Mathematical Reasoning with Conditional Rationale" ([Paper](https://arxiv.org/abs/2310.13308)) - *(October 2023)*
* "Long CoT: Long Chain-of-Thought for Complex Problems" ([Paper](https://arxiv.org/abs/2310.03050)) - *(October 2023)*
* **LPML:** "LPML: LLM-Prompting Markup Language for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2309.04269)) - *(September 2023)*
* **Self-Check:** "SelfCheck: Using LLMs to Zero-Shot Check Their Own Step-by-Step Reasoning" ([Paper](https://arxiv.org/abs/2308.00436)) - *(August 2023)*
* **Step-Plan:** "Step-by-Step Planning Improves Large Language Model Reasoning" ([Paper](https://arxiv.org/abs/2305.12577)) - *(May 2023)*
* **Diversity-of-Thought:** "Making Large Language Models Better Reasoners with Step-Aware Verifier" ([Paper](https://arxiv.org/abs/2305.17755)) - *(May 2023)*
* **Self-Refine:** "Self-Refine: Iterative Refinement with Self-Feedback" ([Paper](https://arxiv.org/abs/2303.17651)) - *(March 2023)*
* **Reflexion:** "Reflexion: Language Agents with Verbal Reinforcement Learning" ([Paper](https://arxiv.org/abs/2303.11366)) - *(March 2023)*
* **MathPrompter:** "MathPrompter: Mathematical Reasoning using Large Language Models" ([Paper](https://arxiv.org/abs/2303.05398)) - *(March 2023)*
* **Faithful CoT:** "Faithful Chain-of-Thought Reasoning" ([Paper](https://arxiv.org/abs/2301.13379)) - *(January 2023)*
* **Algorithmic Prompting:** "Teaching language models to reason algorithmically" ([Blog Post](https://research.google/blog/teaching-language-models-to-reason-algorithmically/)) - *(November 2022)*
* "Teaching algorithmic reasoning via in-context learning" ([Paper](https://arxiv.org/abs/2211.09066)) - *(November 2022)* (Related to Algorithmic Prompting)
* **Self-Consistency:** "Self-Consistency Improves Chain of Thought Reasoning in Language Models" ([Paper](https://arxiv.org/abs/2203.11171)) - *(March 2022)*
* **Chain-of-Thought (CoT):** "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" ([Paper](https://arxiv.org/abs/2201.11903)) - *(January 2022)*

### 3.2 Search & Planning

*Techniques explicitly exploring multiple potential solution paths or intermediate steps, often building tree or graph structures.*

* **BFS-Prover:** "BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving" ([Paper](https://arxiv.org/abs/2502.03438)) - *(February 2025)*
* **Reward-guided Tree Search (STILL-1):** "Enhancing LLM Reasoning with Reward-guided Tree Search" ([Paper](https://arxiv.org/abs/2411.11694)) - *(November 2024)*
* **Q* Framework:** "Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning" ([Paper](https://arxiv.org/abs/2405.03052)) - *(May 2024)*
* **Learning Planning-based Reasoning:** "Learning Planning-based Reasoning by Trajectories Collection and Process Reward Synthesizing" ([Paper](https://arxiv.org/abs/2402.11771)) - *(February 2024)*
* **Language Agent Tree Search (LATS):** "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models" ([Paper](https://arxiv.org/abs/2310.04406)) - *(October 2023)*
* **BEATS:** "BEATS: Optimizing LLM Mathematical Capabilities with BackVerify and Adaptive Disambiguate based Efficient Tree Search" ([Paper](https://arxiv.org/abs/2310.04344)) - *(October 2023)*
* **Graph of Thoughts (GoT):** "Graph of Thoughts: Solving Elaborate Problems with Large Language Models" ([Paper](https://arxiv.org/abs/2308.09687)) ([Code](https://github.com/spcl/graph-of-thoughts)) - *(August 2023)*
* **Tree of Thoughts (ToT):** "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" ([Paper](https://arxiv.org/abs/2305.10601)) ([Code](https://github.com/kyegomez/Tree-of-Thoughts-LLM)) - *(May 2023)*
* **Reasoning via Planning (RAP):** "Reasoning with Language Model is Planning with World Model" ([Paper](https://arxiv.org/abs/2305.14992)) - *(May 2023)*

### 3.3 Reinforcement Learning & Reward Modeling

*Using RL algorithms (e.g., PPO, DPO) and feedback mechanisms (e.g., RLHF, Process Reward Models - PRM) to train models based on preferences or process correctness.*

* "Lessons Learned on Language Models for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2501.07301)) - *(January 2025)*
* **Preference Optimization (Pseudo Feedback):** "Preference Optimization for Reasoning with Pseudo Feedback" ([Paper](https://arxiv.org/abs/2411.16345)) - *(November 2024)*
* **Step-Controlled DPO (SCDPO):** "Step-Controlled DPO: Leveraging Stepwise Error for Enhanced Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2406.04579)) - *(June 2024)*
* **Step-DPO:** "Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs" ([Paper](https://arxiv.org/abs/2406.18629)) - *(June 2024)*
* **SuperCorrect:** "SuperCorrect: Supervising and Correcting Language Models with Error-Driven Insights" ([Paper](https://arxiv.org/abs/2405.18542)) ([Code](https://github.com/YangLing0818/SuperCorrect-llm)) - *(May 2024)*
* **SVPO:** "Step-level Value Preference Optimization for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2405.18357)) ([Code](https://github.com/MARIO-Math-Reasoning/Super_MARIO)) - *(May 2024)*
* **LLaMA-Berry:** "LLaMA-Berry: Pairwise Optimization for O1-like Olympiad-Level Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2405.18045)) - *(May 2024)*
* **OmegaPRM:** "OmegaPRM: Enhancing Process Supervision with Outcome Guidance" ([Paper](https://arxiv.org/abs/2405.19725)) - *(May 2024)*
* **HGS-PRM:** "Hierarchical Granularity Supervision for Mathematical Process Reward Models" ([Paper](https://arxiv.org/abs/2405.19028)) - *(May 2024)*
* **Flow-DPO:** "Flow-DPO: Aligning Language Models via Flow Matching on Preference Trajectories" ([Paper](https://arxiv.org/abs/2405.16058)) - *(May 2024)*
* **AlphaMath Almost Zero:** "AlphaMath Almost Zero: process Supervision without process" ([Paper](https://arxiv.org/abs/2405.03553)) - *(May 2024)*
* **Math-Minos:** "LLM Critics Help Catch Bugs in Mathematics: Towards a Better Mathematical Verifier with Natural Language Feedback" ([Paper](https://arxiv.org/abs/2405.11186)) ([Code](https://github.com/DAMO-NLP-SG/LLM-Critics-Math)) - *(May 2024)*
* **Collaborative Verification:** "Improving LLM Reasoning through Scaling Inference Computation with Collaborative Verification" ([Paper](https://arxiv.org/abs/2404.07928)) - *(April 2024)*
* **MCTS-DPO:** "Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning" ([Paper](https://arxiv.org/abs/2402.10770)) ([Code](https://github.com/YuxiXie/MCTS-DPO)) - *(February 2024)*
* **SCoRe (Self-Correction):** "Self-Correction with Optimal Transport for Reasoning Tasks" ([Paper](https://arxiv.org/abs/2402.12689)) - *(February 2024)*
* **GRPO (DeepSeekMath):** "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" ([Paper](https://arxiv.org/abs/2402.03300)) - *(February 2024)*
* **Math-Shepherd:** "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations" ([Paper](https://arxiv.org/abs/2312.08935)) - *(December 2023)*
* **RL Alignment (KTO / NCA):** "EURUS: Scaling Down Deep Equilibrium Models" ([Paper](https://arxiv.org/abs/2311.18231)) - *(November 2023)*
* **Process Supervision (PRM800K):** "Solving Math Word Problems with Process- and Outcome-Based Feedback" ([Paper](https://arxiv.org/abs/2305.15062)) - *(May 2023)*
* **Process Supervision (Verify Step-by-Step):** "Let's Verify Step by Step" ([Paper](https://arxiv.org/abs/2305.20050)) - *(May 2023)*
* **DPO Algorithm:** "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" ([Paper](https://arxiv.org/abs/2305.18290)) - *(May 2023)*
* **RLHF (General):** "Training language models to follow instructions with human feedback" ([Paper](https://arxiv.org/abs/2203.02155)) - *(March 2022)*
* **RL for Optimization:** "Learning to Optimize with Reinforcement Learning" ([Paper](https://arxiv.org/abs/2103.01148)) - *(March 2021)* (Survey Example)
* **PPO Algorithm:** "Proximal Policy Optimization Algorithms" ([Paper](https://arxiv.org/abs/1707.06347)) - *(July 2017)*

### 3.4 Self-Improvement & Self-Training

*Methods where models iteratively generate data, reflect on outcomes or process, and refine their reasoning abilities, often employing techniques from Sec 3.1 & 3.3.*

* **rStar-Math:** "Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking" ([Paper](https://arxiv.org/abs/2502.10000)) ([Code](https://github.com/microsoft/rStar)) - *(February 2025)*
* **V-STaR:** "V-STaR: Training Verifiers for Self-Taught Reasoners" ([Paper](https://arxiv.org/abs/2405.09859)) - *(May 2024)*
* **Quiet-STaR:** "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking" ([Paper](https://arxiv.org/abs/2403.09629)) - *(March 2024)*
* **Self-Correction (SCoRe/MCTSr):** "Self-Correction with Optimal Transport for Reasoning Tasks" ([Paper](https://arxiv.org/abs/2402.12689)) - *(February 2024)*
* **CoRe (Self-Correction Controller):** "Self-Correction Controller for Large Language Models" ([Paper](https://arxiv.org/abs/2310.12357)) - *(October 2023)*
* **ReST:** "Reinforced Self-Training (ReST) for Language Modeling" ([Paper](https://arxiv.org/abs/2308.08998)) - *(August 2023)*
* **RFT (Scaling Relationship):** "Scaling Relationship on Learning Mathematical Reasoning with Large Language Models" ([Paper](https://arxiv.org/abs/2308.01825)) - *(August 2023)*
* **STaR:** "Self-Taught Reasoner (STaR): Bootstrapping Reasoning With Reasoning" ([Paper](https://arxiv.org/abs/2203.14465)) - *(March 2022)*
* *Note: Methods like Self-Refine, Reflexion (listed in Sec 3.1) also implement self-improvement loops.*

### 3.5 Tool Use & Augmentation

*Enabling LLMs to call external computational or knowledge tools like calculators, code interpreters, search engines, solvers, and planners.*

* **MuMath-Code:** "MuMath-Code: A Multilingual Mathematical Problem Solving Dataset with Code Solutions" ([Paper](https://arxiv.org/abs/2405.00742)) - *(May 2024)*
* "Large Language Models for Operations Research: A Survey" ([Paper](https://arxiv.org/abs/2402.04889)) - *(February 2024)* (Discusses Solver Integration)
* **MARIO Pipeline:** "MARIO: MAth Reasoning with code Interpreter Output - A Reproducible Pipeline" ([Paper](https://arxiv.org/abs/2401.11171)) ([Code](https://github.com/MARIO-Math-Reasoning/MARIO)) - *(January 2024)*
* **MAmmoTH:** "MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning" ([Paper](https://arxiv.org/abs/2401.11445)) ([HF Dataset](https://huggingface.co/datasets/TIGER-Lab/MAmmoTH)) - *(January 2024)*
* **ToRA:** "ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving" ([Paper](https://arxiv.org/abs/2309.17452)) ([Code](https://github.com/microsoft/ToRA) - *(September 2023)*
* "Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-Based Self-Verification" ([Paper](https://arxiv.org/abs/2308.07921)) - *(August 2023)* (Evaluating GPT-4 Code Interpreter)
* **GPT-4 Code Interpreter:** (Official Release/Plugin) ([Blog Post](https://openai.com/blog/chatgpt-plugins#code-interpreter)) - *(March 2023)*
* **ART:** "ART: Automatic multi-step reasoning and tool-use for large language models" ([Paper](https://arxiv.org/abs/2303.09014)) ([Code (Guidance)](https://github.com/microsoft/guidance)) - *(March 2023)*
* **Toolformer:** "Toolformer: Language Models Can Teach Themselves to Use Tools" ([Paper](https://arxiv.org/abs/2302.04761)) - *(February 2023)*
* **PoT (Program of Thoughts):** "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks" ([Paper](https://arxiv.org/abs/2211.12588)) - *(November 2022)*
* **PAL (Program-Aided LM):** "Program-Aided Language Models" ([Paper](https://arxiv.org/abs/2211.10435)) ([Code](https://github.com/reasoning-machines/pal)) - *(November 2022)*

### 3.6 Neurosymbolic Methods & Solver Integration

*Methods focusing on deeper integration between neural models and symbolic representations, reasoning systems (like ITPs, formal logic), or solvers beyond simple tool calls.*

* "Symbolic Mixture-of-Experts" ([Paper](https://arxiv.org/abs/2503.05641)) - *(March 2025)*
* "Discovering Symbolic Operators from Partial Differential Equations with Neural Networks" ([Paper](https://arxiv.org/abs/2503.09986)) - *(March 2025)*
* **CRANE:** "CRANE: Constrained Decoding for Neuro-Symbolic Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2502.09061)) - *(February 2025)*
* "Integrating Vector Symbolic Architectures with Transformers for Explainable Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2502.01657)) - *(February 2025)*
* **LISA:** "LISA: Language Models Integrate Symbolic Abstractions" ([Paper](https://arxiv.org/abs/2405.16557)) - *(May 2024)*
* **LLM+DP:** "From Words to Actions: Empowering Large Language Models to Follow Instructions with Deep Planning" ([Paper](https://arxiv.org/abs/2311.06623)) - *(November 2023)*
* **LINC:** "LINC: A Neurosymbolic Approach for Logical Reasoning by Combining Language Models with First-Order Logic Provers" ([Paper](https://arxiv.org/abs/2310.01066)) - *(October 2023)*
* **SatLM:** "SatLM: Satisfiability-Aided Language Models" ([Paper](https://arxiv.org/abs/2310.05726)) - *(October 2023)*
* **LeanReasoner:** "LeanReasoner: Boosting Logical Reasoning via Prompting Large Language Models" ([Paper](https://arxiv.org/abs/2305.10111)) - *(May 2023)*
* **Logic-LM:** "Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning" ([Paper](https://arxiv.org/abs/2305.12295)) - *(May 2023)*
* **LLM+P:** "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency" ([Paper](https://arxiv.org/abs/2304.11477)) - *(April 2023)*
* **Inter-GPS:** "Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning" ([Paper](https://arxiv.org/abs/2105.04165)) - *(May 2021)*
* **GPT-f:** "Generative Language Modeling for Automated Theorem Proving" ([Paper](https://arxiv.org/abs/2009.03393)) - *(September 2020)*

## 4. 👁️ Multimodal Mathematical Reasoning

> This section focuses on the specific challenges and approaches for mathematical reasoning when non-textual information (images, diagrams, tables) is involved.

* **Survey (Multimodal):** "A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model" ([Paper](https://arxiv.org/abs/2412.11936)) - *(December 2024)*

* *Key Benchmarks:* MathVista, ScienceQA, MATH-Vision, MathVerse, GeoQA/GeoEval, FigureQA, ChartQA, MM-MATH (See Section 6.3 for details)

* **UnAC:** "Unified Abductive Cognition for Multimodal Reasoning" ([Paper](https://arxiv.org/abs/2405.17550)) - *(May 2024)*
* **MAVIS-Instruct:** "MAVIS: Multimodal Automatic Visual Instruction Synthesis for Math Problem Solving" ([Paper](https://arxiv.org/abs/2404.04473)) - *(April 2024)*
* **MathV360K Dataset (Math-LLaVA):** "Math-LLaVA: Bootstrapping Mathematical Reasoning for Large Vision Language Models" ([Paper](https://arxiv.org/abs/2404.02693)) ([Dataset](https://huggingface.co/datasets/Zhiqiang007/MathV360K)) - *(April 2024)*
* **Math-PUMA:** "Math-PUMA: Progressive Upward Multimodal Alignment for Math Reasoning Enhancement" ([Paper](https://arxiv.org/abs/2404.01166)) - *(April 2024)*
* **ErrorRadar:** "ErrorRadar: Evaluating the Multimodal Error Detection of LLMs in Educational Settings" ([Paper](https://arxiv.org/abs/2403.03894)) - *(March 2024)*
* **MathGLM-Vision:** "Solving Mathematical Problems with Multi-Modal Large Language Model" ([Paper](https://arxiv.org/abs/2402.04503)) - *(February 2024)*

* *Models (Examples):* GPT-4V, Gemini Pro Vision, Qwen-VL, LLaVA variants (LLaVA-o1), AtomThink, M-STAR, GLM-4V (See Section 5.3 for details)

## 5. 🤖 Models

> This section lists the specific Large Language Models relevant to mathematical tasks.
> *Note: Classification and details partly informed by Table 1 in survey arXiv:2503.17726.*

### 5.1 Math-Specialized LLMs

*Models specifically pre-trained or fine-tuned for mathematical tasks, often incorporating math-specific data or techniques.*

* **JiuZhang3.0:** "Efficiently Improving Mathematical Reasoning by Training Small Data Synthesis Models" ([Paper](https://arxiv.org/abs/2405.14365)) ([Code](https://github.com/RUCAIBox/JiuZhang3.0)) - *(May 2024)*
* **DART-MATH Models:** "DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving" ([Paper](https://arxiv.org/abs/2405.14194)) - *(May 2024)*
* **Skywork-Math Models:** "Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models - The Story Goes On" ([Paper](https://arxiv.org/abs/2405.10814)) - *(May 2024)*
* **ControlMath:** "ControlMath: Mathematical Reasoning with Process Supervision and Outcome Guidance" ([Paper](https://arxiv.org/abs/2405.19725)) - *(May 2024)*
* **ChatGLM-Math:** "ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline" ([Paper](https://arxiv.org/abs/2404.06864)) ([GitHub](https://github.com/THUDM/ChatGLM-Math)) - *(April 2024)*
* **Rho-1:** "Rho-1: Not All Tokens Are What You Need" ([Paper](https://arxiv.org/abs/2404.07965)) - *(April 2024)*
* **MathCoder2 Models:** "MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code" ([Paper](https://arxiv.org/abs/2404.11081)) ([Code](https://github.com/mathllm/MathCoder2)) - *(April 2024)*
* **Qwen-Math / Qwen2.5-Math:** "Qwen2.5: Advancing Large Language Models for Code, Math, Multilingualism, and Long Context" ([Paper](https://arxiv.org/abs/2406.13559)) ([HF Models](https://huggingface.co/Qwen)) - *(June 2024)*
* **DeepSeekMath:** "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" ([Paper](https://arxiv.org/abs/2402.03300)) ([GitHub](https://github.com/deepseek-ai/DeepSeek-Math)) - *(February 2024)*
* **InternLM-Math:** "InternLM-Math: Open Math Large Language Models Toward Verifiable Reasoning" ([Paper](https://arxiv.org/abs/2402.15296)) ([HF Models](https://huggingface.co/internlm)) - *(February 2024)*
* **Llemma:** "Llemma: An Open Language Model For Mathematics" ([Paper](https://arxiv.org/abs/2310.10631)) ([HF Models](https://huggingface.co/EleutherAI/llemma_7b)) - *(October 2023)*
* **MetaMath:** "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models" ([Paper](https://arxiv.org/abs/2309.12284)) ([Code](https://github.com/meta-math/MetaMath)) - *(September 2023)*
* **WizardMath:** "WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct" ([Paper](https://arxiv.org/abs/2308.09583)) ([HF Models](https://huggingface.co/WizardLM/WizardMath-70B-V1.0)) - *(August 2023)*
* **PaLM 2-L-Math:** "PaLM 2 Technical Report" ([Paper](https://ai.google/static/documents/palm2techreport.pdf)) - *(May 2023)*
* **MathGLM:** "MathGLM: Mathematical Reasoning with Goal-driven Language Models" ([Paper](https://arxiv.org/abs/2305.15112)) - *(May 2023)*
* **MATHDIAL:** "MATHDIAL: A Dialogue-Based Pre-training Approach for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2305.11865)) - *(May 2023)*
* **MATH-PLM:** "MATH-PLM: Pre-training Language Models for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2209.04193)) - *(September 2022)*
* **Minerva:** "Minerva: Solving Quantitative Reasoning Problems with Language Models" ([Blog Post](https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html)) - *(June 2022)*
* **Codex-math:** "Evaluating Large Language Models Trained on Code" ([Paper](https://arxiv.org/abs/2107.03374)) - *(July 2021)*
* **GPT-f:** "Generative Language Modeling for Automated Theorem Proving" ([Paper](https://arxiv.org/abs/2009.03393)) - *(September 2020)*

### 5.2 Reasoning-Focused LLMs

*Models explicitly optimized for complex reasoning tasks, often via advanced RL, search, self-improvement, or specialized architectures.*

* **QaQ:** "QwQ-32B: Embracing the Power of Reinforcement Learning" ([Blog Post](https://qwenlm.github.io/blog/qwq-32b/)) ([GitHub](https://github.com/QwenLM/QwQ)) - *(March 2025)*
* **ERNIE X1:** "Baidu Unveils ERNIE 4.5 and Reasoning Model ERNIE X1" ([Website](https://yiyan.baidu.com/)) - *(March 2025)*
* **rStar-Math Models:** "Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking" ([Paper](https://arxiv.org/abs/2502.10000)) ([Code](https://github.com/microsoft/rStar)) - *(February 2025)* (e.g., rStar-Math-7B-v2 [HF Model](https://huggingface.co/agi-templar/rStar-Math-7B-v2))
* **DeepSeek R1:** "DeepSeek-R1: Pushing the Limits of General Reasoning in Open Large Language Models" ([Paper](https://arxiv.org/abs/2501.12948)) ([GitHub](https://github.com/deepseek-ai/DeepSeek-R1)) - *(January 2025)*
* **Gemini 2.0 Flash Thinking:** "Flash Thinking: Real-time reasoning with Gemini 2.0" ([Blog Post](https://deepmind.google/technologies/gemini/flash-thinking/)) - *(December 2024)*
* **OpenAI o1:** "Introducing o1" ([Blog Post](https://openai.com/index/introducing-o1/)) - *(September 2024)*
* **Marco-o1:** "Marco-o1: An Open Reasoning Model Trained with Process Supervision" ([Paper](https://arxiv.org/abs/2406.17439)) ([HF Models](https://huggingface.co/Nexusflow/Marco-o1-7B)) - *(June 2024)*
* **SocraticLLM:** "SocraticLLM: Iterative Chain-of-Thought Distillation for Large Language Models" ([Paper](https://arxiv.org/abs/2405.10927)) - *(May 2024)*
* **EURUS:** "EURUS: Scaling Down Deep Equilibrium Models" ([Paper](https://arxiv.org/abs/2311.18231)) ([HF Models](https://huggingface.co/Nexusflow)) - *(November 2023)*

### 5.3 Leading General LLMs

*General-purpose models frequently evaluated on mathematical benchmarks. Includes base models for many specialized versions.*

**OpenAI**

* **GPT-4.5:** "Introducing GPT-4.5" ([Blog Post](https://openai.com/index/introducing-gpt-4-5/)) - *(December 2024)*
* **GPT-4o:** "GPT-4o System Card" ([Paper](https://arxiv.org/abs/2410.21276)) - *(October 2024)*
* **GPT-4V:** "GPT-4V(ision) System Card" ([Paper](https://cdn.openai.com/papers/GPTV_System_Card.pdf)) - *(September 2023)*
* **GPT-4:** "GPT-4 Technical Report" ([Paper](https://arxiv.org/abs/2303.08774)) - *(March 2023)*
* **GPT-3:** "Language Models are Few-Shot Learners" ([Paper](https://arxiv.org/abs/2005.14165)) - *(May 2020)*

**Google**

* **Gemini 2:** "Gemini 2: Unlocking multimodal intelligence at scale" ([Blog Post](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/)) - *(December 2024)*
* **Gemma 2:** "Gemma 2 Technical Report" ([Paper](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)) - *(June 2024)*
* **Gemini 1.5 Pro:** "Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context" ([Paper](https://arxiv.org/abs/2403.05530)) - *(February 2024)*
* **Gemma:** "Gemma: Open Models Based on Gemini Research and Technology" ([Paper](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)) - *(February 2024)*
* **Gemini 1.0:** "Gemini: A Family of Highly Capable Multimodal Models" ([Paper](https://arxiv.org/abs/2312.11805)) - *(December 2023)*
* **Flan-PaLM:** "Scaling Instruction-Finetuned Language Models" ([Paper](https://arxiv.org/abs/2210.11416)) - *(October 2022)*
* **PaLM:** "PaLM: Scaling Language Modeling with Pathways" ([Paper](https://arxiv.org/abs/2204.02311)) - *(April 2022)*

**Anthropic**

* **Claude 3.7:** "Claude 3.7 Model Card" ([Paper](https://www-cdn.anthropic.com/claude-3.7-model-card.pdf)) - *(December 2024)*
* **Claude 3.5:** "Claude 3.5 Sonnet Model Card Addendum" ([Paper](https://www-cdn.anthropic.com/claude-3.5-sonnet-model-card-addendum.pdf)) - *(June 2024)*
* **Claude 3:** "The Claude 3 Model Family: Opus, Sonnet, Haiku" ([Paper](https://www-cdn.anthropic.com/claude-3-model-card.pdf)) - *(March 2024)*

**Meta**

* **LLaMA 3:** "The Llama 3 Herd of Models" ([Paper](https://arxiv.org/abs/2407.21783)) ([GitHub](https://github.com/meta-llama/llama3)) - *(July 2024)*
* **LLaMA 2:** "Llama 2: Open Foundation and Fine-Tuned Chat Models" ([Paper](https://arxiv.org/abs/2307.09288)) - *(July 2023)*
* **LLaMA:** "LLaMA: Open and Efficient Foundation Language Models" ([Paper](https://arxiv.org/abs/2302.13971)) - *(February 2023)*

**DeepSeek**

* **DeepSeek-V3:** "DeepSeek-V3: Decoupling Scaling Law for Training and Inference" ([Paper](https://arxiv.org/abs/2412.19437)) ([GitHub](https://github.com/deepseek-ai/DeepSeek-LLM)) - *(December 2024)*
* **DeepSeek-V2:** "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" ([Paper](https://arxiv.org/abs/2405.04434)) - *(May 2024)*
* **DeepSeek LLM:** "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism" ([Paper](https://arxiv.org/abs/2401.02954)) - *(January 2024)*
* **DeepSeekMoE:** "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" ([Paper](https://arxiv.org/abs/2401.12246)) - *(January 2024)*

**Mistral**

* **Mixtral:** "Mixtral of Experts" ([Paper](https://arxiv.org/abs/2401.04088)) ([GitHub](https://github.com/mistralai/mistral-src)) - *(January 2024)*
* **Mistral 7B:** "Mistral 7B" ([Paper](https://arxiv.org/abs/2310.06825)) - *(October 2023)*

**Qwen (Alibaba)**

* **Qwen2.5:** "Qwen2.5: Advancing Large Language Models for Code, Math, Multilingualism, and Long Context" ([Paper](https://arxiv.org/abs/2412.15115)) ([GitHub](https://github.com/QwenLM/Qwen)) - *(December 2024)*
* **Qwen2-VL:** (Multimodal Version) ([HF Models](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)) - *(June 2024)*
* **Qwen 2:** "Qwen2: The new generation of Qwen large language models" ([Paper](https://arxiv.org/abs/2406.04808)) - *(June 2024)*
* **Qwen:** "Qwen Technical Report" ([Paper](https://arxiv.org/abs/2309.16609)) - *(September 2023)*

**Microsoft Phi**

* **Phi-4-Mini:** "Phi-4-Mini: A 2.7B Parameter Model Surpassing Mixtral 8x7B on Reasoning Benchmarks" ([Paper](https://arxiv.org/abs/2503.01743)) - *(March 2025)*
* **Phi-3:** "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone" ([Paper](https://arxiv.org/abs/2404.14219)) - *(April 2024)*
* **Phi-2:** "Phi-2: The surprising power of small language models" ([Blog Post](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)) - *(December 2023)*
* **Phi-1:** "Textbooks Are All You Need" ([Paper](https://arxiv.org/abs/2306.11644)) - *(June 2023)*

**Other Publicly Available Models (Including Foundational/Base Models)**

* **SmolLM2:** "SmolLM 2: Scaling Small Language Models through Sparse Activations" ([Paper](https://arxiv.org/abs/2502.02737)) - *(February 2025)*
* **OLMo 2:** "OLMo 2: A Truly Open 70B Model" ([Paper](https://arxiv.org/abs/2501.00656)) ([GitHub](https://github.com/allenai/OLMo)) - *(January 2025)*
* **YuLan-Mini:** "YuLan-Mini: A High-Performance and Efficient Open-Source Small Language Model" ([Paper](https://arxiv.org/abs/2412.17743)) - *(December 2024)*
* **Yi-Lightning:** "Yi-Lightning: An Efficient and Capable Family of Small Language Models" ([Paper](https://arxiv.org/abs/2412.01253)) ([GitHub](https://github.com/01-ai/Yi)) - *(December 2024)*
* **Yi:** "Yi: Open Foundation Models by 01.AI" ([Paper](https://arxiv.org/abs/2403.04652)) - *(March 2024)*
* **OLMo:** "OLMo: Accelerating the Science of Language Models" ([Paper](https://arxiv.org/abs/2402.00838)) - *(February 2024)*
* **Orion:** "Orion-14B: Scaling Cost-Effective Training and Inference for Large Language Models" ([Paper](https://arxiv.org/abs/2401.06066)) ([GitHub](https://github.com/OrionStarAI/Orion)) - *(January 2024)*
* **YAYI2:** "YAYI 2: Multilingual Open-Source Large Language Models" ([Paper](https://arxiv.org/abs/2312.14862)) ([GitHub](https://github.com/wenge-research/YAYI2)) - *(December 2023)*
* **Baichuan 2:** "Baichuan 2: Open Large-scale Language Models" ([Paper](https://arxiv.org/abs/2309.10305)) ([GitHub](https://github.com/baichuan-inc/Baichuan2)) - *(September 2023)*
* **CodeLLaMA:** "Code Llama: Open Foundation Models for Code" ([Paper](https://arxiv.org/abs/2308.12950)) - *(August 2023)*
* **StarCoder:** "StarCoder: may the source be with you!" ([Paper](https://arxiv.org/abs/2305.06161)) - *(May 2023)*
* **LLaVA:** "Visual Instruction Tuning" ([Paper](https://arxiv.org/abs/2304.08485)) - *(April 2023)*
* **BLOOM:** "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model" ([Paper](https://arxiv.org/abs/2211.05100)) ([Model Hub](https://huggingface.co/bigscience/bloom)) - *(November 2022)*
* **GPT-NeoX:** "GPT-NeoX-20B: An Open-Source Autoregressive Language Model" ([Paper](https://arxiv.org/abs/2204.06745)) ([GitHub](https://github.com/EleutherAI/gpt-neox)) - *(April 2022)*
* **GLM:** "GLM: General Language Model Pretraining with Autoregressive Blank Infilling" ([Paper](https://arxiv.org/abs/2103.10360)) - *(March 2021)*
* **T5:** "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" ([Paper](https://arxiv.org/abs/1910.10683)) - *(October 2019)*

**Specific Multimodal Fine-Tunes (Examples)**

* **LLaVA-o1:** "LLaVA-o1: Pushing the Limits of Open Multimodal Models with Process Supervision" ([Paper](https://arxiv.org/abs/2406.17439)) - *(June 2024)*
* **GLM-4V:** "GLM-4V: A General Multimodal Large Language Model with High Performance" ([Paper](https://arxiv.org/abs/2405.18754)) - *(May 2024)*
* **AtomThink:** "AtomThink: A Multimodal Reasoning Model with Atomic Thought Decomposition" ([Paper](https://arxiv.org/abs/2404.18718)) - *(April 2024)*
* **Math-LLaVA:** "Math-LLaVA: Bootstrapping Mathematical Reasoning for Large Vision Language Models" ([Paper](https://arxiv.org/abs/2404.02693)) - *(April 2024)*
* **M-STAR:** "M-STAR: Boosting Math Reasoning Ability of Multimodal Language Models" ([Paper](https://arxiv.org/abs/2404.10596)) - *(April 2024)*
* **MiniCPM-V:** "MiniCPM-V: A Vision Language Model for OCR and Reasoning" ([Paper](https://arxiv.org/abs/2311.10056)) - *(November 2023)*

**Other Closed-Source / Commercial Models**

* **Grok 3:** "Grok-3: The Next Generation of Grok" ([Blog Post](https://x.ai/news/grok-3)) - *(December 2024)*
* **Command R(+):** "Introducing Command R+: A Scalable Large Language Model Built for Business" ([Blog Post](https://txt.cohere.com/command-r-plus-scalable-llm-built-for-business/)) - *(April 2024)*
* **Grok 1:** "Grok-1: Release Announcement" ([Blog Post](https://x.ai/blog/grok-os)) - *(March 2024)*
* **InternLM:** "InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities" ([Paper](https://arxiv.org/abs/2309.01381)) - *(September 2023)*

## 6. 📊 Datasets & Benchmarks

> This section lists resources for training and evaluating mathematical LLMs.
> *Note: Comprehensive listing and categorization heavily informed by Table 3 in survey arXiv:2503.17726.*

### 6.1 Problem Solving Benchmarks

*Datasets primarily focused on evaluating mathematical problem-solving abilities (word problems, competition math, etc.).*

**Grade School Level (Mostly MWP - Math Word Problems)**

* **Dolphin18K Benchmark:** "How Well Do Large Language Models Perform on Basic Math Problems?" ([Paper](https://arxiv.org/abs/2303.07941)) - *(March 2023)*
* **SVAMP Benchmark:** "Are NLP Models Really Solving Simple Math Word Problems?" ([Paper](https://aclanthology.org/2021.emnlp-main.810/)) ([HF Dataset](https://huggingface.co/datasets/ought/raft_svamp)) - *(November 2021)*
* **GSM8K Benchmark:** "Training Verifiers to Solve Math Word Problems" ([Paper](https://arxiv.org/abs/2110.14168)) ([HF Dataset](https://huggingface.co/datasets/gsm8k)) - *(October 2021)*
* **ASDiv Benchmark:** "ASDiv: A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers" ([Paper](https://aclanthology.org/2021.naacl-main.168/)) ([HF Dataset](https://huggingface.co/datasets/EleutherAI/asdiv)) - *(June 2021)*
* **MultiArith Benchmark:** "Learning to Solve Arithmetic Word Problems with Operation-Based Knowledge" ([Paper](https://aclanthology.org/D17-1083/)) - *(September 2017)*
* **MAWPS Benchmark:** "MAWPS: A Math Word Problem Repository" ([Paper](https://aclanthology.org/N16-1136/)) ([GitHub](https://github.com/facebookresearch/mawps-dataset)) - *(June 2016)*
* **SingleOp Benchmark:** "Solving General Arithmetic Word Problems" ([Paper](https://aclanthology.org/D15-1101/)) - *(July 2015)*
* **AddSub Benchmark:** "Learning to Solve Arithmetic Word Problems with Verb Categorization" ([Paper](https://aclanthology.org/N14-1049/)) - *(June 2014)*

**Competition / High School / University Level**

* **MMLU-Pro Benchmark:** "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark" ([Paper](https://arxiv.org/abs/2406.01574)) ([HF Dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)) - *(June 2024)*
* **MathChat Benchmark:** "MathChat: Benchmarking Mathematical Reasoning and Instruction Following in Multi-Turn Interactions" ([Paper](https://arxiv.org/abs/2405.19444)) - *(May 2024)*
* **Mamo Benchmark:** "LLMs for Mathematical Modeling: Towards Bridging the Gap between Natural and Mathematical Languages" ([Paper](https://arxiv.org/abs/2404.11917)) - *(April 2024)*
* **MathUserEval Benchmark:** Introduced in "ChatGLM-Math: Improving Math Problem-Solving..." ([Paper](https://arxiv.org/abs/2404.06864)) ([GitHub](https://github.com/THUDM/ChatGLM-Math)) - *(April 2024)*
* **MR-MATH Benchmark:** "MR-MATH: A Multi-Resolution Mathematical Reasoning Benchmark for Large Language Models" ([Paper](https://arxiv.org/abs/2404.13834)) - *(April 2024)*
* **MATHTRAP Benchmark:** "MATHTRAP: A Large-Scale Dataset for Evaluating Mathematical Reasoning Ability of Foundation Models" ([Paper](https://arxiv.org/abs/2402.10611)) - *(February 2024)*
* **GPQA Benchmark:** "GPQA: A Graduate-Level Google-Proof Q&A Benchmark" ([Paper](https://arxiv.org/abs/2311.12022)) ([GitHub](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/gpqa)) - *(November 2023)*
* **MwpBench Benchmark:** Introduced in "MathScale: Scaling Instruction Tuning..." ([Paper](https://arxiv.org/abs/2310.17166)) - *(October 2023)*
* **SciBench Benchmark:** "SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models" ([Paper](https://arxiv.org/abs/2307.10635)) ([Website](https://scibench-ucla.github.io/)) - *(July 2023)*
* **AIME Benchmark (Example Analysis):** "Solving Math Word Problems with Process- and Outcome-Based Feedback" ([Paper](https://arxiv.org/abs/2305.15062)) - *(May 2023)* (Uses AIME subset)
* **AGIEval Benchmark:** "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models" ([Paper](https://arxiv.org/abs/2304.06364)) ([GitHub](https://github.com/microsoft/AGIEval)) - *(April 2023)*
* **MATH Benchmark:** "Measuring Mathematical Problem Solving With the MATH Dataset" ([Paper](https://arxiv.org/abs/2103.03874)) ([HF Dataset](https://huggingface.co/datasets/hendrycks/competition_math)) - *(March 2021)*
* **MMLU Benchmark (Math sections):** "Measuring Massive Multitask Language Understanding" ([Paper](https://arxiv.org/abs/2009.03300)) ([Dataset Info](https://paperswithcode.com/dataset/mmlu)) - *(September 2020)*

**Domain-Specific / Other**

* **MathEval Benchmark:** "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" ([Paper](https://arxiv.org/abs/2402.03300)) ([OpenReview](https://openreview.net/forum?id=DexGnh0EcB)) - *(February 2024)* (Introduced/used in DeepSeekMath)
* **GeomVerse Benchmark:** "GeomVerse: A Systematic Evaluation of Large Vision Language Models for Geometric Reasoning" ([Paper](https://arxiv.org/abs/2312.12241)) - *(December 2023)*
* **MR-GSM8K Benchmark:** "MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Model Evaluation" ([Paper](https://arxiv.org/abs/2312.17080)) ([GitHub](https://github.com/dvlab-research/MR-GSM8K)) - *(December 2023)*
* **MGSM Benchmark:** "MGSM: A Multi-lingual GSM8K Benchmark" ([Paper](https://arxiv.org/abs/2310.04952)) ([HF Dataset](https://huggingface.co/datasets/juletxara/mgsm)) - *(October 2023)*
* **ROBUSTMATH Benchmark:** "Evaluating the Robustness of Large Language Models on Math Problem Solving" ([Paper](https://arxiv.org/abs/2309.05690)) - *(September 2023)*
* **TABMWP Benchmark:** "Tabular Math Word Problems" ([Paper](https://arxiv.org/abs/2209.14610)) ([GitHub](https://github.com/lupantech/tabular-math-word-problems)) - *(September 2022)*
* **NumGLUE Benchmark:** "NumGLUE: A Suite of Fundamental yet Challenging Mathematical Reasoning Tasks" ([Paper](https://arxiv.org/abs/2109.06611)) - *(September 2021)*
* **MathQA Benchmark:** "MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms" ([Paper](https://aclanthology.org/N19-1245/)) ([Website](https://math-qa.github.io/)) - *(July 2019)*
* **Mathematics Benchmark:** "Analysing Mathematical Reasoning Abilities of Neural Models" ([Paper](https://arxiv.org/abs/1904.01557)) ([GitHub](https://github.com/deepmind/mathematics_dataset)) - *(April 2019)*

### 6.2 Theorem Proving Benchmarks

*Benchmarks focused on formal mathematical proof generation and verification.*

* **MathConstruct Benchmark:** "MathConstruct: Challenging LLM Reasoning with Constructive Proofs" ([Paper](https://arxiv.org/abs/2502.10197)) - *(February 2025)*
* **ProofNet Benchmark:** "ProofNet: A Benchmark for Autoformalizing and Formally Proving Undergraduate-Level Mathematics" ([Paper](https://arxiv.org/abs/2302.12433)) ([Website](https://proofnet.github.io/)) - *(February 2023)*
* **FOLIO Benchmark:** "FOLIO: Natural Language Reasoning with First-Order Logic" ([Paper](https://arxiv.org/abs/2209.00841)) ([Website](https://folio-benchmark.github.io/)) - *(September 2022)*
* **MiniF2F Benchmark:** "miniF2F: A Cross-System Benchmark for Formal Olympiad-Level Mathematics" ([Paper](https://arxiv.org/abs/2109.00110)) ([GitHub](https://github.com/openai/miniF2F)) - *(September 2021)*
* **NaturalProofs Benchmark:** "NaturalProofs: Mathematical Theorem Proving in Natural Language" ([Paper](https://arxiv.org/abs/2105.07101)) ([GitHub](https://github.com/wellecks/naturalproofs)) - *(May 2021)*
* **INT Benchmark:** "INT: An Inequality Benchmark for Evaluating Generalization in Theorem Proving" ([Paper](https://arxiv.org/abs/2007.02924)) ([GitHub](https://github.com/zarkook/INT)) - *(July 2020)*
* **CoqGym Benchmark:** "Learning to Prove Theorems via Interacting with Proof Assistants" ([Paper](https://arxiv.org/abs/1905.09381)) ([GitHub](https://github.com/princeton-vl/CoqGym)) - *(May 2019)*
* **HolStep Benchmark:** "HOLStep: A Machine Learning Dataset for Higher-Order Logic Theorem Proving" ([Paper](https://arxiv.org/abs/1703.00431)) - *(March 2017)*

### 6.3 Multimodal Benchmarks

*Benchmarks incorporating visual or other non-textual information. (Related to Sec 4)*

* **MM-MATH Benchmark:** Mentioned in "A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model" ([Paper](https://arxiv.org/abs/2412.11936)) - *(December 2024)*
* **DocReason25K Benchmark:** "DocReason: A Benchmark for Document Image Reasoning with Large Multimodal Models" ([Paper](https://arxiv.org/abs/2405.16898)) - *(May 2024)*
* **U-MATH Benchmark:** "U-MATH: A Comprehensive Benchmark for Evaluating Multimodal Math Problem Solving" ([Paper](https://arxiv.org/abs/2405.19028)) - *(May 2024)*
* **We-Math Benchmark:** "We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?" ([Paper](https://arxiv.org/abs/2405.17361)) - *(May 2024)*
* **M3CoT Benchmark:** Introduced in "Unified Abductive Cognition for Multimodal Reasoning" ([Paper](https://arxiv.org/abs/2405.17550)) - *(May 2024)*
* **CMM-Math Benchmark:** "CMM-Math: A Comprehensive Chinese Multimodal Math Benchmark" ([Paper](https://arxiv.org/abs/2405.16783)) - *(May 2024)*
* **MathVerse Benchmark:** "MathVerse: Does Your Multi-modal LLM Truly Understand Math?" ([Paper](https://arxiv.org/abs/2404.13834)) ([Website](https://mathverse.github.io/)) - *(April 2024)*
* **MR-MATH Benchmark:** "MR-MATH: A Multi-Resolution Mathematical Reasoning Benchmark for Large Language Models" ([Paper](https://arxiv.org/abs/2404.13834)) - *(April 2024)*
* **ErrorRadar Benchmark:** "ErrorRadar: Evaluating the Multimodal Error Detection of LLMs in Educational Settings" ([Paper](https://arxiv.org/abs/2403.03894)) - *(March 2024)*
* **MATH-Vision Benchmark:** "MATH-Vision: Evaluating Mathematical Reasoning of Large Vision Language Models" ([Paper](https://arxiv.org/abs/2402.17177)) - *(February 2024)*
* **GeoEval Benchmark:** Introduced in "GeomVerse: A Systematic Evaluation of Large Vision Language Models for Geometric Reasoning" ([Paper](https://arxiv.org/abs/2312.12241)) - *(December 2023)*
* **MMMU Benchmark (Math subset):** "MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI" ([Paper](https://arxiv.org/abs/2311.16502)) ([Website](https://mmmu-benchmark.github.io/)) - *(November 2023)*
* **MathVista Benchmark:** "MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts" ([Paper](https://arxiv.org/abs/2310.02255)) ([Website](https://mathvista.github.io/)) - *(October 2023)*
* **ScienceQA Benchmark:** "Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering" ([Paper](https://arxiv.org/abs/2209.09958)) ([Website](https://scienceqa.github.io/)) - *(September 2022)*
* **ChartQA Benchmark:** "ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning" ([Paper](https://arxiv.org/abs/2203.10244)) ([Website](https://charts.allenai.org/)) - *(March 2022)*
* **GeoQA Benchmark:** "GeoQA: A Geometric Question Answering Benchmark Towards Multimodal Numerical Reasoning" ([Paper](https://arxiv.org/abs/2009.05014)) - *(September 2020)*
* **FigureQA Benchmark:** "FigureQA: An Annotated Figure Dataset for Visual Reasoning" ([Paper](https://arxiv.org/abs/1710.07300)) ([Website](http://vision.cs.ucla.edu/figureqa/)) - *(October 2017)*

### 6.4 Training Datasets

*Datasets primarily used for pre-training or fine-tuning models on mathematical tasks.*

* **LeanNavigator generated data:** "Generating Millions Of Lean Theorems With Proofs By Exploring State Transition Graphs" ([Paper](https://arxiv.org/abs/2503.04772)) - *(March 2025)*
* **OpenMathMix Dataset (QaDS):** "Exploring the Mystery of Influential Data for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2405.01201)) - *(May 2024)*
* **Skywork-MathQA Dataset:** "Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models - The Story Goes On" ([Paper](https://arxiv.org/abs/2405.10814)) - *(May 2024)*
* **MathChatSync Dataset:** Introduced in "MathChat: Benchmarking Mathematical Reasoning..." ([Paper](https://arxiv.org/abs/2405.19444)) - *(May 2024)*
* **AutoMathText Dataset (AutoDS):** "Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts" ([Paper](https://arxiv.org/abs/2404.05960)) ([Code](https://github.com/yifanzhang-pro/AutoMathText)) ([HF Dataset](https://huggingface.co/datasets/math-ai/AutoMathText)) - *(April 2024)*
* **MathCode-Pile Dataset:** "MathCoder2: Better Math Reasoning from Continued Pretraining on Model-translated Mathematical Code" ([Paper](https://arxiv.org/abs/2404.11081)) ([Code](https://github.com/mathllm/MathCoder2)) - *(April 2024)*
* **MathV360K Dataset:** "Math-LLaVA: Bootstrapping Mathematical Reasoning for Large Vision Language Models" ([Paper](https://arxiv.org/abs/2404.02693)) ([HF Dataset](https://huggingface.co/datasets/Zhiqiang007/MathV360K)) - *(April 2024)*
* **MAmmoTH2 Data Strategy:** "MAmmoTH2: Scaling Instructions from the Web" ([Paper](https://arxiv.org/abs/2403.05274)) - *(March 2024)*
* **OpenMathInstruct-1 Dataset:** "OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset" ([Paper](https://arxiv.org/abs/2402.10176)) - *(February 2024)*
* **OpenWebMath Corpus:** "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" ([Paper](https://arxiv.org/abs/2402.03300)) ([GitHub](https://github.com/deepseek-ai/DeepSeek-Math)) - *(February 2024)*
* **MathVL Dataset:** Introduced in "MathGLM-Vision: Solving Mathematical Problems..." ([Paper](https://arxiv.org/abs/2402.04503)) - *(February 2024)*
* **MathInstruct Dataset:** "MAmmoTH: Building Math Generalist Models through Hybrid Instruction Tuning" ([Paper](https://arxiv.org/abs/2401.11445)) ([HF Dataset](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)) - *(January 2024)*
* **OpenMathInstruct-2 Dataset:** "Accelerating AI for Math with Massive Open-Source Instruction Data" ([Paper](https://arxiv.org/abs/2410.01560)) - *(October 2023)*
* **Proof-Pile / Proof-Pile 2 Corpora:** "Llemma: An Open Language Model For Mathematics" ([Paper](https://arxiv.org/abs/2310.10631)) - *(October 2023)*
* **MathScaleQA Dataset:** "MathScale: Scaling Instruction Tuning for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2310.17166)) - *(October 2023)*
* **SciInstruct Dataset:** "SciInstruct: a Self-Reflective Instruction Annotated Dataset for Training Scientific Language Models" ([Paper](https://arxiv.org/abs/2309.06631)) ([Code](https://github.com/THUDM/SciGLM)) - *(September 2023)*
* **MATH-Instruct Dataset:** "MATH-Instruct: A Large-Scale Mathematics Instruction-Tuning Dataset" ([Paper](https://arxiv.org/abs/2309.05653)) - *(September 2023)*

### 6.5 Augmented / Synthetic Datasets

*Datasets often generated synthetically or via augmentation techniques, used for specific training goals (e.g., verifiers, tool use, reasoning steps). (Supports techniques in Sec 3.3, 3.4)*

* **DART-Math Datasets (DART):** "DART-Math: Difficulty-Aware Rejection Tuning for Mathematical Problem-Solving" ([Paper](https://arxiv.org/abs/2405.14194)) - *(May 2024)*
* **PEN Dataset:** "PEN: Step-by-Step Training with Planning-Enhanced Explanations for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2405.17361)) - *(May 2024)*
* **KPMath / KPMath-Plus Dataset (KPDDS):** "Key-Point-Driven Data Synthesis with its Enhancement on Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2402.14836)) - *(February 2024)*
* **MMIQC Dataset (IQC):** "Augmenting Math Word Problems via Iterative Question Composing" ([Paper](https://arxiv.org/abs/2402.07576)) - *(February 2024)*
* **MetaMathQA Dataset:** "MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models" ([Paper](https://arxiv.org/abs/2309.12284)) ([HF Dataset](https://huggingface.co/datasets/meta-math/MetaMathQA)) - *(September 2023)*
* **PRM800K Dataset:** "Solving Math Word Problems with Process- and Outcome-Based Feedback" ([Paper](https://arxiv.org/abs/2305.15062)) - *(May 2023)*
* **Math50k Dataset:** "Teaching Small Language Models to Reason" ([Paper](https://arxiv.org/abs/2212.08410)) - *(December 2022)*
* **MathQA-Python Dataset:** "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks" ([Paper](https://arxiv.org/abs/2211.12588)) - *(November 2022)*
* **miniF2F+informal Dataset:** "Draft, sketch, and prove: Guiding formal theorem provers with informal proofs" ([Paper](https://arxiv.org/abs/2210.12283)) - *(October 2022)*
* **Lila Dataset:** "Lila: A Unified Benchmark for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2210.17517)) ([Website](https://allenai.org/lila/)) - *(October 2022)*
* **Aggregate Dataset (for Minerva):** "Solving Quantitative Reasoning Problems With Language Models" ([Paper](https://arxiv.org/abs/2206.14858)) - *(June 2022)*
* **NaturalProofs-Gen Dataset:** "NaturalProofs: Mathematical Theorem Proving in Natural Language" ([Paper](https://arxiv.org/abs/2105.07101)) ([GitHub](https://github.com/wellecks/naturalproofs)) - *(May 2021)*

## 7. 🛠️ Tools & Libraries

*Software tools, frameworks, and libraries relevant for working with LLMs in mathematics.*

* **Data Processing (IBM):** "IBM Data Prep Kit" ([GitHub](https://github.com/IBM/data-prep-kit)) - *(November 2023)*
* **Data Processing (Datatrove):** "Datatrove" ([GitHub](https://github.com/huggingface/datatrove)) - *(October 2023)*
* **Framework (LPML):** "LPML: LLM-Prompting Markup Language for Mathematical Reasoning" ([Paper](https://arxiv.org/abs/2309.04269)) - *(September 2023)*
* **Framework (LMDeploy):** "LMDeploy" ([GitHub](https://github.com/InternLM/lmdeploy)) - *(July 2023)*
* **Evaluation (OpenCompass):** "OpenCompass" ([GitHub](https://github.com/open-compass/opencompass)) - *(May 2023)*
* **Framework (Guidance):** "Guidance" ([GitHub](https://github.com/microsoft/guidance)) - *(February 2023)*
* **Framework (LangChain):** "LangChain" ([Website](https://www.langchain.com/)) - *(October 2022)*
* **Fine-tuning (LoRA):** "LoRA: Low-Rank Adaptation of Large Language Models" ([Paper](https://arxiv.org/abs/2106.09685)) - *(June 2021)*
* **ITP (Lean):** "Lean Theorem Prover" ([Website](https://lean-lang.org/))
* **ITP (Isabelle):** "Isabelle" ([Website](https://isabelle.in.tum.de/))
* **ITP (Coq):** "The Coq Proof Assistant" ([Website](https://coq.inria.fr/))

## 8. 🤝 Contributing

We are looking for contributors to help build this resource. Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

## 9. 📄 Citation

If you find this repository useful, please consider citing:

```bibtex
@misc{awesome-math-llm,
  author = {doublelei and Contributors},
  title = {Awesome-Math-LLM: A Curated List of Large Language Models for Mathematics},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{[https://github.com/doublelei/Awesome-Math-LLM](https://github.com/doublelei/Awesome-Math-LLM)}}
}
```

## 10. ⚖️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.