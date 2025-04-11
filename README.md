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

- [1. Surveys & Overviews](#1-surveys--overviews)
- [2. Core Reasoning Techniques](#2-core-reasoning-techniques)
  - [2.1 Chain-of-Thought & Prompting Strategies](#21-chain-of-thought--prompting-strategies)
  - [2.2 Search & Planning](#22-search--planning)
  - [2.3 Reinforcement Learning & Reward Modeling](#23-reinforcement-learning--reward-modeling)
  - [2.4 Self-Improvement & Self-Training](#24-self-improvement--self-training)
  - [2.5 Tool Use & Augmentation](#25-tool-use--augmentation)
  - [2.6 Neurosymbolic Methods](#26-neurosymbolic-methods)
- [3. Mathematical Domains & Tasks](#3-mathematical-domains--tasks)
  - [3.1 Arithmetic & Word Problems](#31-arithmetic--word-problems)
  - [3.2 Algebra, Geometry, Calculus, etc.](#32-algebra-geometry-calculus-etc)
  - [3.3 Competition Math](#33-competition-math)
  - [3.4 Formal Theorem Proving](#34-formal-theorem-proving)
  - [3.5 Symbolic Manipulation](#35-symbolic-manipulation)
- [4. Multimodal Mathematical Reasoning](#4-multimodal-mathematical-reasoning)
- [5. Models](#5-models)
  - [5.1 Math-Specialized LLMs](#51-math-specialized-llms)
  - [5.2 Reasoning-Focused LLMs](#52-reasoning-focused-llms)
  - [5.3 Leading General LLMs](#53-leading-general-llms)
- [6. Datasets & Benchmarks](#6-datasets--benchmarks)
  - [6.1 Problem Solving Benchmarks](#61-problem-solving-benchmarks)
  - [6.2 Theorem Proving Benchmarks](#62-theorem-proving-benchmarks)
  - [6.3 Multimodal Benchmarks](#63-multimodal-benchmarks)
  - [6.4 Training Datasets](#64-training-datasets)
- [7. Tools & Libraries](#7-tools--libraries)
- [8. Challenges & Future Directions](#8-challenges--future-directions)
- [9. Contributing](#9-contributing)
- [10. Citation](#10-citation)
- [11. License](#11-license)

### Recent Highlights

[2025-03]

- **ERNIE**: "Baidu Unveils ERNIE 4.5 and Reasoning Model ERNIE X1" [[website](https://yiyan.baidu.com/)]
- **QaQ**: "QwQ-32B: Embracing the Power of Reinforcement Learning" [[blog](https://qwenlm.github.io/blog/qwq-32b/)] [[repo](https://github.com/QwenLM/QwQ)]


## 1. Surveys & Overviews

*Meta-analyses and survey papers about LLMs for mathematics*

- ([https://arxiv.org/abs/2503.17726](https://arxiv.org/abs/2503.17726)) - Forootani, A. (arXiv:2503.17726). Covers evolution, methodologies (CoT, Tools, RL), models, datasets, challenges. [1, 6]
- ([https://arxiv.org/abs/2502.14333](https://arxiv.org/abs/2502.14333)) - Wei, T.-R., et al. (arXiv:2502.14333). Focuses on feedback mechanisms (step/outcome, training/training-free). [7, 8, 9]
- ([https://arxiv.org/abs/2502.17419](https://arxiv.org/abs/2502.17419)) - Zeng, Z., et al. (arXiv:2502.17419). Discusses System 2 reasoning, MCTS, reward modeling, self-improvement, RL. [3, 10, 11]
- MLLM Survey: "A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model" ([https://arxiv.org/abs/2412.11936](https://arxiv.org/abs/2412.11936)) - Yan, Y., et al. (arXiv:2412.11936). First survey on multimodal math reasoning, covering benchmarks, methods, challenges. [2]
- DL4Math: "A Survey of Deep Learning for Mathematical Reasoning" [2022-12] [[paper](https://arxiv.org/abs/2212.09206)]
- LLM4Math: "Large Language Models for Mathematical Reasoning: Progresses and Challenges" [2024-02] [[paper](https://arxiv.org/abs/2402.15694)]
- ([https://arxiv.org/abs/2402.06196](https://arxiv.org/abs/2402.06196)) - Minaee, S., et al. (arXiv:2402.06196). General overview of LLM families, training, datasets, evaluation. [12]
- ([https://github.com/atfortes/Awesome-LLM-Reasoning](https://github.com/atfortes/Awesome-LLM-Reasoning)) - Curated list focusing on general LLM reasoning techniques. [13]
- ([https://github.com/zzli2022/Awesome-System2-Reasoning-LLM](https://github.com/zzli2022/Awesome-System2-Reasoning-LLM)) - Curated list focusing on System 2 reasoning (RL, MCTS, Self-Improve). [14]
- ([https://github.com/InfiMM/Awesome-Multimodal-LLM-for-Math-STEM](https://github.com/InfiMM/Awesome-Multimodal-LLM-for-Math-STEM)) - Curated list for MLLMs in Math/STEM. [15]

## 2. Core Reasoning Techniques

*Core approaches and methodologies for enhancing mathematical reasoning in Large Language Models*

### 2.1 Chain-of-Thought & Prompting Strategies

*Fundamental techniques involving step-by-step reasoning generation.*

- **Chain-of-Thought**: "Chain of Thought Prompting Elicits Reasoning in Large Language Models" ([https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)) - Wei, J., et al. (NeurIPS 2022). Foundational CoT paper.
- **Self-Consistency**: "Self-Consistency Improves Chain of Thought Reasoning in Language Models" ([https://arxiv.org/abs/2203.11171](https://arxiv.org/abs/2203.11171)) - Wang, X., et al. (ICLR 2023). Improves CoT by sampling multiple reasoning paths.
- **Algorithmic Prompting**: Introduces Algorithmic Prompting ([https://research.google/blog/teaching-language-models-to-reason-algorithmically/](https://research.google/blog/teaching-language-models-to-reason-algorithmically/)). [16, 17]
- **Faithful CoT**: "Faithful Chain-of-Thought Reasoning" ([https://arxiv.org/abs/2301.13379](https://arxiv.org/abs/2301.13379)) - Lyu, Q., et al. (IJCNLP-AACL 2023). Focuses on improving the faithfulness of CoT. [18]
- *Concept*: Prompting LLMs to output intermediate reasoning steps before the final answer. Variants include self-checking, reflection, planning, Long CoT, and Algorithmic Prompting. [6, 7, 16]

### 2.2 Search & Planning

*Techniques exploring multiple potential solution paths.*

- **Tree of Thoughts (ToT)**: "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" ([https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)) - Yao, S., et al. (NeurIPS 2023). Explores different reasoning branches and self-evaluates progress. [13, 6]
  - Code:([https://github.com/kyegomez/Tree-of-Thoughts-LLM](https://github.com/kyegomez/Tree-of-Thoughts-LLM))
- **Graph-of-Thoughts (GoT)**: Represents reasoning as a graph ([https://arxiv.org/abs/2308.09687](https://arxiv.org/abs/2308.09687)) - Besta, M., et al. (arXiv 2023). [20, 21]
- **Monte Carlo Tree Search (MCTS)**: Simulation-based search for navigating solution space. [14, 22, 9, 11]
  - Paper:([https://arxiv.org/abs/2502.10000](https://arxiv.org/abs/2502.10000)) - Qi, Z., et al. (arXiv 2025). Uses MCTS in self-improvement. [22, 23]
- **Best-First Search (BFS)**: Used in theorem proving and problem solving. [24]
  - Paper:([https://arxiv.org/abs/2502.03438](https://arxiv.org/abs/2502.03438)) - Xin, R., et al. (arXiv 2025). [24, 25]
- **Search-Based Methods**: "Enhancing LLM Reasoning with Reward-guided Tree Search" [2023-10] [[paper](https://arxiv.org/abs/2310.09177)]

### 2.3 Reinforcement Learning & Reward Modeling

*Optimizing LLMs using feedback signals (rewards) through RL algorithms.*

- **RLHF**: "Training language models to follow instructions with human feedback" [2022-05] [[paper](https://arxiv.org/abs/2203.02155)]
- **DPO**: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" [2023-05] [[paper](https://arxiv.org/abs/2305.18290)]
- **Step-DPO**: "Step-DPO: Step-wise Preference Optimization for Long-chain Reasoning of LLMs" [2023-11] [[paper](https://arxiv.org/abs/2310.06671)]
- **GRPO**: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" [2024-02] [[paper](https://arxiv.org/abs/2402.03300)] Introduces Group Relative Policy Optimization. [14, 6, 11, 24, 28]
- **Process Supervision**: "AlphaMath Almost Zero: Process Supervision without Process" [2023-12] [[paper](https://arxiv.org/abs/2312.00481)]
- **PPO**: [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) - Schulman, J., et al. (arXiv 2017). [24]
- **Reward Models**:
  - **Outcome Reward Models (ORM)**: Evaluate the final answer. [14, 10, 6, 7, 9, 11, 26, 27]
  - **Process Reward Models (PRM)**: Evaluate intermediate reasoning steps. [14, 10, 6, 7, 9, 11, 26, 27]
- **Automated Process Supervision**: "Let Models Explain Themselves: Step-by-Step Verification for Large Language Models" ([https://arxiv.org/abs/2312.08935](https://arxiv.org/abs/2312.08935)) - Wang, P., et al. (arXiv 2023). Uses automated process supervision. [9, 27]
- **Automated PRM Data Generation**: "Learning Process-Reward Models for Math Reasoning from Minimal Human Feedback" (https://arxiv.org/abs/2406.13559) - Luo, L., et al. (arXiv 2024). Automated PRM data generation using MCTS. [9, 29]
- **Verification Methods**: "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations" [2023-11] [[paper](https://arxiv.org/abs/2308.07615)]
- *Concept*: Optimizing LLMs using feedback signals (rewards) through RL algorithms like PPO, DPO, GRPO. [14, 6, 7, 9, 11, 26, 27]

### 2.4 Self-Improvement & Self-Training

*LLMs generate their own training data, evaluate attempts, and learn from successes.*

- **STaR**: "STaR: Bootstrapping Reasoning With Reasoning" ([https://arxiv.org/abs/2203.14465](https://arxiv.org/abs/2203.14465)) - Zelikman, E., et al. (arXiv 2022). Foundational self-taught reasoner. [14, 15, 12]
- **rStar-Math**: "Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking" ([https://arxiv.org/abs/2502.10000](https://arxiv.org/abs/2502.10000)) - Qi, Z., et al. (arXiv 2025). Combines MCTS, RL, and self-evolution. [22, 15, 23] Also see [rStar-Math Project](https://github.com/agi-templar/rStar-Math-v2)
- **Quiet-STaR**: "Token-level Self-evolution for Large Language Models" ([https://arxiv.org/abs/2403.09629](https://arxiv.org/abs/2403.09629)) - Zelikman, E., et al. (arXiv 2024). Token-level exploration during training. [15]
- **ReST**: "Reinforced Self-Training (ReST) for Language Modeling" ([https://arxiv.org/abs/2308.08998](https://arxiv.org/abs/2308.08998)) - Gulcehre, C., et al. (arXiv 2023). [15]
- **Self-Refine**: "Self-Refine: Iterative Refinement with Self-Feedback" ([https://arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651)) - Madaan, A., et al. (NeurIPS 2023). Test-time refinement. [15, 30]
- *Concept*: LLMs generate their own training data through exploration (e.g., MCTS), evaluate attempts, and learn from successes. [14, 10, 22, 11, 27]

### 2.5 Tool Use & Augmentation

*Enabling LLMs to call external tools (calculators, code interpreters, solvers, search engines).*

- **PAL**: "Program-Aided Language Models" ([https://arxiv.org/abs/2211.10435](https://arxiv.org/abs/2211.10435)) - Gao, L., et al. (ICML 2023). Generates code executed by an interpreter. [6, 31, 32]
  - Code: [reasoning-machines/pal](https://github.com/reasoning-machines/pal) [4]
- **ART**: "ART: Automatic multi-step reasoning and tool-use for large language models" ([https://arxiv.org/abs/2303.09014](https://arxiv.org/abs/2303.09014)) - Paranjape, B., et al. (arXiv 2023). Dynamically selects and uses tools. [13, 6, 33, 34, 25, 35]
  - Code (Guidance library used in ART): [microsoft/guidance](https://github.com/microsoft/guidance) [5, 16]
- **ToRA**: "ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving" [2023-11] [[paper](https://arxiv.org/abs/2309.17452)]
- **Program of Thoughts**: "Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks" [2022-11] [[paper](https://arxiv.org/abs/2211.12588)]
- **Toolformer**: "Toolformer: Language Models Can Teach Themselves to Use Tools" ([https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)) - Schick, T., et al. (NeurIPS 2023). LLM learns to use APIs. [24]
- **Logic-LM**: "Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning" ([https://arxiv.org/abs/2305.12295](https://arxiv.org/abs/2305.12295)) - Pan, L., et al. (EMNLP 2023 Findings). Integrates logical solvers. [13, 18]
- **SatLM**: "SAT-LM: Satisfiability-Aided Language Models for Logic Puzzle Solving" ([https://arxiv.org/abs/2310.05726](https://arxiv.org/abs/2310.05726)) - Ye, X., et al. (NeurIPS 2023). Integrates SAT solvers. [13, 18]
- **Code Interpreters**: "Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-Based Self-Verification" [2023-08] [[paper](https://arxiv.org/abs/2308.07921)]

### 2.6 Neurosymbolic Methods

*Integrating neural models (LLMs) with symbolic reasoning methods.*

- **VSA Integration**: "Improving Rule-based Reasoning in LLMs via Neurosymbolic Representations" ([https://arxiv.org/abs/2502.01657](https://arxiv.org/abs/2502.01657)) - Dhanraj, V., & Eliasmith, C. (arXiv 2025). Uses Vector Symbolic Algebras (VSAs) to encode hidden states.
- **Constrained Decoding (CRANE)**: "CRANE: Reasoning with constrained LLM generation" ([https://arxiv.org/abs/2502.09061](https://arxiv.org/abs/2502.09061)) - Suresh, A., et al. (arXiv 2025). Balances constrained decoding with reasoning flexibility. [38, 39]
- **Symbolic Operator Prediction**: "From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs" ([https://arxiv.org/abs/2503.09986](https://arxiv.org/abs/2503.09986)) - Jin, C., et al. (arXiv 2025). LLM predicts operators to guide symbolic regression. [40, 41]
- **Symbolic Mixture-of-Experts**: "Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning" ([https://arxiv.org/abs/2503.05641](https://arxiv.org/abs/2503.05641)) - Chen, Z., et al. (arXiv 2025). Routes problems to expert LLMs based on symbolic skill representations. [30, 42]
- *Concept*: Integrating neural models (LLMs) with symbolic reasoning methods for enhanced reliability, interpretability, and rule-following. [36, 37, 23]

## 3. Mathematical Domains & Tasks

*Different mathematical domains and applications where LLMs are applied*

### 3.1 Arithmetic & Word Problems

*Solving grade-school level math word problems involving multi-step arithmetic.*

- **GSM8K Solving**: "Training Verifiers to Solve Math Word Problems" ([https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)) - Cobbe, K., et al. (arXiv 2021). Introduces GSM8K benchmark. [21, 44, 46, 47]
- **Scratchpad Reasoning**: "Show Your Work: Scratchpads for Intermediate Computation with Language Models" [2021-12] [[paper](https://arxiv.org/abs/2112.00114)]
- **Pseudo-Dual Learning**: "Pseudo-Dual Learning: Solving Math Word Problems with Reexamination" [2023-10] [[paper](https://arxiv.org/abs/2310.04292)]
- **Symbolic Reasoning**: "Reasoning in Large Language Models Through Symbolic Math Word Problems" [2023-05] [[paper](https://arxiv.org/abs/2305.20050)]
- **Error Detection (MR-GSM8K)**: "MR-GSM8K: A Meta-Reasoning Benchmark for Large Language Models" ([https://arxiv.org/abs/2312.17080](https://arxiv.org/abs/2312.17080)) - Liu, Z., et al. (arXiv 2023). Introduces MR-GSM8K for evaluating error detection. [45, 48, 46]

### 3.2 Algebra, Geometry, Calculus, etc.

*Problems spanning standard high school and undergraduate curricula.*

- **AlphaGeometry**: "Solving Olympiad Geometry without Human Demonstrations" [2024-01] [[paper](https://www.nature.com/articles/s41586-023-06747-5)]
- **UniGeo**: "Unifying Geometry Logical Reasoning via Reformulating Mathematical Expression" [2022-10] [[paper](https://arxiv.org/abs/2210.01196)]
- **Inter-GPS**: "Inter-GPS: Interpretable Geometry Problem Solving with Formal Language and Symbolic Reasoning" [2021-06] [[paper](https://arxiv.org/abs/2105.04166)]
- **MATH Benchmark**: "Measuring Mathematical Problem Solving With the MATH Dataset" ([https://arxiv.org/abs/2103.03874](https://arxiv.org/abs/2103.03874)) - Hendrycks, D., et al. (arXiv 2021). Introduces the MATH benchmark covering these areas.
- **SciBench**: "SciBench: Evaluating College-Level Scientific Problem Solving Abilities of Large Language Models" ([https://arxiv.org/abs/2307.10635](https://arxiv.org/abs/2307.10635)) - Wang, X., et al. (NeurIPS 2023 Datasets and Benchmarks). Introduces SciBench including calculus etc. [52, 54, 55]

### 3.3 Competition Math

*Challenging problems from competitions like AMC, AIME, IMO.*

- **Olympiad Geometry (AlphaGeometry)**: "Solving Olympiad Geometry without Human Demonstrations" [2024-01] [[paper](https://www.nature.com/articles/s41586-023-06747-5)]
- **USAMO Evaluation**: "Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad" ([https://arxiv.org/abs/2503.21934](https://arxiv.org/abs/2503.21934)) - Liu, Z., et al. (arXiv 2025). Evaluates SOTA models on recent USAMO problems. [56, 57]
- **Olympiad Human Evaluation**: "Brains vs. Bytes: Evaluating LLM Proficiency in Olympiad Mathematics" ([https://arxiv.org/abs/2504.01995](https://arxiv.org/abs/2504.01995)) - Mahdavi, H., et al. (arXiv 2025). Human evaluation of LLM proofs for Olympiad problems.
- **OlympiadBench**: "OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems" [2023-11] [[paper](https://arxiv.org/abs/2311.07575)]
- *Key Benchmarks*: MATH [6, 38, 49, 50, 28], AIME subsets [38, 52], OlympiadBench [2, 38]

### 3.4 Formal Theorem Proving

*Generating formal mathematical proofs verifiable by Interactive Theorem Provers (ITPs).*

- **GPT-f**: "GPT-f: Generative Language Modeling for Automated Theorem Proving" [2021-05] [[paper](https://arxiv.org/abs/2009.03393)]
- **miniF2F**: "miniF2F: A Cross-System Benchmark for Formal Olympiad-Level Mathematics" [2022-09] [[paper](https://arxiv.org/abs/2109.00110)] Introduces MiniF2F benchmark. [7, 21, 59, 36]
- **COPRA**: "COPRA: COllaborative PRoof Assistant with GPT-4" [2024-02] [[paper](https://arxiv.org/abs/2402.10108)]
- **Formal Mathematics Survey**: "Formal Mathematical Reasoning: A New Frontier in AI" [2023-11] [[paper](https://arxiv.org/abs/2306.03544)]
- **Llemma**: "Llemma: An Open Language Model For Mathematics" [2023-10] [[paper](https://arxiv.org/abs/2310.10631)]
- **Lean Data Synthesis**: "LeanNavigator: Generating Diverse Theorems and Proofs by Exploring State Graphs" ([https://arxiv.org/abs/2503.04772](https://arxiv.org/abs/2503.04772)) - Jiang, A., et al. (arXiv 2025). Automated formal proof data synthesis. [8, 44, 46, 30]
- **BFS-Prover**: "BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving" ([https://arxiv.org/abs/2502.03438](https://arxiv.org/abs/2502.03438)) - Xin, R., et al. (arXiv 2025). Uses BFS and expert iteration. [24, 25]
- *Key Tool*: Lean Theorem Prover [60, 19, 61, 62, 63, 64]

### 3.5 Symbolic Manipulation

*Using LLMs for tasks involving symbolic expressions, potentially integrating with symbolic solvers.*

- See [Section 2.6 Neurosymbolic Methods](#26-neurosymbolic-methods) for integrated approaches.
- See [Section 2.5 Tool Use & Augmentation](#25-tool-use--augmentation) for interaction with solvers like SymPy/Mathematica via tools (Logic-LM, SatLM).

## 4. Multimodal Mathematical Reasoning

*Solving math problems involving non-textual information (diagrams, plots, tables, handwritten equations).*

- **Survey**: "A Survey of Mathematical Reasoning in the Era of Multimodal Large Language Model" ([https://arxiv.org/abs/2412.11936](https://arxiv.org/abs/2412.11936)) - Yan, Y., et al. (arXiv 2024). Comprehensive survey. [2, 37, 66]
- **MathVista Benchmark**: "MathVista: Evaluating Mathematical Reasoning of Foundation Models in Visual Contexts" ([https://arxiv.org/abs/2310.02255](https://arxiv.org/abs/2310.02255)) - Lu, P., et al. (NeurIPS 2023). Introduces MathVista benchmark.
- **ScienceQA Benchmark**: "Learn to Explain: Multimodal Reasoning, Representation, and Debiasing in Science Question Answering" ([https://arxiv.org/abs/2209.09958](https://arxiv.org/abs/2209.09958)) - Lu, P., et al. (NeurIPS 2022). Introduces ScienceQA benchmark. [58, 56, 67, 54]
- **MATH-Vision Benchmark**: "Measuring Multimodal Mathematical Reasoning with the MATH-Vision Dataset" ([https://arxiv.org/abs/2402.17177](https://arxiv.org/abs/2402.17177)) - Wang, X., et al. (NeurIPS 2024). Introduces MATH-Vision benchmark.
- **ErrorRadar**: "ErrorRadar: Evaluating the Multimodal Error Detection of LLMs in Educational Settings" [2024-03] [[paper](https://arxiv.org/abs/2403.03894)]
- **MathVerse**: "MathVerse: Assessing Visual Mathematical Understanding in Multimodal LLMs" [2024-04] [[paper](https://arxiv.org/abs/2404.13834)]
- **MathV360K Dataset**: "Math-LLaVA: Bootstrapping Mathematical Reasoning for Large Vision Language Models" ([https://arxiv.org/abs/2404.02693](https://arxiv.org/abs/2404.02693)) [2024-04]. Large multimodal math QA dataset.
- **Math-PUMA**: "Math-PUMA: Progressive Upward Multimodal Alignment for Math Reasoning Enhancement" [2024-04] [[paper](https://arxiv.org/abs/2404.01166)]
- **MAVIS-Instruct**: "MAVIS: Multimodal Automatic Visual Instruction Synthesis for Math Problem Solving" [2024-04] [[paper](https://arxiv.org/abs/2404.04473)]
- **Models (Examples)**: GPT-4V [6, 68], Gemini Vision [2, 6, 68], Qwen-VL [65, 6], LLaVA variants [2, 6, 69], AtomThink [6, 11].
- **Datasets & Benchmarks**:
  - [MathVista](https://mathvista.github.io/)
  - [ScienceQA](https://scienceqa.github.io/) [13, 15, 58, 56, 37, 62, 67, 53, 54]
  - [MATH-Vision](https://arxiv.org/abs/2402.17177)
  - [GeoQA/GeoEval](https://arxiv.org/abs/2412.11936) [2, 37]
  - [MathV360K](https://huggingface.co/datasets/Zhiqiang007/MathV360K) (Training Data)
  - See also: [Awesome-Multimodal-LLM-for-Math-STEM Datasets](https://github.com/InfiMM/Awesome-Multimodal-LLM-for-Math-STEM#mllm-mathstem-dataset) [13]

## 5. Models

*Prominent LLMs relevant to mathematics.*

### 5.1 Math-Specialized LLMs

*Models specifically pre-trained or fine-tuned for mathematical tasks.*

- **DeepSeekMath**: ([https://github.com/deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)) - (DeepSeek) Pre-trained on math-heavy web data. Base, Instruct, RL versions. [14, 65, 51, 54]
- **Qwen-Math / Qwen2.5-Math**: ([https://arxiv.org/abs/2406.13559](https://arxiv.org/abs/2406.13559)) - (Alibaba) Math-focused versions, strong performance. [3, 27, 30, 42, 70]
- **InternLM-Math**: ([https://huggingface.co/internlm/internlm2-math-base-7b](https://huggingface.co/internlm/internlm2-math-base-7b)) - (Shanghai AI Lab) Adapted for math, Base and SFT checkpoints. [14, 20, 65, 24, 34, 17, 66]
- **Minerva**: ([https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html](https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html)) - (Google) Fine-tuned on scientific/math text. [6, 66]
- **Llemma**: ([https://arxiv.org/abs/2310.10631](https://arxiv.org/abs/2310.10631)) - (EleutherAI) Open models pre-trained for math. [7, 52, 66]
- **WizardMath**: ([https://arxiv.org/abs/2308.09583](https://arxiv.org/abs/2308.09583)) - Fine-tuned using reinforced Evol-Instruct.
- **MetaMath**: ([https://arxiv.org/abs/2309.12284](https://arxiv.org/abs/2309.12284)) - Focuses on augmenting math problems for fine-tuning.

### 5.2 Reasoning-Focused LLMs

*Models explicitly optimized for complex reasoning tasks, often via RL/search.*

- **OpenAI 'o' series (o1, o3-mini)**: Closed models, strong reasoning via RL/search. [3, 14, 10, 22, 13, 51, 18, 71, 6, 35, 27, 28, 47, 72, 73]
- **DeepSeek 'R' series (R1)**: ([https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)) - Closed models, strong reasoning via RL (GRPO). [3, 14, 10, 13, 15, 51, 71, 6, 11, 24, 35, 47, 74] Also see [Repo](https://github.com/deepseek-ai/DeepSeek-R1).
- **QaQ**: "QwQ-32B: Embracing the Power of Reinforcement Learning" [2025-03] [[blog](https://qwenlm.github.io/blog/qwq-32b/)] [[repo](https://github.com/QwenLM/QwQ)]
- **ERNIE X1**: "Baidu Unveils ERNIE 4.5 and Reasoning Model ERNIE X1" [2025-03] [[website](https://yiyan.baidu.com/)]
- **Gemini 2.0 Flash Thinking**: "Gemini 2.0 Flash Thinking" [2025-03] [[blog](https://deepmind.google/technologies/gemini/flash-thinking/)]

### 5.3 Leading General LLMs

*General-purpose models frequently evaluated on mathematical benchmarks.*

**OpenAI** [[link](https://openai.com/)]
- **GPT-3**: "Language Models are Few-Shot Learners" [2020-05] [[paper](https://arxiv.org/abs/2005.14165)]
- **GPT-4**: "GPT-4 Technical Report" [2023-03] [[paper](https://arxiv.org/abs/2303.08774)]
- **GPT-4o**: "GPT-4o System Card" [2024-10] [[paper](https://arxiv.org/abs/2410.21276)]
- **GPT-4.5**: "Introducing GPT-4.5" [2025-02] [[blog](https://openai.com/index/introducing-gpt-4-5/)]

**Google** [[link](https://gemini.google.com/)]
- **PaLM, Flan-PaLM**: [18, 6, 11]
- **Gemini 1.5 Pro**: "Gemini 1.5 Unlocking multimodal understanding across millions of tokens of context" [2024-05] [[paper](https://arxiv.org/abs/2403.05530)] [[blog](https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#gemini-15)]
- **Gemini 2**: "Gemini 1.5 Flash: Fast and Efficient Multimodal Reasoning" [2024-05] [[blog](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/)]
- **Gemma**: "Gemma: Open Models Based on Gemini Research and Technology" [2024-02] [[paper](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf)] [[blog](https://blog.google/technology/developers/gemma-open-models/)]
- **Gemma 2**: "Gemma 2: Improving Open Language Models at a Practical Size" [2024-06] [[paper](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)]

**Anthropic** [[link](https://www.anthropic.com/)]
- **Claude 3**: "The Claude 3 Model Family: Opus, Sonnet, Haiku" [2024-03] [[paper](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.pdf)] [[blog](https://www.anthropic.com/news/claude-3-family)]
- **Claude 3.5**: "Claude 3.5 Sonnet Model Card Addendum" [2024-10] [[paper](https://www-cdn.anthropic.com/fed9cc193a14b84131812372d8d5857f8f304c52/Model_Card_Claude_3_Addendum.pdf)] [[blog](https://www.anthropic.com/claude/haiku)]
- **Claude 3.7**: "Claude 3.7 Sonnet System Card" [2025-03] [[paper](https://www-cdn.anthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claude_3.7_Sonnet.pdf)] [[blog](https://www.anthropic.com/news/claude-3-7-sonnet)]

**Meta**
- **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models" [2023-02] [[paper](https://arxiv.org/abs/2302.13971)]
- **LLaMA 2**: "Llama 2: Open Foundation and Fine-Tuned Chat Models" [2023-07] [[paper](https://arxiv.org/abs/2307.09288)] [[repo](https://github.com/facebookresearch/llama)]
- **LLaMA 3**: "The Llama 3 Herd of Models" [2024-04] [[blog](https://ai.meta.com/blog/meta-llama-3/)] [[repo](https://github.com/meta-llama/llama3)] [[paper](https://arxiv.org/abs/2407.21783)]

**DeepSeek**
- **DeepSeek**: "DeepSeek LLM: Scaling Open-Source Language Models with Longtermism" [2024-01] [[paper](https://arxiv.org/abs/2401.02954)] [[repo](https://github.com/deepseek-ai/DeepSeek-LLM)]
- **DeepSeekMoE**: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" [2024-01] [[paper](https://arxiv.org/abs/2401.12246)] [[repo](https://github.com/deepseek-ai/DeepSeek-MoE)]
- **DeepSeek-V2**: "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model" [2024-05] [[paper](https://arxiv.org/abs/2405.04434)] [[repo](https://github.com/deepseek-ai/DeepSeek-V2)]
- **DeepSeek-V3**: "DeepSeek-V3 Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.19437)]

**Mistral**
- **Mistral**: "Mistral 7B" [2023-10] [[paper](https://arxiv.org/abs/2310.06825)] [[repo](https://github.com/mistralai/mistral-src)]
- **Mixtral**: "Mixtral of Experts" [2024-01] [[paper](https://arxiv.org/abs/2401.04088)] [[blog](https://mistral.ai/news/mixtral-of-experts/)] (Utilizes MoE)

**Qwen**
- **Qwen**: "Qwen Technical Report" [2023-09] [[paper](https://arxiv.org/abs/2309.16609)] [[repo](https://github.com/QwenLM/Qwen)]
- **Qwen 2**: "Qwen2: A Family of Open-Source LLMs with 14B-128B Parameters" [2024-09] [[paper](https://arxiv.org/abs/2409.12488)] [[repo](https://github.com/QwenLM/Qwen)]
- **Qwen2.5**: "Qwen2.5 Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.15115)]

**Microsoft Phi**
- **Phi-1.5**: "Textbooks Are All You Need II: phi-1.5 technical report" [2023-09] [[paper](https://arxiv.org/abs/2309.05463)] [[model](https://huggingface.co/microsoft/phi-1_5)]
- **Phi-2**: "Phi-2: The surprising power of small language models" [2023-12] [[blog](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)]
- **Phi-3**: "Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone" [2024-04] [[paper](https://arxiv.org/abs/2404.14219)]
- **Phi-4**: "Phi-4 Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.08905)]
- **Phi-4-Mini**: "Phi-4-Mini Technical Report: Compact yet Powerful Multimodal Language Models via Mixture-of-LoRAs" [2025-03] [[paper](https://arxiv.org/abs/2503.01743)]

**Other Publicly Available Models**
- **GPT-NeoX**: "GPT-NeoX-20B: An Open-Source Autoregressive Language Model" [2022-04] [ACL 2022 Workshop] [[paper](https://arxiv.org/abs/2204.06745)] [[repo](https://github.com/EleutherAI/gpt-neox)]
- **BLOOM**: "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model" [2022-11] [[paper](https://arxiv.org/abs/2211.05100)] [[model](https://huggingface.co/models?search=bigscience/bloom)]
- **Baichuan 2**: "Baichuan 2: Open Large-scale Language Models" [2023-09] [[paper](https://arxiv.org/abs/2309.10305)] [[repo](https://github.com/baichuan-inc/Baichuan2)]
- **YAYI2**: "YAYI 2: Multilingual Open-Source Large Language Models" [2023-12] [[paper](https://arxiv.org/abs/2312.14862)] [[repo](https://github.com/wenge-research/YAYI2)]
- **Orion**: "Orion-14B: Open-source Multilingual Large Language Models" [2024-01] [[paper](https://arxiv.org/abs/2401.06066)] [[repo](https://github.com/OrionStarAI/Orion)]
- **OLMo**: "OLMo: Accelerating the Science of Language Models" [2024-02] [[paper](https://arxiv.org/abs/2402.00838)] [[repo](https://github.com/allenai/OLMo)]
- **Yi**: "Yi: Open Foundation Models by 01.AI" [2024-03] [[paper](https://arxiv.org/abs/2403.04652)] [[repo](https://github.com/01-ai/Yi)]
- **OLMoE**: "OLMoE: Open Mixture-of-Experts Language Models" [2024-09] [[paper](https://arxiv.org/abs/2409.02060)][[repo](https://github.com/allenai/OLMoE?tab=readme-ov-file#pretraining)]
- **Yi-Lightning**: "Yi-Lightning Technical Report" [2024-12] [[paper](https://arxiv.org/abs/2412.01253)]
- **YuLan-Mini**: "YuLan-Mini: An Open Data-efficient Language Model" [2024-12] [[paper](https://arxiv.org/abs/2412.17743)]
- **OLMo 2**: "2 OLMo 2 Furious" [2024-12] [[paper](https://arxiv.org/abs/2501.00656)]
- **SmolLM2**: "SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model" [2025-02] [[paper](https://arxiv.org/abs/2502.02737)]

**Other Closed-Source / Commercial Models**
- **Grok 1**: "Grok 1: Open Release of Grok-1" [2024-03] [[blog](https://x.ai/news/grok-os)]
- **Grok 2**: "Grok-2 Beta Release" [2024-08] [[blog](https://x.ai/news/grok-2)]
- **Grok 3**: "Grok 3 Beta — The Age of Reasoning Agents" [2025-02] [[blog](https://x.ai/news/grok-3)]
- **Command R(+)**: [65, 47, 72]
- **InternLM series**: [14, 22, 20, 34]

## 6. Datasets & Benchmarks

*Benchmarks, datasets, and evaluation methodologies for mathematical LLMs*

### 6.1 Problem Solving Benchmarks

*Datasets primarily focused on evaluating mathematical problem-solving abilities.*

**Grade School:**
- **GSM8K**: ([https://huggingface.co/datasets/gsm8k](https://huggingface.co/datasets/gsm8k)) - Grade School Math 8K word problems. [Source Paper](https://arxiv.org/abs/2110.14168)
- **GSM8K Robustness Variants**: ([https://arxiv.org/abs/2402.06453](https://arxiv.org/abs/2402.06453)) - Robustness variants of GSM8K. [2, 38, 28, 73]
- **SVAMP, AddSub, ASDiv**: Additional grade-school level datasets.

**Competition Level:**
- **MATH**: ([https://huggingface.co/datasets/hendrycks/competition_math](https://huggingface.co/datasets/hendrycks/competition_math)) - High school competition problems (AMC, AIME). [Source Paper](https://arxiv.org/abs/2103.03874)
- **AIME subsets**: Specific subsets focusing on AIME problems.
- **OlympiadBench**: ([https://arxiv.org/abs/2311.07575](https://arxiv.org/abs/2311.07575)) - Challenging Olympiad-level problems.
- **Recent Exam Problems**: (e.g., USAMO 2025 [56, 57], Gaokao [9, 75]) - Problems drawn from recent official exams.
- **CHAMP**: "CHAMP: Mathematical Reasoning with Chain of Thought in Large Language Models" [2024-05] [[paper](https://arxiv.org/abs/2405.19254)]

**University/Advanced Level:**
- **MMLU (Math sections)**: ([https://paperswithcode.com/dataset/mmlu](https://paperswithcode.com/dataset/mmlu)) - Math portions of the Massive Multitask Language Understanding benchmark. [44, 53, 28, 47]
- **MMLU-Pro**: ([https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)) - More challenging MMLU variant. [Source Paper](https://arxiv.org/abs/2406.01574) [6, 31, 35, 30, 42, 47, 74, 76]
- **GPQA**: ([https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/gpqa](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/gpqa)) - Graduate-Level Google-Proof Q&A (STEM). [Source Paper](https://arxiv.org/abs/2311.12022) [35, 60, 30, 59, 42, 70, 72, 63, 64]
- **SciBench**: ([https://scibench-ucla.github.io/](https://scibench-ucla.github.io/)) - College-level STEM problems from textbooks. [Source Paper](https://arxiv.org/abs/2307.10635) [51, 71, 35, 52, 53, 54, 55]
- **U-MATH**: ([https://arxiv.org/abs/2402.10417](https://arxiv.org/abs/2402.10417)) - University-level problems. [6, 28, 73]
- **LILA**: "LILA: A Unified Benchmark for Mathematical Reasoning" [2022-10] [[paper](https://arxiv.org/abs/2210.17517)]

**Domain-Specific / Specialized:**
- **MathQA**: "MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms" [2022-05] [[paper](https://aclanthology.org/N19-1245/)] [[dataset](https://math-qa.github.io/)]
- **GeomVerse**: "GeomVerse: A Systematic Evaluation of Large Models for Geometric Reasoning" [2023-12] [[paper](https://arxiv.org/abs/2312.12241)]
- **TABMWP**: "Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning" [2023-01] [[paper](https://arxiv.org/abs/2209.14610)] [[dataset](https://github.com/lupantech/tabular-math-word-problems)] - Reasoning over text and tables. [13, 15]
- **AMPS**: "Measuring Mathematical Problem Solving With the MATH Dataset" [2021-03] [[paper](https://arxiv.org/abs/2103.03874)] (Associated dataset with MATH)
- **ROBUSTMATH**: "MATHATTACK: Attacking Large Language Models Towards Math Solving Ability" [2023-09] [[paper](https://arxiv.org/abs/2309.05690)]
- **MathEval**: ([https://openreview.net/forum?id=DexGnh0EcB](https://openreview.net/forum?id=DexGnh0EcB)) - Comprehensive suite across domains/difficulties. [51, 71, 9, 75, 77]
- **MR-GSM8K**: ([https://github.com/dvlab-research/MR-GSM8K](https://github.com/dvlab-research/MR-GSM8K)) - Meta-reasoning (evaluating solutions). [Source Paper](https://arxiv.org/abs/2312.17080) [43, 45, 48, 46]
- **FOLIO**: ([https://arxiv.org/abs/2209.00841](https://arxiv.org/abs/2209.00841)) - First-order logic reasoning. [6, 38, 24]

### 6.2 Theorem Proving Benchmarks

*Benchmarks focused on formal mathematical proof generation and verification.*

- **MiniF2F**: ([https://github.com/openai/miniF2F](https://github.com/openai/miniF2F)) - Formalized Olympiad/HS/UG problems (Lean, Isabelle, Metamath). [Source Paper](https://arxiv.org/abs/2109.00110) [6, 7, 24, 25, 52, 59, 36]
- **NaturalProofs, ProofNet, HolStep, CoqGym**: Other benchmarks in formal methods. [6, 7, 59, 36]

### 6.3 Multimodal Benchmarks

*Benchmarks incorporating visual or other non-textual information.*

- **MathVista**: ([https://mathvista.github.io/](https://mathvista.github.io/)) - Diverse visual contexts (charts, diagrams, etc.). [Source Paper](https://arxiv.org/abs/2310.02255)
- **ScienceQA**: ([https://scienceqa.github.io/](https://scienceqa.github.io/)) - Multimodal science questions (incl. math). [Source Paper](https://arxiv.org/abs/2209.09958) [13, 15, 58, 56, 37, 62, 67, 53, 54]
- **MATH-Vision**: ([https://arxiv.org/abs/2402.17177](https://arxiv.org/abs/2402.17177)) - Competition math with visual contexts. [Source Paper](https://arxiv.org/abs/2402.17177)
- **MathVerse**: "MathVerse: Assessing Visual Mathematical Understanding in Multimodal LLMs" [2024-04] [[paper](https://arxiv.org/abs/2404.13834)]
- **GeoQA / GeoEval**: Geometry problems benchmark. [Source Paper (Survey)](https://arxiv.org/abs/2412.11936) [2, 37]
- **FigureQA, ChartQA variants, DocReason25K**: Figure/chart/document understanding benchmarks. [13, 15, 37]
- **MM-MATH**: Process evaluation in multimodal math. [Source Paper (Survey)](https://arxiv.org/abs/2412.11936) [2, 37]
- **ErrorRadar**: "ErrorRadar: Evaluating the Multimodal Error Detection of LLMs in Educational Settings" [2024-03] [[paper](https://arxiv.org/abs/2403.03894)]

### 6.4 Training Datasets

*Datasets primarily used for training or fine-tuning models on mathematical tasks.*

- **MathInstruct**: ([https://huggingface.co/datasets/TIGER-Lab/MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)) - Instruction tuning dataset.
- **OpenMathInstruct-1**: "OpenMathInstruct-1: A 1.8M Math Instruction Tuning Dataset" [2024-03] [[paper](https://arxiv.org/abs/2402.10176)]
- **OpenMathInstruct-2**: "OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data" [2023-10] [[paper](https://arxiv.org/abs/2310.12591)]
- **MATH-Instruct**: "Mammoth: Building Math Generalist Models Through Hybrid Instruction Tuning" [2023-09] [[paper](https://arxiv.org/abs/2309.05653)]
- **OpenWebMath**: ([https://github.com/deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math)) (Corpus used by DeepSeekMath) - Large math-focused web corpus.
- **PRM800K**: Process reward data. [Source Paper (Lightman et al., 2023)](https://arxiv.org/abs/2305.15062) [6]
- **MetaMathQA**: Synthesized data. [Source Paper](https://arxiv.org/abs/2309.12284)
- **MathV360K**: ([https://huggingface.co/datasets/Zhiqiang007/MathV360K](https://huggingface.co/datasets/Zhiqiang007/MathV360K)) - Large multimodal math dataset. [Source Paper](https://arxiv.org/abs/2404.02693)
- **LeanNavigator generated data**: (4.7M theorems) [Source Paper](https://arxiv.org/abs/2503.04772) [8, 44, 46, 30]

## 7. Tools & Libraries

*Software tools, frameworks, and libraries relevant for working with LLMs in mathematics.*

**Interactive Theorem Provers (ITPs):**
- **Lean**: ([https://lean-lang.org/](https://lean-lang.org/)) [60, 19, 61, 62, 63, 64]
- **Isabelle**: ([https://isabelle.in.tum.de/](https://isabelle.in.tum.de/))
- **Coq**: ([https://coq.inria.fr/](https://coq.inria.fr/))

**LLM Interaction/Frameworks:**
- **LangChain**: ([https://www.langchain.com/](https://www.langchain.com/))
- **LMDeploy**: ([https://github.com/InternLM/lmdeploy](https://github.com/InternLM/lmdeploy)) [20, 66]
- **Guidance**: ([https://github.com/microsoft/guidance](https://github.com/microsoft/guidance)) [5, 34]
- **LPML**: "LPML: LLM-Prompting Markup Language for Mathematical Reasoning" [2023-09] [[paper](https://arxiv.org/abs/2309.04269)]

**Specific Math Reasoning Tools/Implementations:**
- **Math-CodeInterpreter**: "Solving Challenging Math Word Problems Using GPT-4 Code Interpreter with Code-Based Self-Verification" [2023-08] [[paper](https://arxiv.org/abs/2308.07921)]
- **MathPrompter**: "MathPrompter: Mathematical Reasoning Using Large Language Models" [2023-07] [[paper](https://arxiv.org/abs/2303.05398)]
- **BEATS**: "BEATS: Optimizing LLM Mathematical Capabilities with BackVerify and Adaptive Disambiguate based Efficient Tree Search" [2023-10] [[paper](https://arxiv.org/abs/2310.04344)]
- **BoostStep**: "BoostStep: Boosting mathematical capability of Large Language Models via improved single-step reasoning" [2023-10] [[paper](https://arxiv.org/abs/2310.08573)]

**Evaluation Frameworks:**
- **OpenCompass**: ([https://github.com/open-compass/opencompass](https://github.com/open-compass/opencompass)) [22, 34, 17, 73]

**Data Processing:**
- **IBM Data Prep Kit**: ([https://github.com/IBM/data-prep-kit](https://github.com/IBM/data-prep-kit)) [71]
- **Datatrove**: ([https://github.com/huggingface/datatrove](https://github.com/huggingface/datatrove)) [71]

**Fine-tuning / Adaptation:**
- **LoRA**: "LoRA: Low-Rank Adaptation of Large Language Models" [2021-06] [[paper](https://arxiv.org/abs/2106.09685)]

## 8. Challenges & Future Directions

*Research on capabilities, limitations, and future directions of LLMs for mathematics*

**Challenges:**
- **Reliability & Soundness**: Overcoming calculation errors and logical inconsistencies (hallucinations) remains crucial, especially for formal proofs. [1, 6, 58, 56, 46]
- **Complexity Handling**: Scaling reasoning to handle very long proofs and deeply complex problems is an ongoing challenge. [6, 58, 56, 33]
- **Multimodal Integration**: Effectively fusing and reasoning over text, diagrams, charts, etc. needs improvement. [2, 6, 68, 57]
- **Evaluation**: Developing robust benchmarks that assess true reasoning (not just final answers) and resist data contamination is vital. [1, 2, 6, 56, 33, 43, 45, 75]
- **Data Scarcity**: Need for more high-quality, large-scale data, especially with reasoning steps, formal proofs, and multimodal contexts. [1, 2, 22, 6, 46, 30]
- **Interpretability**: Understanding *how* LLMs reach conclusions is key for trust. [1, 78, 6]
- **Robustness**: Models can be brittle to input perturbations. See "A Causal Framework to Quantify the Robustness of Mathematical Reasoning with Language Models" [2023-05] [[paper](https://arxiv.org/abs/2305.14291)] and "MATHATTACK: Attacking Large Language Models Towards Math Solving Ability" [2023-09] [[paper](https://arxiv.org/abs/2309.05690)].
- **Tokenization Impact**: Arithmetic performance can be affected by tokenization. See "How Well Do Large Language Models Perform in Arithmetic Tasks?" [2023-04] [[paper](https://arxiv.org/abs/2304.02015)].

**Future Directions & Emerging Research:**
- **Hybrid Neural-Symbolic Methods**: Combining LLM strengths with symbolic rigor. [1, 2, 6, 9, 11, 56, 33, 36, 37, 23, 75]
- **Verification & Correction**: Better automated verification and self-correction. [1] See also "Learning from Mistakes Makes LLM Better Reasoner" [2023-10] [[paper](https://arxiv.org/abs/2310.13522)].
- **Enhanced Tool Use**: Seamless integration with more sophisticated external tools. [1]
- **Advanced Training Paradigms**: Refining RL (especially process rewards) and self-improvement. [1]
- **Focus on Reasoning Process**: Training and evaluating intermediate steps, not just final answers. [2]
- **Multimodal Advancements**: Better architectures and data for multimodal fusion. [1] See "Math-PUMA: Progressive Upward Multimodal Alignment for Math Reasoning Enhancement" [2024-04] [[paper](https://arxiv.org/abs/2404.01166)].
- **Interpretability & Explainability**: Making reasoning processes transparent. [1]
- **Improved Evaluation**: Creating more challenging, robust, process-oriented benchmarks. [1]
- **Interactive Reasoning & Human-Centric Design**: LLM-human collaboration. [1] See "Exploring Pre-service Teachers' Perceptions of Large Language Models-Generated Hints in Online Mathematics Learning" [2023-06] [[paper](https://arxiv.org/abs/2306.17129)].
- **Continual Learning**: Enabling models to learn continuously from new data or mistakes. See "Learning from Mistakes Makes LLM Better Reasoner" [2023-10] [[paper](https://arxiv.org/abs/2310.13522)].
- **Program Search**: Using LLMs for mathematical discovery via program search. See "Mathematical Discoveries from Program Search with Large Language Models" [2023-11] [[paper](https://arxiv.org/abs/2311.13444)].
- **Scaling Relationships**: Understanding how scale affects math reasoning. See "Scaling Relationship on Learning Mathematical Reasoning with Large Language Models" [2023-10] [[paper](https://arxiv.org/abs/2310.07177)].
- **Capability Analysis**: Analyzing inherent capabilities of models. See "Common 7B Language Models Already Possess Strong Math Capabilities" [2023-10] [[paper](https://arxiv.org/abs/2310.04560)].

## 9. Contributing

We are looking for contributors to help build this resource. Please read the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

## 10. Citation

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

## 11. License

This repository is licensed under the [MIT License](LICENSE).