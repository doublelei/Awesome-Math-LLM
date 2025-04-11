Okay, here is a README file in Markdown format for the `awesome-LLM-in-math` repository, based on the research report:

# Awesome LLM in Math

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of awesome resources, papers, code, and datasets for Large Language Models (LLMs) focused on Mathematics.

The field of Artificial Intelligence (AI) has seen transformative progress with Large Language Models (LLMs). Leveraging these models for complex reasoning, particularly in mathematics, is a critical area of research.[1, 2] Mathematical reasoning is fundamental to human intelligence and essential across science and engineering.[1, 2] Evaluating and enhancing the mathematical capabilities of AI, from solving word problems to assisting with complex proofs [1, 3, 4, 5], is key to advancing AI cognition. This repository aims to organize the growing body of research and resources in this dynamic field.

## Contents

\-(\#-surveys--overviews)
\-(\#-core-reasoning-techniques)
\-(\#chain-of-thought--prompting-strategies)
\-(\#search--planning)
\-(\#reinforcement-learning--reward-modeling)
\-(\#self-improvement--self-training)
\-(\#tool-use--augmentation)

  - [Neurosymbolic Methods](https://www.google.com/search?q=%23neurosymbolic-methods)
    \-(\#-mathematical-domains--tasks)
  - [Arithmetic & Word Problems](https://www.google.com/search?q=%23arithmetic--word-problems)
  - [Algebra, Geometry, Calculus, etc.](https://www.google.com/search?q=%23algebra-geometry-calculus-etc)
  - [Competition Math](https://www.google.com/search?q=%23competition-math)
    \-(\#formal-theorem-proving)
    \-(\#symbolic-manipulation)
    \-(\#-multimodal-mathematical-reasoning)
  - [Models](https://www.google.com/search?q=%23-models)
    \-(\#math-specialized-llms)
    \-(\#reasoning-focused-llms)
      - [Leading General LLMs](https://www.google.com/search?q=%23leading-general-llms)
        \-(\#-datasets--benchmarks)
        \-(\#problem-solving-benchmarks)
        \-(\#theorem-proving-benchmarks)
        \-(\#multimodal-benchmarks)
        \-(\#training-datasets)
        \-(\#-tools--libraries)
  - [Challenges & Future Directions](https://www.google.com/search?q=%23-challenges--future-directions)
  - [Contributing](https://www.google.com/search?q=%23contributing)

## 📜 Surveys & Overviews

  * ([https://arxiv.org/abs/2503.17726](https://arxiv.org/abs/2503.17726)) - Forootani, A. (arXiv:2503.17726). Covers evolution, methodologies (CoT, Tools, RL), models, datasets, challenges. [1, 6]
  * ([https://arxiv.org/abs/2502.14333](https://arxiv.org/abs/2502.14333)) - Wei, T.-R., et al. (arXiv:2502.14333). Focuses on feedback mechanisms (step/outcome, training/training-free). [7, 8, 9]
  * ([https://arxiv.org/abs/2502.17419](https://arxiv.org/abs/2502.17419)) - Zeng, Z., et al. (arXiv:2502.17419). Discusses System 2 reasoning, MCTS, reward modeling, self-improvement, RL. [3, 10, 11]
  * ([https://arxiv.org/abs/2412.11936](https://arxiv.org/abs/2412.11936)) - Yan, Y., et al. (arXiv:2412.11936). First survey on multimodal math reasoning, covering benchmarks, methods, challenges. [2]
  * ([https://arxiv.org/abs/2402.06196](https://arxiv.org/abs/2402.06196)) - Minaee, S., et al. (arXiv:2402.06196). General overview of LLM families, training, datasets, evaluation. [12]
  * ([https://github.com/atfortes/Awesome-LLM-Reasoning](https://github.com/atfortes/Awesome-LLM-Reasoning)) - Curated list focusing on general LLM reasoning techniques. [13]
  * ([https://github.com/zzli2022/Awesome-System2-Reasoning-LLM](https://github.com/zzli2022/Awesome-System2-Reasoning-LLM)) - Curated list focusing on System 2 reasoning (RL, MCTS, Self-Improve). [14]
  * ([https://github.com/InfiMM/Awesome-Multimodal-LLM-for-Math-STEM](https://github.com/InfiMM/Awesome-Multimodal-LLM-for-Math-STEM)) - Curated list for MLLMs in Math/STEM. [15]

## 🧠 Core Reasoning Techniques

### Chain-of-Thought & Prompting Strategies

  * **Concept:** Prompting LLMs to output intermediate reasoning steps before the final answer. Variants include self-checking, reflection, planning, Long CoT, and Algorithmic Prompting. [6, 7, 16]
  * **Papers:**
      * ([https://arxiv.org/abs/2201.11903](https://arxiv.org/abs/2201.11903)) - Wei, J., et al. (NeurIPS 2022). Foundational CoT paper.
      * ([https://research.google/blog/teaching-language-models-to-reason-algorithmically/](https://research.google/blog/teaching-language-models-to-reason-algorithmically/)) - Introduces Algorithmic Prompting. [16, 17]
      * ([https://arxiv.org/abs/2301.13379](https://arxiv.org/abs/2301.13379)) - Lyu, Q., et al. (IJCNLP-AACL 2023). Focuses on improving the faithfulness of CoT. [18]

### Search & Planning

  * **Concept:** Exploring multiple potential solution paths using search algorithms.
  * **Techniques:**
      * **Tree-of-Thoughts (ToT):** Explores different reasoning branches and self-evaluates progress. [13, 6]
          * Paper:([https://arxiv.org/abs/2305.10601](https://arxiv.org/abs/2305.10601)) - Yao, S., et al. (NeurIPS 2023). [13, 19]
          * Code:([https://github.com/kyegomez/Tree-of-Thoughts-LLM](https://github.com/kyegomez/Tree-of-Thoughts-LLM))
      * **Graph-of-Thoughts (GoT):** Represents reasoning as a graph (implementation debated). [20]
          * Paper:([https://arxiv.org/abs/2308.09687](https://arxiv.org/abs/2308.09687)) - Besta, M., et al. (arXiv 2023). [21]
      * **Monte Carlo Tree Search (MCTS):** Simulation-based search for navigating solution space. [14, 22, 9, 11]
          * Paper:([https://arxiv.org/abs/2502.10000](https://arxiv.org/abs/2502.10000)) - Qi, Z., et al. (arXiv 2025). Uses MCTS in self-improvement. [22, 23]
      * **Best-First Search (BFS):** Used in theorem proving and problem solving. [24]
          * Paper:([https://arxiv.org/abs/2502.03438](https://arxiv.org/abs/2502.03438)) - Xin, R., et al. (arXiv 2025). [24, 25]

### Reinforcement Learning & Reward Modeling

  * **Concept:** Optimizing LLMs using feedback signals (rewards) through RL algorithms like PPO, DPO, GRPO. [14, 6, 7, 9, 11, 26, 27]
  * **Reward Models:**
      * **Outcome Reward Models (ORM):** Evaluate the final answer. [14, 10, 6, 7, 9, 11, 26, 27]
      * **Process Reward Models (PRM):** Evaluate intermediate reasoning steps. [14, 10, 6, 7, 9, 11, 26, 27]
  * **Papers/Techniques:**
      * ([https://arxiv.org/abs/2501.09686](https://arxiv.org/abs/2501.09686)) - Guo, D., et al. (arXiv 2025). Introduces GRPO. [14, 6, 11, 24, 28]
      * ([https://arxiv.org/abs/2312.08935](https://arxiv.org/abs/2312.08935)) - Wang, P., et al. (arXiv 2023). Uses automated process supervision. [9, 27]
      * (https://arxiv.org/abs/2406.13559) - Luo, L., et al. (arXiv 2024). Automated PRM data generation using MCTS. [9, 29]
      * [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) - Rafailov, R., et al. (NeurIPS 2023). [24]
      * [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) - Schulman, J., et al. (arXiv 2017). [24]

### Self-Improvement & Self-Training

  * **Concept:** LLMs generate their own training data through exploration (e.g., MCTS), evaluate attempts, and learn from successes. [14, 10, 22, 11, 27]
  * **Papers/Techniques:**
      * ([https://arxiv.org/abs/2203.14465](https://arxiv.org/abs/2203.14465)) - Zelikman, E., et al. (arXiv 2022). Foundational self-taught reasoner. [14, 15, 12]
      * ([https://arxiv.org/abs/2502.10000](https://arxiv.org/abs/2502.10000)) - Qi, Z., et al. (arXiv 2025). Combines MCTS, RL, and self-evolution. [22, 15, 23]
      * ([https://arxiv.org/abs/2403.09629](https://arxiv.org/abs/2403.09629)) - Zelikman, E., et al. (arXiv 2024). Token-level exploration during training. [15]
      * ([https://arxiv.org/abs/2308.08998](https://arxiv.org/abs/2308.08998)) - Gulcehre, C., et al. (arXiv 2023). [15]
      * ([https://arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651)) - Madaan, A., et al. (NeurIPS 2023). Test-time refinement. [15, 30]

### Tool Use & Augmentation

  * **Concept:** Enabling LLMs to call external tools (calculators, code interpreters, solvers, search engines) to overcome computational limitations. [6, 16]
  * **Papers/Techniques:**
      * [PAL: Program-Aided Language Models](https://arxiv.org/abs/2211.10435) - Gao, L., et al. (ICML 2023). Generates code executed by an interpreter. [6, 31, 32]
          * Code: [reasoning-machines/pal](https://github.com/reasoning-machines/pal) [4]
      * (https://arxiv.org/abs/2303.09014) - Paranjape, B., et al. (arXiv 2023). Dynamically selects and uses tools. [13, 6, 33, 34, 25, 35]
          * Code (Guidance library used in ART): [microsoft/guidance](https://github.com/microsoft/guidance) [5, 16]
      * ([https://arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761)) - Schick, T., et al. (NeurIPS 2023). LLM learns to use APIs. [24]
      * ([https://arxiv.org/abs/2305.12295](https://arxiv.org/abs/2305.12295)) - Pan, L., et al. (EMNLP 2023 Findings). Integrates logical solvers. [13, 18]
      * ([https://arxiv.org/abs/2310.05726](https://arxiv.org/abs/2310.05726)) - Ye, X., et al. (NeurIPS 2023). Integrates SAT solvers. [13, 18]

### Neurosymbolic Methods

  * **Concept:** Integrating neural models (LLMs) with symbolic reasoning methods for enhanced reliability, interpretability, and rule-following. [36, 37, 23]
  * **Papers/Techniques:**
      * ([https://arxiv.org/abs/2502.01657](https://arxiv.org/abs/2502.01657)) - Dhanraj, V., & Eliasmith, C. (arXiv 2025). Uses Vector Symbolic Algebras (VSAs) to encode hidden states.
      * ([https://arxiv.org/abs/2502.09061](https://arxiv.org/abs/2502.09061)) - Suresh, A., et al. (arXiv 2025). Balances constrained decoding with reasoning flexibility. [38, 39]
      * ([https://arxiv.org/abs/2503.09986](https://arxiv.org/abs/2503.09986)) - Jin, C., et al. (arXiv 2025). LLM predicts operators to guide symbolic regression. [40, 41]
      * ([https://arxiv.org/abs/2503.05641](https://arxiv.org/abs/2503.05641)) - Chen, Z., et al. (arXiv 2025). Routes problems to expert LLMs based on symbolic skill representations. [30, 42]

## 🧮 Mathematical Domains & Tasks

### Arithmetic & Word Problems

  * **Focus:** Solving grade-school level math word problems involving multi-step arithmetic.
  * **Key Benchmark:** GSM8K [20, 43, 44, 45]
  * **Papers:**
      * ([https://arxiv.org/abs/2110.14168](https://arxiv.org/abs/2110.14168)) - Cobbe, K., et al. (arXiv 2021). Introduces GSM8K. [21, 44, 46, 47]
      * ([https://arxiv.org/abs/2312.17080](https://arxiv.org/abs/2312.17080)) - Liu, Z., et al. (arXiv 2023). Introduces MR-GSM8K for evaluating error detection. [45, 48, 46]

### Algebra, Geometry, Calculus, etc.

  * **Focus:** Problems spanning standard high school and undergraduate curricula.
  * **Key Benchmarks:** MATH [6, 38, 49, 50, 28], SciBench [51, 52, 53, 54, 55]
  * **Papers:**
      * ([https://arxiv.org/abs/2103.03874](https://arxiv.org/abs/2103.03874)) - Hendrycks, D., et al. (arXiv 2021). Introduces the MATH benchmark.
      * ([https://arxiv.org/abs/2307.10635](https://arxiv.org/abs/2307.10635)) - Wang, X., et al. (NeurIPS 2023 Datasets and Benchmarks). Introduces SciBench. [52, 54, 55]

### Competition Math

  * **Focus:** Challenging problems from competitions like AMC, AIME, IMO.
  * **Key Benchmarks:** MATH [6, 38, 49, 50, 28], AIME subsets [38, 52], OlympiadBench [2, 38]
  * **Papers:**
      * ([https://arxiv.org/abs/2503.21934](https://arxiv.org/abs/2503.21934)) - Liu, Z., et al. (arXiv 2025). Evaluates SOTA models on recent USAMO problems. [56, 57]
      * ([https://arxiv.org/abs/2504.01995](https://arxiv.org/abs/2504.01995)) - Mahdavi, H., et al. (arXiv 2025). Human evaluation of LLM proofs for Olympiad problems.

### Formal Theorem Proving

  * **Focus:** Generating formal mathematical proofs verifiable by Interactive Theorem Provers (ITPs) like Lean, Isabelle, Coq. [6, 58, 24, 46]
  * **Key Benchmark:** MiniF2F [6, 7, 24, 52, 59, 36]
  * **Key Tool:** Lean Theorem Prover [60, 19, 61, 62, 63, 64]
  * **Papers/Techniques:**
      * ([https://arxiv.org/abs/2503.04772](https://arxiv.org/abs/2503.04772)) - Jiang, A., et al. (arXiv 2025). Automated formal proof data synthesis. [8, 44, 46, 30]
      * (https://arxiv.org/abs/2502.03438) - Xin, R., et al. (arXiv 2025). Uses BFS and expert iteration. [24, 25]
      * [MiniF2F: a cross-system benchmark for formal Olympiad-level mathematics](https://arxiv.org/abs/2109.00110) - Zheng, Q., et al. (arXiv 2021). Introduces MiniF2F. [7, 21, 59, 36]

### Symbolic Manipulation

  * **Focus:** Using LLMs for tasks involving symbolic expressions, potentially integrating with symbolic solvers (SymPy, Mathematica). [13, 6]
  * **Papers/Techniques:** See [Neurosymbolic Methods](https://www.google.com/search?q=%23neurosymbolic-methods) section.

## 🖼️ Multimodal Mathematical Reasoning

  * **Focus:** Solving math problems involving non-textual information (diagrams, plots, tables, handwritten equations). [2, 15, 65, 6]
  * **Papers:**
      * ([https://arxiv.org/abs/2412.11936](https://arxiv.org/abs/2412.11936)) - Yan, Y., et al. (arXiv 2024). Comprehensive survey. [2, 37, 66]
      * ([https://arxiv.org/abs/2310.02255](https://arxiv.org/abs/2310.02255)) - Lu, P., et al. (NeurIPS 2023). Introduces MathVista benchmark.
      * ([https://arxiv.org/abs/2209.09958](https://arxiv.org/abs/2209.09958)) - Lu, P., et al. (NeurIPS 2022). Introduces ScienceQA benchmark. [58, 56, 67, 54]
      * ([https://arxiv.org/abs/2402.17177](https://arxiv.org/abs/2402.17177)) - Wang, X., et al. (NeurIPS 2024). Introduces MATH-Vision benchmark.
  * **Models (Examples):** GPT-4V [6, 68], Gemini Vision [2, 6, 68], Qwen-VL [65, 6], LLaVA variants [2, 6, 69], AtomThink.[6, 11]
  * **Datasets & Benchmarks:**
      * [MathVista](https://mathvista.github.io/)
      * ([https://scienceqa.github.io/](https://scienceqa.github.io/)) [13, 15, 58, 56, 62, 67, 53, 54]
      * ([https://arxiv.org/abs/2402.17177](https://arxiv.org/abs/2402.17177))
      * [GeoQA/GeoEval](https://arxiv.org/abs/2412.11936) [2, 37]
      * [MathV360K](https://huggingface.co/datasets/Zhiqiang007/MathV360K) (Training Data)
      * See also:([https://github.com/InfiMM/Awesome-Multimodal-LLM-for-Math-STEM\#mllm-mathstem-dataset](https://www.google.com/search?q=https://github.com/InfiMM/Awesome-Multimodal-LLM-for-Math-STEM%23mllm-mathstem-dataset)) [13]

## 🤖 Models

### Math-Specialized LLMs

  * (https://github.com/deepseek-ai/DeepSeek-Math) - (DeepSeek) Pre-trained on math-heavy web data. Base, Instruct, RL versions. [14, 65, 51, 54]
  * [Qwen-Math / Qwen2.5-Math](https://arxiv.org/abs/2406.13559) - (Alibaba) Math-focused versions, strong performance. [3, 27, 30, 42, 70]
  * [InternLM-Math](https://huggingface.co/internlm/internlm2-math-base-7b) - (Shanghai AI Lab) Adapted for math, Base and SFT checkpoints. [14, 20, 65, 24, 34, 17, 66]
  * [Minerva](https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html) - (Google) Fine-tuned on scientific/math text. [6, 66]
  * [Llemma](https://arxiv.org/abs/2310.10631) - (EleutherAI) Open models pre-trained for math. [7, 52, 66]
  * [WizardMath](https://arxiv.org/abs/2308.09583) - Fine-tuned using reinforced Evol-Instruct.
  * [MetaMath](https://arxiv.org/abs/2309.12284) - Focuses on augmenting math problems for fine-tuning.

### Reasoning-Focused LLMs

  * **OpenAI 'o' series (o1, o3-mini):** Closed models, strong reasoning via RL/search. [3, 14, 10, 22, 13, 51, 18, 71, 6, 35, 27, 28, 47, 72, 73]
  * **DeepSeek 'R' series (R1):** Closed models, strong reasoning via RL (GRPO). [3, 14, 10, 13, 15, 51, 71, 6, 11, 24, 35, 47, 74]

### Leading General LLMs

  * **OpenAI:** GPT-3, GPT-4, GPT-4o [3, 18, 6, 16, 68, 66, 28]
  * **Google:** PaLM, Flan-PaLM [18, 6, 11], Gemini series [2, 6, 56, 68, 47, 63]
  * **Anthropic:** Claude series [6, 35, 47, 72, 63, 64]
  * **Meta:** Llama series (Llama 2, Llama 3, Llama 3.3, Llama 4) [4, 65, 6, 35, 63, 73]
  * **DeepSeek:** DeepSeek V2, V3 [14, 51, 71, 30, 47, 74]
  * **Mistral:** Mistral 7B, Mixtral
  * **Other:** Command R(+) [65, 47, 72], Qwen series [3, 14, 30], InternLM series [14, 22, 20, 34]

## 📊 Datasets & Benchmarks

### Problem Solving Benchmarks

  * **Grade School:**
      * ([https://huggingface.co/datasets/gsm8k](https://huggingface.co/datasets/gsm8k)) - Grade School Math 8K word problems.
      * ([https://arxiv.org/abs/2402.06453](https://arxiv.org/abs/2402.06453)) - Robustness variants of GSM8K. [2, 38, 28, 73]
      * SVAMP, AddSub, ASDiv
  * **Competition Level:**
      * ([https://huggingface.co/datasets/hendrycks/competition\_math](https://huggingface.co/datasets/hendrycks/competition_math)) - High school competition problems (AMC, AIME).
      * AIME subsets
      * OlympiadBench
      * Recent Exam Problems (e.g., USAMO 2025 [56, 57], Gaokao [9, 75])
  * **University/Advanced Level:**
      * ([https://arxiv.org/abs/2402.10417](https://arxiv.org/abs/2402.10417)) - University-level problems. [6, 28, 73]
      * (https://scibench-ucla.github.io/) - College-level STEM problems from textbooks. [51, 71, 35, 52, 53, 54, 55]
      * [GPQA](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/gpqa) - Graduate-Level Google-Proof Q\&A (STEM). [35, 60, 30, 59, 42, 70, 72, 63, 64]
      * [MMLU](https://paperswithcode.com/dataset/mmlu) (Math sections) [44, 53, 28, 47]
      * [MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) - More challenging MMLU variant. [6, 31, 35, 30, 42, 47, 74, 76]
  * **Specialized/Diagnostic:**
      * [MathEval](https://openreview.net/forum?id=DexGnh0EcB) - Comprehensive suite across domains/difficulties. [51, 71, 9, 75, 77]
      * (https://github.com/dvlab-research/MR-GSM8K) - Meta-reasoning (evaluating solutions). [43, 45, 48, 46]
      * [FOLIO](https://arxiv.org/abs/2209.00841) - First-order logic reasoning. [6, 38, 24]
      * TabMWP [13, 15] - Reasoning over text and tables.

### Theorem Proving Benchmarks

  * [MiniF2F](https://github.com/openai/miniF2F) - Formalized Olympiad/HS/UG problems (Lean, Isabelle, Metamath). [6, 7, 24, 25, 52, 59, 36]
  * NaturalProofs, ProofNet, HolStep, CoqGym [6, 7, 59, 36]

### Multimodal Benchmarks

  * [MathVista](https://mathvista.github.io/) - Diverse visual contexts (charts, diagrams, etc.).
  * ([https://scienceqa.github.io/](https://scienceqa.github.io/)) - Multimodal science questions (incl. math). [13, 15, 58, 56, 37, 62, 67, 53, 54]
  * ([https://arxiv.org/abs/2402.17177](https://arxiv.org/abs/2402.17177)) - Competition math with visual contexts.
  * GeoQA / GeoEval [2, 37] - Geometry problems.
  * FigureQA, ChartQA variants, DocReason25K [13, 15, 37] - Figure/chart/document understanding.
  * MM-MATH [2, 37] - Process evaluation in multimodal math.

### Training Datasets

  * [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
  * [OpenWebMath](https://github.com/deepseek-ai/DeepSeek-Math) (Corpus used by DeepSeekMath)
  * PRM800K [6] - Process reward data.
  * MetaMathQA  - Synthesized data.
  * [MathV360K](https://huggingface.co/datasets/Zhiqiang007/MathV360K) - Large multimodal math dataset.
  * LeanNavigator generated data (4.7M theorems) [8, 44, 46, 30]

## 🛠️ Tools & Libraries

  * **Interactive Theorem Provers (ITPs):**
      * [Lean](https://lean-lang.org/) [60, 19, 61, 62, 63, 64]
      * Isabelle, Coq
  * **LLM Interaction/Frameworks:**
      * [LangChain](https://www.langchain.com/)
      * [LMDeploy](https://github.com/InternLM/lmdeploy) [20, 66]
      * [Guidance](https://github.com/microsoft/guidance) [5, 34]
  * **Evaluation Frameworks:**
      * [OpenCompass](https://github.com/open-compass/opencompass) [22, 34, 17, 73]
  * **Data Processing:**
      * (https://github.com/IBM/data-prep-kit) [71]
      * [Datatrove](https://github.com/huggingface/datatrove) [71]

## 챌 Challenges & Future Directions

  * **Reliability & Soundness:** Overcoming calculation errors and logical inconsistencies (hallucinations) remains crucial, especially for formal proofs. [1, 6, 58, 56, 46]
  * **Complexity Handling:** Scaling reasoning to handle very long proofs and deeply complex problems is an ongoing challenge. [6, 58, 56, 33]
  * **Multimodal Integration:** Effectively fusing and reasoning over text, diagrams, charts, etc. needs improvement. [2, 6, 68, 57]
  * **Evaluation:** Developing robust benchmarks that assess true reasoning (not just final answers) and resist data contamination is vital. [1, 2, 6, 56, 33, 43, 45, 75]
  * **Data Scarcity:** Need for more high-quality, large-scale data, especially with reasoning steps, formal proofs, and multimodal contexts. [1, 2, 22, 6, 46, 30]
  * **Interpretability:** Understanding *how* LLMs reach conclusions is key for trust. [1, 78, 6]
  * **Future Directions:** Hybrid Neural-Symbolic methods, better verification/correction, enhanced tool use, advanced RL/Self-Improvement, focus on reasoning process, interactive reasoning, improved evaluation. [1, 2, 6, 9, 11, 56, 33, 36, 37, 23, 75]

## Contributing

Contributions are welcome\! Please feel free to submit a Pull Request or open an Issue to add papers, code, datasets, or other relevant resources. Please follow the existing format.


Works cited
A Survey on Mathematical Reasoning and Optimization with Large Language Models - arXiv, accessed April 9, 2025, https://arxiv.org/abs/2503.17726
arxiv.org, accessed April 9, 2025, https://arxiv.org/abs/2412.11936
From System 1 to System 2: A Survey of Reasoning Large Language Models - arXiv, accessed April 9, 2025, https://arxiv.org/pdf/2502.17419?
MathFusion: Enhancing Mathematic Problem-solving of LLM through Instruction Fusion - arXiv, accessed April 9, 2025, https://arxiv.org/html/2503.16212v1
Evaluating language models for mathematics through interactions - PMC, accessed April 9, 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC11181017/
zzli2022/Awesome-System2-Reasoning-LLM: Latest ... - GitHub, accessed April 9, 2025, https://github.com/zzli2022/Awesome-System2-Reasoning-LLM
From System 1 to System 2: A Survey of Reasoning Large Language Models - arXiv, accessed April 9, 2025, https://arxiv.org/html/2502.17419v2
rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking, accessed April 9, 2025, https://thenaai.org/uploads/admin/article_thumb/20250210/94b8994058899cefa8eeddddb5865651.pdf
luban-agi/Awesome-LLM-reasoning - GitHub, accessed April 9, 2025, https://github.com/luban-agi/Awesome-LLM-reasoning
atfortes/Awesome-LLM-Reasoning: Reasoning in LLMs ... - GitHub, accessed April 9, 2025, https://github.com/atfortes/Awesome-LLM-Reasoning
InfiMM/Awesome-Multimodal-LLM-for-Math-STEM - GitHub, accessed April 9, 2025, https://github.com/InfiMM/Awesome-Multimodal-LLM-for-Math-STEM
Awesome-LLM: a curated list of Large Language Model - GitHub, accessed April 9, 2025, https://github.com/turna1/Awesome-Multmodal_LLM
Awesome-LLM: a curated list of Large Language Model - GitHub, accessed April 9, 2025, https://github.com/Hannibal046/Awesome-LLM
jd-coderepos/awesome-llms: 🤓 A collection of AWESOME structured summaries of Large Language Models (LLMs) - GitHub, accessed April 9, 2025, https://github.com/jd-coderepos/awesome-llms
horseee/Awesome-Efficient-LLM: A curated list for Efficient Large Language Models - GitHub, accessed April 9, 2025, https://github.com/horseee/Awesome-Efficient-LLM
JShollaj/awesome-llm-interpretability - GitHub, accessed April 9, 2025, https://github.com/JShollaj/awesome-llm-interpretability
arxiv.org, accessed April 9, 2025, https://arxiv.org/pdf/2503.17726
A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics - arXiv, accessed April 9, 2025, https://arxiv.org/html/2502.14333v1
[2502.14333] A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics - arXiv, accessed April 9, 2025, https://arxiv.org/abs/2502.14333
arxiv.org, accessed April 9, 2025, https://arxiv.org/pdf/2502.14333
arxiv.org, accessed April 9, 2025, https://arxiv.org/pdf/2502.17419
[2402.06196] Large Language Models: A Survey - arXiv, accessed April 9, 2025, https://arxiv.org/abs/2402.06196
Evaluating Large Language Models: A Comprehensive Survey - arXiv, accessed April 9, 2025, https://arxiv.org/pdf/2310.19736
Towards Large Reasoning Models: A Survey of Reinforced Reasoning with Large Language Models - arXiv, accessed April 9, 2025, https://arxiv.org/html/2501.09686v1
LeanProgress: Guiding Search for Neural Theorem Proving via Proof Progress Prediction, accessed April 9, 2025, https://arxiv.org/html/2502.17925v2
[2503.21934] Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad - arXiv, accessed April 9, 2025, https://arxiv.org/abs/2503.21934
[2504.01995] Brains vs. Bytes: Evaluating LLM Proficiency in Olympiad Mathematics - arXiv, accessed April 9, 2025, https://arxiv.org/abs/2504.01995
Benchmarking LLMs' Math Reasoning Abilities against Hard Perturbations - arXiv, accessed April 9, 2025, https://arxiv.org/pdf/2502.06453
Teaching language models to reason algorithmically - Google Research, accessed April 9, 2025, https://research.google/blog/teaching-language-models-to-reason-algorithmically/
[2502.03438] BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving - arXiv, accessed April 9, 2025, https://arxiv.org/abs/2502.03438
reasoning-machines/pal: PaL: Program-Aided Language Models (ICML 2023) - GitHub, accessed April 9, 2025, https://github.com/reasoning-machines/pal
Shuyib/PAL: Using an LLM to answer math word problems common in IRL. Program aided language modeling. - GitHub, accessed April 9, 2025, https://github.com/Shuyib/PAL
ART: Automatic multi-step reasoning and tool-use for large language models, accessed April 9, 2025, https://paperswithcode.com/paper/art-automatic-multi-step-reasoning-and-tool
Automatic Reasoning and Tool-use (ART) - Prompt Engineering Guide, accessed April 9, 2025, https://www.promptingguide.ai/techniques/art
ART: Automatic multi-step reasoning and tool-use for large language models - arXiv, accessed April 9, 2025, https://arxiv.org/abs/2303.09014
hijkzzz/Awesome-LLM-Strawberry: A collection of LLM papers, blogs, and projects, with a focus on OpenAI o1 🍓 and reasoning techniques. - GitHub, accessed April 9, 2025, https://github.com/hijkzzz/Awesome-LLM-Strawberry
MATH Dataset - Papers With Code, accessed April 9, 2025, https://paperswithcode.com/dataset/math
nlile/hendrycks-MATH-benchmark · Datasets at Hugging Face, accessed April 9, 2025, https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark
MR-GSM8K: A META-REASONING BENCHMARK - OpenReview, accessed April 9, 2025, https://openreview.net/pdf/d5d6c38a884aa9905c80b5013c92718069df6130.pdf
GSM8K Dataset - Papers With Code, accessed April 9, 2025, https://paperswithcode.com/dataset/gsm8k
MR-GSM8K: A Meta-Reasoning Revolution in Large Language Model Evaluation - arXiv, accessed April 9, 2025, https://arxiv.org/html/2312.17080v2
Chain-of-Reasoning: unified framework for Mathematical Reasoning in LLMs via a Multi-Paradigm Perspective | by SACHIN KUMAR | Medium, accessed April 9, 2025, https://medium.com/@techsachin/chain-of-reasoning-unified-framework-for-mathematical-reasoning-in-llms-via-a-multi-paradigm-2d2255d4c78e
[D] How does LLM solves new math problems? : r/MachineLearning - Reddit, accessed April 9, 2025, https://www.reddit.com/r/MachineLearning/comments/1ihsftt/d_how_does_llm_solves_new_math_problems/
Lean (proof assistant) - Wikipedia, accessed April 9, 2025, https://en.wikipedia.org/wiki/Lean_(proof_assistant)
Lean - Microsoft Research, accessed April 9, 2025, https://www.microsoft.com/en-us/research/project/lean/
Programming Language and Theorem Prover — Lean, accessed April 9, 2025, https://lean-lang.org/
Generating Millions Of Lean Theorems With Proofs By Exploring State Transition Graphs, accessed April 9, 2025, https://arxiv.org/html/2503.04772v1
Beyond Limited Data: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving - arXiv, accessed April 9, 2025, https://arxiv.org/html/2502.00212v1
arxiv.org, accessed April 9, 2025, https://arxiv.org/pdf/2503.04772
Evaluating Mathematical Reasoning in LLMs - Toloka, accessed April 9, 2025, https://toloka.ai/events/advancing-mathematical-reasoning-in-llms
miniF2F-test Benchmark (Automated Theorem Proving) - Papers With Code, accessed April 9, 2025, https://paperswithcode.com/sota/automated-theorem-proving-on-minif2f-test
MiniF2F Dataset - Papers With Code, accessed April 9, 2025, https://paperswithcode.com/dataset/minif2f
Improving Rule-based Reasoning in LLMs via Neurosymbolic Representations - arXiv, accessed April 9, 2025, https://arxiv.org/html/2502.01657v1
Improving Rule-based Reasoning in LLMs via Neurosymbolic Representations - arXiv, accessed April 9, 2025, https://www.arxiv.org/abs/2502.01657
arxiv.org, accessed April 9, 2025, https://arxiv.org/pdf/2502.01657
CRANE: Reasoning with constrained LLM generation - arXiv, accessed April 9, 2025, https://arxiv.org/html/2502.09061v2
From Equations to Insights: Unraveling Symbolic Structures in PDEs with LLMs - arXiv, accessed April 9, 2025, https://arxiv.org/html/2503.09986v1
Symbolic Mixture-of-Experts: Adaptive Skill-based Routing for Heterogeneous Reasoning, accessed April 9, 2025, https://arxiv.org/html/2503.05641
MathVista: Evaluating Math Reasoning in Visual Contexts, accessed April 9, 2025, https://mathvista.github.io/
Measuring Multimodal Mathematical Reasoning with the MATH-Vision Dataset, accessed April 9, 2025, https://proceedings.neurips.cc/paper_files/paper/2024/file/ad0edc7d5fa1a783f063646968b7315b-Paper-Datasets_and_Benchmarks_Track.pdf
Daily Papers - Hugging Face, accessed April 9, 2025, https://huggingface.co/papers?q=ScienceQA
ScienceQA Dataset - Papers With Code, accessed April 9, 2025, https://paperswithcode.com/dataset/scienceqa
SciBench: Evaluating College-Level Scientific Problem-Solving Abilities of Large Language Models - GitHub, accessed April 9, 2025, https://raw.githubusercontent.com/mlresearch/v235/main/assets/wang24z/wang24z.pdf
README.md · Zhiqiang007/MathV360K at main - Hugging Face, accessed April 9, 2025, https://huggingface.co/datasets/Zhiqiang007/MathV360K/blame/main/README.md
Mathematical Reasoning | Papers With Code, accessed April 9, 2025, https://paperswithcode.com/task/mathematical-reasoning?page=18&q=
deepseek-ai/DeepSeek-Math: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models - GitHub, accessed April 9, 2025, https://github.com/deepseek-ai/DeepSeek-Math
Updated model card for Qwen2 by Aravind-11 · Pull Request #37192 - GitHub, accessed April 9, 2025, https://github.com/huggingface/transformers/pull/37192
InternLM/model_cards/internlm2.5_20b.md at main - GitHub, accessed April 9, 2025, https://github.com/InternLM/InternLM/blob/main/model_cards/internlm2.5_20b.md
internlm/internlm2-math-base-7b - Hugging Face, accessed April 9, 2025, https://huggingface.co/internlm/internlm2-math-base-7b
Large Language Models and Mathematical Reasoning Failures - arXiv, accessed April 9, 2025, https://arxiv.org/html/2502.11574v1
MMLU Pro Benchmark - Vals AI, accessed April 9, 2025, https://www.vals.ai/benchmarks/mmlu_pro-04-04-2025
GPQA Benchmark - Vals AI, accessed April 9, 2025, https://www.vals.ai/benchmarks/gpqa-04-04-2025
GPQA: A Graduate-Level Google-Proof Q&A Benchmark - Klu.ai, accessed April 9, 2025, https://klu.ai/glossary/gpqa-eval
GPQA benchmark 2025 - Metaculus, accessed April 9, 2025, https://www.metaculus.com/questions/21920/gpqa-benchmark-2025/
huangting4201/HT-InternLM - GitHub, accessed April 9, 2025, https://github.com/huangting4201/HT-InternLM
MMLU-Pro Leaderboard - a Hugging Face Space by TIGER-Lab, accessed April 9, 2025, https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro
MathEval: A Comprehensive Benchmark for Evaluating Large Language Models on Mathematical Reasoning Capabilities | OpenReview, accessed April 9, 2025, https://openreview.net/forum?id=DexGnh0EcB
SciBench: Evaluating Math Reasoning in Visual Contexts, accessed April 9, 2025, https://scibench-ucla.github.io/
MMLU-Pro Dataset | Papers With Code, accessed April 9, 2025, https://paperswithcode.com/dataset/mmlu-pro
MathEval Dataset - Papers With Code, accessed April 9, 2025, https://paperswithcode.com/dataset/matheval
