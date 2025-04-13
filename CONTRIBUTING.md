# Contributing to Awesome-Math-LLM

First off, thank you for considering contributing! Your help is essential for keeping this list up-to-date and comprehensive. We aim to curate high-quality resources dedicated to Large Language Models (LLMs) for mathematics.

## How Can I Contribute?

There are several ways you can contribute:

* **Adding new resources:** Found a relevant paper, model, dataset, tool, or benchmark we missed? Please add it!
* **Correcting existing entries:** Notice a broken link, incorrect title, missing date, or wrong information? Help us fix it!
* **Improving organization:** Have suggestions for better sectioning or categorization? Open an issue to discuss!

## Getting Started

* **Check for Duplicates:** Before adding a resource, please use the search function (`Ctrl+F` or `Cmd+F`) to ensure it hasn't already been listed.
* **Ensure Relevance:** Submissions should be directly related to the intersection of Large Language Models and mathematics (reasoning, problem-solving, theorem proving, foundational capabilities, etc.).

## Adding Resources (Preferred Method: Pull Requests)

1.  **Fork the Repository:** Click the 'Fork' button at the top right of the main repository page.
2.  **Clone your Fork:** Clone your forked repository to your local machine.
    ```bash
    git clone [https://github.com/YOUR_USERNAME/Awesome-Math-LLM.git](https://github.com/YOUR_USERNAME/Awesome-Math-LLM.git)
    cd Awesome-Math-LLM
    ```
3.  **Create a New Branch:** Create a descriptive branch name for your changes.
    ```bash
    git checkout -b add-resource-xyz
    ```
4.  **Add Your Contribution:**
    * Find the most relevant subsection in the `README.md` file.
    * Add your resource entry, strictly adhering to the **Formatting Guidelines** below.
    * Ensure the list within the subsection remains sorted in **reverse chronological order** (newest entries first).
5.  **Commit Your Changes:**
    ```bash
    git add README.md
    git commit -m "Add: [Resource Title] to [Section Name]"
    ```
    (Use a concise commit message describing your addition.)
6.  **Push to Your Fork:**
    ```bash
    git push origin add-resource-xyz
    ```
7.  **Open a Pull Request (PR):** Go back to your fork on GitHub and click the 'New pull request' button. Provide a clear description of the resource you added and why it's relevant. If your PR addresses an existing issue, link it (e.g., `Closes #123`).

## Formatting Guidelines

Consistency is key! Please format your entries exactly as follows:

* **Format:**
    ```markdown
    * **[Optional Prefix]:** "[Resource Title]" ([Primary Link]) ([Secondary Link]) - *(Month Year)*
    ```
* **Components:**
    * `*`: Start with a bullet point.
    * `**[Optional Prefix]:**`: Use a **bolded** prefix *only* if it identifies a well-known named model, dataset, technique, or concept for added clarity (e.g., `**Llemma:**`, `**Chain-of-Thought:**`, `**MATH Benchmark:**`). Omit this if the title alone is sufficient.
    * `"[Resource Title]"`: The official, accurate title of the paper, report, blog post, dataset name, etc., enclosed in double quotes. **This is required.**
    * `([Primary Link])`: The main canonical link (e.g., `[Paper](URL_to_arXiv_abstract)`, `[Blog Post](URL)`, `[Website](URL)`), enclosed in parentheses. Prefer official abstract/landing pages over direct PDFs.
    * `([Secondary Link])`: An *optional* second link for code repositories, datasets, models, etc. (e.g., `([GitHub](URL))`, `([HF Dataset](URL))`, `([HF Models](URL))`), also enclosed in parentheses.
    * `- *(Month Year)*`: The first publication or release date (Month spelled out), enclosed in italicized parentheses (e.g., `- *(October 2023)*`, `- *(May 2024)*`). If the month is unknown, use `*(Year)*`.
* **Example:**
    ```markdown
    * **Llemma:** "Llemma: An Open Language Model For Mathematics" ([Paper](https://arxiv.org/abs/2310.10631)) ([HF Models](https://huggingface.co/EleutherAI/llemma_7b)) - *(October 2023)*
    ```
* **Ordering:** Place your entry in the correct subsection based on its content, maintaining reverse chronological order (newest first) based on the `*(Month Year)*`.

## Opening Issues

Feel free to open an issue for:

* Suggesting new sections or categories.
* Discussing significant changes or restructuring.
* Reporting broken links or errors you can't fix yourself.
* Asking questions about contributions.

## Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms. Please be respectful and constructive in all interactions. (We recommend creating a `CODE_OF_CONDUCT.md` file, possibly using the Contributor Covenant template: [https://www.contributor-covenant.org/](https://www.contributor-covenant.org/))

---

Thank you again for your contributions!