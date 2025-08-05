# Evaluation Framework Documentation

This document outlines the current methodology used for evaluating multilingual language models in this project. We may update the methodology in the future. The main objective was to have something that is unified and comparable and straightforward to build upon. The framework is designed to be fair, consistent, and robust, providing a standardized way to measure model performance across a diverse set of languages and tasks.

## Current Approach: English Zero-Shot Prompting

The current working base of our evaluation methodology is a **unified English zero-shot prompting strategy**. This means:

1.  **Instructions are in English**: All models receive their instructions in clear, standardized English. This removes the quality of prompt translation as a variable, ensuring a fair comparison.
2.  **Content is in the Target Language**: The actual content to be evaluated (e.g., a question for a QA task, a sentence for translation) is always presented in the target language. This directly tests the model's ability to understand instructions in one language and apply them to content in another.
3.  **Zero-Shot (with a Twist)**: We do not provide in-context examples from the test datasets. However, for Question Answering tasks, we provide a static, English-based "scratchpad" example. This doesn't teach the model the answer, but rather the *format* for its reasoning and final output, which is crucial for reliable response parsing.

---

## Task-Specific Prompting Strategies

Below is a breakdown of the prompt structure for each of the active evaluation tasks.

### 1. Translation (`translation`)

-   **Objective**: To evaluate the model's ability to translate text both to and from a target language.
-   **Prompt Structure**: A direct, zero-shot English instruction.
    ```
    Translate the following text to the {target_language_name} language; use the {script} script; reply only with the translation:

    {original_sentence}
    ```

### 2. Classification (`classification`)

-   **Objective**: To evaluate the model's ability to classify a paragraph of text into one of five topics.
-   **Prompt Structure**: A direct, zero-shot English instruction providing the available topics.
    ```
    Classify the following text into one of these topics: {topic1}, {topic2}, {topic3}, {topic4}, {topic5}.
    Reply with only the topic name.

    Text:
    {paragraph_in_target_language}
    ```

### 3. Question Answering (`mmlu`, `arc`, `truthfulqa`)

-   **Objective**: To evaluate the model's knowledge and reasoning abilities on multiple-choice questions.
-   **Prompt Structure**: A zero-shot English instruction combined with a "reasoning scratchpad" format.
    ```
    Solve the following multiple choice question. Reason step-by-step and then write the final answer as a single letter.

    Response format: <reasoning> #### <letter>

    ---

    {question_and_choices_in_target_language}
    ```

### 4. Math Word Problems (`mgsm`)

-   **Objective**: To evaluate the model's ability to solve mathematical reasoning problems.
-   **Prompt Structure**: Similar to the QA tasks, this uses a zero-shot English instruction with a reasoning scratchpad, but asks for a number as the final answer.
    ```
    Solve the following math problem. Reason step-by-step and then write the final answer as a number.

    Response format: <reasoning> #### <number>

    ---

    {math_problem_in_target_language}
    ```

---

## Advantages and Disadvantages of this Methodology

### Advantages

-   **Fairness and Control**: By using standardized English prompts, we eliminate the quality of prompt translation as a confounding variable, leading to a fairer comparison between models.
-   **Robustness**: This approach directly tests a model's cross-lingual instruction-following capabilities, which is a key measure of its multilingual prowess.
-   **Simplicity and Maintainability**: The zero-shot approach significantly simplifies the codebase, making it easier to maintain and extend.

### Disadvantages

-   **Brittleness of Response Parsing**: The evaluation of QA and Math tasks is highly dependent on the model's ability to perfectly adhere to the `#### <answer>` format. Models that produce correct reasoning but fail to follow the format will be unfairly penalized.
-   **Potential for Cross-Lingual Confusion**: Less capable models may struggle with instructions in one language and content in another, which could impact their performance. 