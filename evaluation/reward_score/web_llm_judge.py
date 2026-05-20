import json
import re
from openai import AsyncOpenAI
import os
from typing import Any, Dict, Optional, Tuple
from pydantic import BaseModel
from typing import Literal

# https://github.com/MiroMindAI/MiroThinker/blob/main/apps/miroflow-agent/src/io/output_formatter.py#L19
def _extract_boxed_content(text: str) -> str:
        """
        Extract content from \\boxed{} patterns in the text.
        Uses safe regex patterns to avoid catastrophic backtracking.
        Returns the last matched content, or empty string if no match found.
        """
        if not text:
            return ""

        # Primary pattern: handles single-level brace nesting
        primary_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
        matches = re.findall(primary_pattern, text, re.DOTALL)

        # Fallback pattern: simpler match for any content until first closing brace
        if not matches:
            fallback_pattern = r"\\boxed\{([^}]+)\}"
            matches = re.findall(fallback_pattern, text, re.DOTALL)

        return matches[-1].strip() if matches else ""
   
GAIA_VALIDATION_TEXT_103_SCORER_PROMPT = """You are an evaluation assistant. Please determine if the predicted answer is equivalent to the labeled answer.

Question: {question}

Labeled Answer: {correct_answer}

Predicted Answer: {response}

Did the model give an answer **equivalent** to the labeled answer? Please respond with "Correct" if they are equivalent, or "Incorrect" if they are not equivalent. Do not include any other text.
"""

# https://github.com/MiroMindAI/MiroThinker/blob/main/apps/miroflow-agent/benchmarks/evaluators/eval_utils.py#L367
async def verify_answer_gaia_validation_text_103(
    question: str, target: str, predicted_answer: str
) -> Tuple[bool, str]:
    base_url = os.getenv("WEB_LLM_JUDGE_URL")
    if not base_url:
        raise ValueError("WEB_LLM_JUDGE_URL is not set")
    model = os.getenv("WEB_LLM_JUDGE_MODEL", 'Qwen/Qwen3-32B')
    evaluation_llm_client = AsyncOpenAI(base_url=base_url,
                                        api_key='EMPTY')
    
    
    prompt = GAIA_VALIDATION_TEXT_103_SCORER_PROMPT.format(
        question=question, correct_answer=target, response=predicted_answer
    )

    max_tries = 10
    for attempt in range(max_tries):
        try:
            if model == 'Qwen/Qwen3-32B':
                response = await evaluation_llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=8192,
                    temperature=0.1,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20, 
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            else:
                response = await evaluation_llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    # max_tokens=8192,
                    max_tokens=512,
                    temperature=0.1,
                )
            
            # response = await evaluation_llm_client.chat.completions.create(
            #     model="Qwen/Qwen3-32B",
            #     messages=[{"role": "user", "content": prompt}],
            #     max_tokens=8192
            # )

            content = response.choices[0].message.content
            # content = content.split('</think>')[-1].strip()
            # print("GAIA LLM Judge Response: ", content)
            # print("LLM Judge Response: ", content)

            if response:
                break
        except Exception as e:
            if attempt == (max_tries - 1):
                raise e
            
    # return content == "Correct", content
    return content.lower() == "correct", content

# https://github.com/MiroMindAI/MiroThinker/blob/f658ef3fea98a93829e77be3018c469945347fa9/apps/miroflow-agent/benchmarks/evaluators/eval_utils.py#L363
async def verify_answer_gaia_validation_text_103_gpt(
    question: str, target: str, predicted_answer: str
) -> Tuple[bool, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    evaluation_llm_client = AsyncOpenAI(api_key=api_key)

    prompt = GAIA_VALIDATION_TEXT_103_SCORER_PROMPT.format(
        question=question, correct_answer=target, response=predicted_answer
    )

    max_tries = 10
    for attempt in range(max_tries):
        try:
            response = await evaluation_llm_client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content
            # print("LLM Judge Response: ", content)

            if response:
                break
        except Exception as e:
            if attempt == (max_tries - 1):
                raise e

    if content == "Correct":
        return True, "CORRECT"
    else:
        return False, "INCORRECT"

# https://github.com/MiroMindAI/MiroThinker/blob/main/apps/miroflow-agent/benchmarks/evaluators/eval_utils.py#L39
EVALUATION_PROMPT_SIMPLEQA = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k". 
    - Predicted answers "120k", "124k", and 115k" are all CORRECT. 
    - Predicted answers "100k" and "113k" are INCORRECT. 
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name. 
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {}
Gold target: {}
Predicted answer: {}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()


async def verify_answer_llm_simpleqa(
    question: str, target: str, predicted_answer: str
) -> str:
    """
    Use LLM to verify if the predicted answer is correct.
    Expects the LLM to choose between A (correct), B or C (incorrect).
    """
    
    base_url = os.getenv("WEB_LLM_JUDGE_URL")
    if not base_url:
        raise ValueError("WEB_LLM_JUDGE_URL is not set")
    model = os.getenv("WEB_LLM_JUDGE_MODEL", 'Qwen/Qwen3-32B')

    evaluation_llm_client = AsyncOpenAI(base_url=base_url,
                                        api_key='EMPTY')
    
    messages = [
        {
            "role": "user",
            "content": EVALUATION_PROMPT_SIMPLEQA.format(
                question, target, predicted_answer
            ),
        }
    ]
    CHOICE_MAP = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}

    for i in range(3):
        try:
            if model == 'Qwen/Qwen3-32B':
                llm_response = await evaluation_llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=8192,
                    temperature=0.1,
                    top_p=0.8,
                    presence_penalty=1.5,
                    extra_body={
                        "top_k": 20, 
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            else:
                llm_response = await evaluation_llm_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    # max_tokens=8192,
                    max_tokens=512,
                    temperature=0.1,
                )
            
            content = llm_response.choices[0].message.content
            
            # content = content.split('</think>')[-1].strip()
            # print("SimpleQA LLM Judge Response: ", content)
            match = re.search(r"(A|B|C)", content)
            if match:
                choice = CHOICE_MAP[match.group(0)]
                return choice == "CORRECT", content
        except Exception as e:
            print(f"LLM evaluation failed: {e}")

    return False, "NOT_ATTEMPTED"

# ================================================
# verify_answer_browsecomp

# Prompt from WebAgent
# https://github.com/Alibaba-NLP/WebAgent/blob/f25dae54daf0ce2874ffd5ed5ffb20feca7c4c4e/WebSailor/src/prompt.py
# ================================================

JUDGE_PROMPT_BC = """"Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}
[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as ’None’ if there is no exact, final answer to extract from the response.
[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer ’yes’ if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer ’no’ otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect. 

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there
is no confidence score available."""

async def verify_answer_browsecomp(
    question: str, target: str, predicted_answer: str
) -> Tuple[bool, str]:
    """
    Use BrowseComp judge to verify if the predicted answer is correct.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    evaluation_llm_client = AsyncOpenAI(api_key=api_key)

    prompt = JUDGE_PROMPT_BC.format(
        question=question, correct_answer=target, response=predicted_answer
    )

    try:
        response = await evaluation_llm_client.beta.chat.completions.parse(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.choices[0].message.content

        match = re.search(r"correct\s*:\s*(yes|no)", content, re.IGNORECASE)
        if match:
            judge = match.group(1).lower()
        else:
            judge = None

        if judge == "yes":
            return True, "CORRECT"
        else:
            return False, "INCORRECT"

    except Exception as e:
        print(f"BrowseComp evaluation failed: {e}")
        return False, "NOT_ATTEMPTED"


# ================================================
# verify_answer_hle
# ================================================

HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""

class HLEExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] = True  # 100% reliability


async def verify_answer_hle(question: str, target: str, predicted_answer: str) -> Tuple[bool, str]:
    """
    Use HLE-style LLM judge to verify if the predicted answer is correct.
    Returns the evaluation result as a string: "CORRECT", "INCORRECT", or "NOT_ATTEMPTED".

    Args:
        question: The question being answered
        target: The correct/target answer
        predicted_answer: The model's predicted answer

    Returns:
        String indicating the evaluation result
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    evaluation_llm_client = AsyncOpenAI(api_key=api_key)

    prompt = HLE_JUDGE_PROMPT.format(
        question=question, correct_answer=target, response=predicted_answer
    )

    try:
        response = await evaluation_llm_client.beta.chat.completions.parse(
            model="o4-mini-2025-04-16",
            max_completion_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            response_format=HLEExtractedAnswer,
        )

        content = response.choices[0].message.parsed

        # Print HLE reasoning
        # print(f"LLM as Judge Reasoning: {content.reasoning}")
        # print(f"LLM as Judge Result: {content.correct}")
        # print(f"LLM as Judge Confidence: {content.confidence}%")

        # Convert HLE format to eval_utils format
        if content.correct == "yes":
            return True, "CORRECT"
        else:
            return False, "INCORRECT"

    except Exception as e:
        if "Incorrect API key provided" in str(e):
            print(f"LLM evaluation failed: {e}")
            exit()
        print(f"LLM evaluation failed: {e}")
        return False, "NOT_ATTEMPTED"

# ================================================
# verify_answer_xbench_deepsearch

# Prompt from XBench-Evals
# https://github.com/xbench-ai/xbench-evals/blob/main/eval_grader.py#L25
# ================================================

JUDGE_PROMPT_XBENCH = """
你是一个通用人工智能助手。根据下面给出的[正确答案], 判断以下对[原问题]的[回答]的回答是否正确。

[原问题]: {question}

[正确答案]: {correct_answer}

[回答]:{response}

你的判断必须按照以下格式和标准进行:

最终答案: 从[回答]中提取出的最终准确答案。如果[回答]中没有明确的最终答案, 则填写'无'。

解释: 根据[正确答案]解释为什么[最终答案]是正确的或错误的。只关注[最终答案]与[正确答案]之间是否存在实质性差异, 不要评论题目的背景, 不要尝试重新解题, 不要为任何不同于[正确答案]的答案辩护, 只专注于判断答案是否一致。

结论: 如果[最终答案]与上方给出的[正确答案]一致, 或者在数值题目中处于可接受的微小误差范围内, 则填写'正确'; 否则（即存在任何不一致、歧义、不等价或提取出的答案错误的情况）填写'错误'。
""".strip()


async def verify_answer_xbench_deepsearch(
    question: str, target: str, predicted_answer: str
) -> Tuple[bool, str]:
    """
    Use XBench-DeepSearch judge to verify if the predicted answer is correct.
    """

    def parse_match_result(match):
        if match is None:
            return match
        match = match.group(0)
        try:
            target = match.split(":")[1].strip()
            return target
        except Exception:
            return match  # return naive result in case of failed

    if predicted_answer is None:
        return False, "INCORRECT"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    evaluation_llm_client = AsyncOpenAI(api_key=api_key)

    judge_prompt = JUDGE_PROMPT_XBENCH.format(
        question=question,
        correct_answer=target,
        response=predicted_answer,
    )
    try:
        response = await evaluation_llm_client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": judge_prompt}],
        )
        judge_response = response.choices[0].message.content
    except Exception:
        judge_response = None
    if judge_response is None:
        return False, "NOT_ATTEMPTED"

    # Extract grader conclusions
    extract_match = re.search(r"最终答案:*(.*)", judge_response)
    extract_match = parse_match_result(extract_match)

    correct_match = re.search(r"结论:*\s*(正确|错误)", judge_response)
    correct_match = parse_match_result(correct_match)

    explain_match = re.search(r"解释:*(.*)", judge_response)
    explain_match = parse_match_result(explain_match)

    # Print debug info
    print(f"XBench Judge - Extract: {extract_match}, Correct: {correct_match}")

    if correct_match == "正确":
        return True, "CORRECT"
    elif correct_match == "错误":
        return False, "INCORRECT"
    else:
        # If we can't parse the result, return NOT_ATTEMPTED to trigger retry
        print(
            f"Warning: Could not parse XBench judge response, correct_match={correct_match}"
        )
        return False, "NOT_ATTEMPTED"


# ================================================
# verify_answer_deepsearchqa
#
# Official prompt from DeepSearchQA benchmark
# https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code
# ================================================

JUDGE_PROMPT_DEEPSEARCHQA = """Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


async def verify_answer_deepsearchqa(
    question: str, target: str, predicted_answer: str, extra_info: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """
    Use DeepSearchQA-specific judge to verify if the predicted answer is correct.
    Uses the official DeepSearchQA evaluation prompt with JSON output format.

    Args:
        question: The question being answered
        target: The correct/target answer
        predicted_answer: The model's predicted answer
        extra_info: Optional dict with 'answer_type' key (e.g. "Single Answer" or "Set Answer")

    Returns:
        Tuple of (is_correct: bool, content: str)
    """
    if predicted_answer is None:
        return False, "INCORRECT"

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    evaluation_llm_client = AsyncOpenAI(api_key=api_key)

    # Determine prompt_type from extra_info
    prompt_type = "Single Answer"  # Default
    if extra_info and "answer_type" in extra_info:
        answer_type = extra_info["answer_type"]
        if answer_type == "Set Answer":
            prompt_type = "Set Answer"

    judge_prompt = JUDGE_PROMPT_DEEPSEARCHQA.format(
        prompt_type=prompt_type,
        prompt=question,
        answer=target,
        response=predicted_answer,
    )

    try:
        response = await evaluation_llm_client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=[{"role": "user", "content": judge_prompt}],
        )
        judge_response = response.choices[0].message.content
    except Exception as e:
        print(f"DeepSearchQA judge failed: {e}")
        return False, "NOT_ATTEMPTED"

    if judge_response is None:
        return False, "NOT_ATTEMPTED"

    # Parse JSON response
    try:
        # Extract JSON from the response (might be wrapped in markdown code blocks)
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", judge_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON without code blocks
            json_match = re.search(r"\{.*\}", judge_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                print("Warning: Could not find JSON in DeepSearchQA judge response")
                return False, "NOT_ATTEMPTED"

        result = json.loads(json_str)
        answer_correctness = result.get("Answer Correctness", {})

        explanation = answer_correctness.get("Explanation", "")
        correctness_details = answer_correctness.get("Correctness Details", {})
        excessive_answers = answer_correctness.get("Excessive Answers", [])

        # Calculate statistics
        num_expected = len(correctness_details)
        num_correct = sum(1 for v in correctness_details.values() if v)
        num_excessive = len(excessive_answers)

        # Print debug info
        print(
            f"DeepSearchQA Judge - Correct: {num_correct}/{num_expected}, Excessive: {num_excessive}"
        )
        print(f"DeepSearchQA Judge - Explanation: {explanation}")

        # Determine if answer is correct
        if correctness_details:
            all_correct = all(correctness_details.values())
            if all_correct and not excessive_answers:
                return True, "CORRECT"
            else:
                return False, "INCORRECT"
        else:
            return False, "NOT_ATTEMPTED"

    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON from DeepSearchQA judge: {e}")
        print(f"Response: {judge_response[:200]}...")
        return False, "NOT_ATTEMPTED"
    except Exception as e:
        print(f"Warning: Error processing DeepSearchQA judge response: {e}")
        return False, "NOT_ATTEMPTED"


scorers = {
    "gaia": verify_answer_gaia_validation_text_103,
    "gaia_gpt": verify_answer_gaia_validation_text_103_gpt,
    "simpleqa": verify_answer_llm_simpleqa,
    "browsecomp": verify_answer_browsecomp,
    "hle": verify_answer_hle,
    "xbench_deepsearch": verify_answer_xbench_deepsearch,
    "deepsearchqa": verify_answer_deepsearchqa,
}

# ------------ Public API -----------------------------------------------------
async def compute_score(data_source: str,
                  model_output: str,
                  ground_truth: str,
                  extra_info: dict, 
                  scorer_type: str) -> Tuple[bool, float, str]:
    """
    Arguments
    ---------
    model_output : str   – agent's raw answer
    ground_truth : str   – reference answer
    extra_info   : dict  – MUST contain key "question"

    Returns
    -------
    (is_correct, score, normalized_student_answer)
        score is 1.0 if correct, else 0.0
    """
    model_output = str(model_output)
    ground_truth = str(ground_truth)
    boxed_result = _extract_boxed_content(model_output)
    question    = extra_info["question"]
    if boxed_result == "":
        return {"score": 0.0, "error": "No boxed result found"}
    else:
        try:
            if scorer_type == 'deepsearchqa':
                is_correct, content = await scorers[scorer_type](question, ground_truth, boxed_result, extra_info=extra_info)
            else:
                is_correct, content = await scorers[scorer_type](question, ground_truth, boxed_result)
        except Exception as e:
            print(f"[judge-error] {e}")
            return {"score": 0.0, "error": "LLM Judge Error"}
        return {"score": float(is_correct), "content": content}