"""System prompt for the SR2AM agent runner.

Contains:
- AGENT_SYSTEM_PROMPT: Base prompt for run_agent.py
- get_custom_system_prompt(): Appends the answer-format suffix
- get_agent_system_prompt(): Builds the full prompt with current or fixed datetime
"""

from datetime import datetime

AGENT_SYSTEM_PROMPT = """\
{CURRENT_TIME}
You are K2-Researcher, a reasoning assistant with the ability to perform web searches, browse web pages, and execute Python code to help you answer the user's question. \
"""


def get_custom_system_prompt(dataset: str, base_prompt: str) -> str:
    """Build a system prompt by appending the answer format suffix.

    Always uses the same suffix regardless of dataset — the paper uses a single
    answer format instruction for all domains.
    """
    return base_prompt + "\n\nYou should follow the answer format described in the question."


FIXED_DATETIME = "Sun Aug 31 2025 23:34:17"


def get_agent_system_prompt(dataset: str, fix_datetime: bool = False) -> str:
    """Build the agent system prompt with current or fixed datetime.

    By default uses live datetime. Pass fix_datetime=True to use the fixed
    datetime from training data (required for paper reproduction).
    """
    if fix_datetime:
        current_time = FIXED_DATETIME
    else:
        current_time = datetime.now().strftime("%a %b %d %Y %H:%M:%S %z")
    base = AGENT_SYSTEM_PROMPT.format(CURRENT_TIME=f"CURRENT_TIME: {current_time}")
    return get_custom_system_prompt(dataset, base)
