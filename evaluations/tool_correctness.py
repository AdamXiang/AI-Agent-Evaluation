"""
Tool Correctness evaluation for the refund agent (single-turn).

Runs every golden through the agent, collects the tools it called, and compares
them with the golden's expected_tools. By default only tool names are compared;
pass evaluation_params=[ToolCallParams.INPUT_PARAMETERS] to compare arguments too.

Each run starts from a clean state with "today" frozen to FROZEN_TODAY.

Run from the project root:
    python evaluations/tool_correctness.py
"""

import os
import sys
from pathlib import Path
from uuid import uuid4

# The dataset defines the frozen date, so it is imported first.
# It does not depend on the agent, so it can be imported before sys.path changes.
from dataset import FROZEN_TODAY, dataset

# Freeze "today" for every tool, so results do not change with the calendar.
# refund_rules.today() reads this variable each time it is called.
os.environ["REFUND_FAKE_TODAY"] = FROZEN_TODAY

# The agent folder uses flat imports (e.g. `from tools import ...`), so the folder
# itself must be on sys.path, and imports from it must use the same flat style.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))


from deepeval import evaluate
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from dotenv import load_dotenv
from model import eval_model
from tools import reset_issued_refunds

from agent import agent  # agent/agent.py, not the agent/ folder

load_dotenv()

for golden in dataset.goldens:
    # Clear refunds issued by previous goldens, so each test case starts clean.
    reset_issued_refunds()
    response = agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            # A fresh thread_id keeps each golden's conversation isolated.
            "configurable": {"thread_id": str(uuid4())},
            "callbacks": [CallbackHandler()],
        },
    )

    # The agent replies through structured output (the Answer schema).
    actual_output = response["structured_response"].answer

    # Collect every tool call the agent made, in order.
    # NOTE: this includes the structured-output pseudo-tool "Answer", which is
    # not in any expected_tools list. Cases where the agent correctly calls no
    # tools (e.g. asking for a username) fail because of it; see the README.
    tools_called = []
    for message in response["messages"]:
        for tool_call in getattr(message, "tool_calls", []) or []:
            tools_called.append(
                ToolCall(name=tool_call["name"], input_parameters=tool_call["args"])
            )

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        tools_called=tools_called,
        expected_tools=golden.expected_tools,
    )

    dataset.add_test_case(test_case)

metrics = [ToolCorrectnessMetric(threshold=0.8, model=eval_model, include_reason=True)]

evaluate(test_cases=dataset.test_cases, metrics=metrics)
