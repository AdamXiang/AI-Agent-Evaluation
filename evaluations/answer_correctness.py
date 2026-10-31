"""
Answer Correctness evaluation for the refund agent (single-turn, GEval).

Runs every golden through the agent, then asks an LLM judge whether the
agent's final answer reaches the same conclusion as the golden's
expected_output (e.g. "refund of $23 issued" vs "no refund, outside 30 days").
Wording may differ; only the logical conclusion is compared.

Each run starts from a clean state with "today" frozen to FROZEN_TODAY.

Run from the project root:
    python evaluations/answer_correctness.py
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
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
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

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=actual_output,
        expected_output=golden.expected_output,
    )

    dataset.add_test_case(test_case)

metrics = [
    GEval(
        name="Answer Correctness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        # Compare conclusions, not wording: phrasing varies between runs.
        evaluation_steps=[
            "make sure that the actual output and the expected output have the same logical conclusion"
        ],
        threshold=0.8,
        model=eval_model,
        verbose_mode=True,  # Print the judge's step-by-step reasoning
    )
]

evaluate(test_cases=dataset.test_cases, metrics=metrics)
