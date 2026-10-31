"""
Step Efficiency evaluation for the refund agent.

Runs every golden in the dataset through the agent and scores each run with
DeepEval's StepEfficiencyMetric. An LLM judge reads the full trace captured by
the LangChain CallbackHandler and checks whether the agent reached its answer
without redundant steps (repeated tool calls, unnecessary tools, retries).

This metric does not use the goldens' expected_tools. To check which tools were
called and with what arguments, use ToolCorrectnessMetric instead.

Run from the project root:
    python evaluations/step_efficiency.py
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
# itself must be on sys.path. Every import below from that folder must use the
# same flat style: importing a module through a different path (e.g. `agent.tools`)
# creates a second copy with its own state, and resetting it would have no effect.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import StepEfficiencyMetric
from model import eval_model
from tools import reset_issued_refunds

from agent import agent

metrics = [
    StepEfficiencyMetric(
        threshold=0.8,
        model=eval_model,  # LLM judge that scores each trace
        include_reason=True,  # Report which steps were judged redundant
    )
]


# evals_iterator yields each golden and scores the trace captured during the loop body.
for golden in dataset.evals_iterator(metrics=metrics):
    # Clear refunds issued by previous goldens, so each test case starts clean.
    reset_issued_refunds()

    agent.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={
            # A fresh thread_id keeps each golden's conversation isolated.
            "configurable": {"thread_id": str(uuid4())},
            # Captures messages and tool calls so the metric can evaluate the trace.
            "callbacks": [CallbackHandler()],
        },
    )
