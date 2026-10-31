"""
Plan Adherence evaluation for the refund agent (trace-based).

Runs every golden through the agent and scores each run with DeepEval's
PlanAdherenceMetric. An LLM judge reads the trace captured by the LangChain
CallbackHandler and checks whether the agent's actions followed the plan it
set out (e.g. it did not skip the eligibility check or issue a refund before
calculating it).

Plan Quality asks "was the plan good?"; Plan Adherence asks "was it followed?".

Each run starts from a clean state with "today" frozen to FROZEN_TODAY.

Run from the project root:
    python evaluations/plan_adherence.py
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

from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import PlanAdherenceMetric
from model import eval_model
from tools import reset_issued_refunds

from agent import agent  # agent/agent.py, not the agent/ folder

metrics = [PlanAdherenceMetric(threshold=0.8, model=eval_model, include_reason=True)]


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
