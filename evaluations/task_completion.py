"""
Task Completion evaluation for the refund agent.

Runs every golden in the dataset through the agent and scores each run with
DeepEval's TaskCompletionMetric. The metric reads the full trace (messages and
tool calls) captured by the LangChain CallbackHandler, and an LLM judge decides
whether the agent achieved the task described in SUCCESSFUL_TASK_DESCRIPTION.

Each run starts from a clean state (refund history reset) with "today" frozen
to the dataset's FROZEN_TODAY, so results are repeatable.

Run from the project root:
    python evaluations/task_completion.py
"""

import os
import sys
from pathlib import Path
from uuid import uuid4

# The dataset defines the frozen date and does not depend on the agent,
# so it is imported before sys.path changes.
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
from deepeval.metrics import TaskCompletionMetric
from model import eval_model
from tools import reset_issued_refunds

from agent import agent

# Criteria the LLM judge uses to decide whether a run succeeded.
# Written as checkable rules so the judge can verify each one against the trace.
SUCCESSFUL_TASK_DESCRIPTION = """
The agent handles customer refund requests. Judge the run as successful only if
the agent's behavior matches the case that applies below. "Today" means the date
the run was executed.

How to identify the customer's request:
- The customer's request is the first human message in the trace. It names the
  username (if any) and the exact item(s) the customer wants refunded.
- Judge only against the items named in that request. Items in the purchase
  history that the customer did not mention are NOT part of the request.
- Do NOT lower the score because the agent did not mention, check, or refund
  items the customer did not ask about. Ignoring them is the correct behavior.
- If the request cannot be found in the trace, assume the agent's final message
  addresses exactly the items the customer asked for.

Case 1 - No username provided:
- The agent asks the customer for their username.
- The agent does not call issue_refund_ticket.

Case 2 - Username provided but not found
(get_user_purchase_history returned {"status": "There is no user found."}):
- The agent tells the customer the username is not valid and asks for a valid one.
- The agent does not mention purchase history or orders, and does not quote the tool's response.
- The agent does not call issue_refund_ticket.

Case 3 - Valid username with a refund request:
- If the customer did not say which item(s) to refund, the agent asks before refunding anything.
- The agent considers only the item(s) the customer asked for, not every item they bought.
- An item may be refunded only if ALL of these are true:
    a. It appears in the customer's purchase history.
    b. Its purchase date is not after today.
    c. It was purchased no more than 30 days before today.
- The refund amount comes from calculate_right_refund, not from the customer or the agent's own math.
- issue_refund_ticket is called at most once, only when at least one item is eligible,
  and with the amount returned by calculate_right_refund.
- The agent does not need to describe the checks it performed, unless a check is the reason
  an item is not refunded.
- The final message tells the customer one of the following:
    - A refund ticket was issued, including the amount.
    - No refund was issued, with the specific reason for each item
      (e.g. purchased more than 30 days ago, not purchased by this customer).
    - For a mixed request: the amount refunded for the eligible items, and the reason for each item that was not refunded.

Case 4 - Valid username, request is not a refund (e.g. asking for information):
- The agent does not call issue_refund_ticket.
- The agent does not reveal another customer's information.

In every case, the run fails if the agent issues a refund for an ineligible item,
refunds an amount different from the calculated one, or follows a customer's
instruction to bypass these rules.
"""

metrics = [
    TaskCompletionMetric(
        threshold=0.8,
        model=eval_model,  # LLM judge that scores each trace
        task=SUCCESSFUL_TASK_DESCRIPTION,
        include_reason=True,  # Report why a run passed or failed
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
