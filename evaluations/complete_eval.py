"""
Complete evaluation: all seven metrics in one run.

Each golden is run through the agent ONCE, and that single response is scored
by two groups of metrics:

- Trace metrics (scored on the trace captured by the CallbackHandler):
  TaskCompletion, StepEfficiency, PlanQuality, PlanAdherence.
- Single-turn metrics (scored on an LLMTestCase built from the same response):
  ArgumentCorrectness, ToolCorrectness (names AND arguments), and GEval
  Answer Correctness.

Reusing one response per golden halves the number of agent calls compared with
running the individual scripts, and guarantees every metric judges the same run.

Run from the project root:
    python evaluations/complete_eval.py
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

from deepeval import evaluate
from deepeval.evaluate import CacheConfig, DisplayConfig
from deepeval.integrations.langchain import CallbackHandler
from deepeval.metrics import (
    ArgumentCorrectnessMetric,
    GEval,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall, ToolCallParams
from dotenv import load_dotenv

# Support both `python -m evaluations.complete_eval` (package) and
# `python evaluations/complete_eval.py` (script).
if __package__:
    from .dataset import dataset
    from .model import eval_model
else:
    from dataset import dataset
    from model import eval_model


# Criteria the LLM judge uses for TaskCompletionMetric. Same text as in
# task_completion.py; keep the two in sync.
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


def build_metrics():
    """
    Build the metrics, using the same thresholds and judge as the individual scripts.

    Returns:
        tuple: (trace_metrics, single_turn_metrics)
            trace_metrics: scored on the trace via dataset.evals_iterator().
            single_turn_metrics: scored on LLMTestCases via evaluate().
    """
    trace_metrics = [
        TaskCompletionMetric(
            threshold=0.8,
            model=eval_model,
            task=SUCCESSFUL_TASK_DESCRIPTION,
            include_reason=True,
        ),
        StepEfficiencyMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
        ),
        PlanQualityMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
        ),
        PlanAdherenceMetric(
            threshold=0.8,
            model=eval_model,
            include_reason=True,
        ),
    ]
    single_turn_metrics = [
        ArgumentCorrectnessMetric(
            threshold=None,  # Library default threshold
            model=eval_model,
            include_reason=True,
        ),
        ToolCorrectnessMetric(
            threshold=0.9,
            model=eval_model,
            include_reason=True,
            # Compare arguments too, not just tool names.
            evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        ),
        GEval(
            name="Answer Correctness",
            evaluation_params=[
                SingleTurnParams.INPUT,
                SingleTurnParams.ACTUAL_OUTPUT,
                SingleTurnParams.EXPECTED_OUTPUT,
            ],
            evaluation_steps=[
                "Check that the actual output and expected output have the same "
                "logical conclusion about the user's request.",
            ],
            threshold=0.8,
            model=eval_model,
            verbose_mode=True,
        ),
    ]
    return trace_metrics, single_turn_metrics


def main():
    """Run every golden once and score it with all seven metrics."""
    load_dotenv()
    # The agent folder uses flat imports, so it must be on sys.path. Importing
    # here (not at module top) keeps `import complete_eval` free of side effects.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
    from tools import reset_issued_refunds

    from agent import agent  # agent/agent.py, not the agent/ folder

    trace_metrics, single_turn_metrics = build_metrics()
    # Disable DeepEval's result cache so every run re-scores fresh agent output.
    cache_config = CacheConfig(use_cache=False, write_cache=False)
    display_config = DisplayConfig(inspect_after_run=False)
    test_cases = []

    # The callback captures the trace required by the four agentic metrics.
    for golden in dataset.evals_iterator(
        metrics=trace_metrics,
        cache_config=cache_config,
        display_config=display_config,
    ):
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

        # Collect every tool call the agent made, in order.
        # NOTE: includes the structured-output pseudo-tool "Answer"; see README.
        tools_called = []
        for message in response["messages"]:
            for tool_call in getattr(message, "tool_calls", []) or []:
                tools_called.append(
                    ToolCall(
                        name=tool_call["name"],
                        input_parameters=tool_call["args"],
                    )
                )

        # Reuse the same response for the three remaining metrics. Keep these
        # cases local so repeated runs do not accumulate dataset.test_cases.
        test_cases.append(
            LLMTestCase(
                input=golden.input,
                actual_output=response["structured_response"].answer,
                expected_output=golden.expected_output,
                tools_called=tools_called,
                expected_tools=golden.expected_tools,
            )
        )

    evaluate(
        test_cases=test_cases,
        metrics=single_turn_metrics,
        cache_config=cache_config,
        display_config=display_config,
    )
    print(f"\nCompleted all 7 metrics for {len(test_cases)} test cases.")


if __name__ == "__main__":
    main()
