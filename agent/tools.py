"""
LangChain tools the refund agent can call.

Tools registered with the agent (see agent.py):
- get_user_purchase_history: look up a customer's purchases.
- calculate_right_refund: compute the refund for eligible items.
- issue_refund_ticket: issue the refund, with server-side safety checks.

get_today_date is defined but not registered; it is kept for reference.

Design notes:
- Docstrings are part of the prompt: the model reads them to decide when and
  how to call each tool, so they describe return fields precisely.
- Business rules live in refund_rules.py. The tools re-check them, so a wrong
  decision by the model cannot produce a wrong refund.
- Error messages tell the model what to do next ("do not retry"), because the
  model follows them literally.
"""

import logging
import uuid

from dotenv import load_dotenv
from langchain.tools import tool
from objects import PurchaseItem
from refund_rules import evaluate_items, get_history, normalize, parse_date, today

load_dotenv()

# Basic logging setup (ideally configured once, in the application's entry point).
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# Module-level logger, avoids the "info() call on root logger" warning.
logger = logging.getLogger(__name__)

# Returned for unknown users. The system prompt tells the agent how to handle
# this exact value, so keep both in sync if the wording changes.
USER_NOT_FOUND = {"status": "There is no user found."}

# In-memory record of amounts already refunded per user, to block duplicate tickets.
# Replace with a database in production.
_issued_refunds: dict[str, float] = {}


def reset_issued_refunds() -> None:
    """Clear refund history. Not a tool: call it between evaluation runs
    so one test case's refunds do not affect the next."""
    _issued_refunds.clear()


# ---------------------------------------------------------------------------
# Agent tools
# ---------------------------------------------------------------------------


@tool
def get_user_purchase_history(user: str) -> dict:
    """
    Retrieve the purchase history for a specific customer.

    Args:
        user (str): The customer's username. Case and surrounding spaces are ignored.

    Returns:
        dict: If the user exists, a dict keyed by item name. Each item contains:
              - price (number): the amount paid
              - date (str): the purchase date, YYYY-MM-DD
              - days_since_purchase (int): days between the purchase date and today;
                a negative value means the purchase date is in the future
              If the user does not exist or has no purchases, returns
              {"status": "There is no user found."}.
    """
    logger.info("Using get_user_purchase_history tool on user: %s", user)

    history = get_history(user)
    if not history:
        return USER_NOT_FOUND

    # Compute the day count here so the model never does date arithmetic itself.
    current_date = today()
    result = {}
    for item, info in history.items():
        purchased = parse_date(info["date"])
        result[item] = {
            "price": info["price"],
            "date": purchased.isoformat(),
            "days_since_purchase": (current_date - purchased).days,
        }
    return result


@tool
def calculate_right_refund(user: str, items: list[PurchaseItem]) -> dict:
    """
    Calculate the refund amount for the eligible items a customer wants to return.
    Call this only when at least one requested item is eligible. As a safety net,
    items that were not purchased, are older than 30 days, or are dated in the
    future are excluded from the amount and listed in rejected_items.

    Args:
        user (str): The customer's username.
        items (list[PurchaseItem]): The eligible items the customer asked to return.
                                    Do not include ineligible items.

    Returns:
        dict: If the user exists:
              - refund_amount (number): total refund for the eligible items
              - eligible_items (dict): {item_name: price} for items that will be refunded
              - rejected_items (dict): {item_name: reason} for items that cannot be refunded
              If the user does not exist, returns {"status": "There is no user found."}.
    """
    logger.info("Using calculate_right_refund tool for user: %s", user)

    history = get_history(user)
    if not history:
        return USER_NOT_FOUND

    eligible, rejected = evaluate_items(history, [i.name for i in items], today())

    return {
        "refund_amount": sum(eligible.values()),
        "eligible_items": eligible,
        "rejected_items": rejected,
    }


@tool
def issue_refund_ticket(user: str, refund_amount: float) -> str:
    """
    Issue a refund ticket to the processing team.
    Call this only once per request, after calculate_right_refund, using the
    refund_amount it returned. Do not call it when refund_amount is 0.

    Args:
        user (str): The customer's username.
        refund_amount (float): The refund amount returned by calculate_right_refund.

    Returns:
        str: A confirmation with the ticket ID and amount, or an error message
             explaining why the ticket was not issued.
    """
    logger.info(
        "Using issue_refund_ticket for user: %s with amount: %s", user, refund_amount
    )

    key = normalize(user)
    history = get_history(key)
    if not history:
        return "Refund not issued: there is no user found."

    if refund_amount <= 0:
        return "Refund not issued: the refund amount must be greater than 0."

    # Safety net: the total refunded can never exceed what is currently refundable.
    eligible, _ = evaluate_items(history, list(history.keys()), today())
    max_refundable = sum(eligible.values())
    already_refunded = _issued_refunds.get(key, 0)

    if already_refunded + refund_amount > max_refundable:
        logger.warning(
            "Rejected refund for %s: requested %s, already refunded %s, max %s",
            key,
            refund_amount,
            already_refunded,
            max_refundable,
        )
        # Final answer for this request: tell the agent explicitly not to retry,
        # otherwise it loops on calculate_right_refund / issue_refund_ticket.
        if already_refunded > 0:
            reason = (
                "part of this customer's eligible purchases has already been refunded"
            )
        else:
            reason = "the amount exceeds what this customer is eligible to be refunded"
        return (
            f"Refund not issued: {reason}. This is final. Do not retry or recalculate. "
            "Tell the customer the refund could not be issued for this reason."
        )

    _issued_refunds[key] = already_refunded + refund_amount
    # Short random ID, enough to reference the ticket in the customer reply.
    ticket_id = uuid.uuid4().hex[:8].upper()

    return (
        f"Refund ticket {ticket_id} has been issued successfully "
        f"with the amount of {refund_amount}."
    )


@tool
def get_today_date() -> str:
    """
    Get today's date in ISO format (YYYY-MM-DD).

    Takes no arguments. Eligibility is already computed by the other tools,
    so use this only when you need to state today's date to the customer.

    Note: not registered with the agent in agent.py. Exposing it caused
    unnecessary calls, since days_since_purchase already covers the date check.

    Returns:
        str: Today's date in 'YYYY-MM-DD' format.
    """
    logger.info("Using the get_today_date tool.")

    return today().isoformat()
