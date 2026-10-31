"""
Refund business rules and shared helpers used by the agent's tools.

This module is the single source of truth for:
- the refund window (REFUND_WINDOW_DAYS),
- what "today" is (today(), which can be frozen for repeatable evaluations),
- how usernames and item names are matched (normalize()),
- which requested items are eligible and why others are not (evaluate_items()).

Keep these functions free of LangChain/tool decorators so they can also be
imported by the evaluation code to compute expected results, and be unit tested
without calling a model.
"""

import os
from datetime import date, datetime

from data import users_data

# An item can be refunded up to and including this many days after purchase.
REFUND_WINDOW_DAYS = 30


def normalize(value: str) -> str:
    """Normalize usernames and item names so lookups ignore case and whitespace."""
    return value.strip().lower()


def today() -> date:
    """
    Single source of truth for 'today', shared by every tool.

    Set the REFUND_FAKE_TODAY environment variable (YYYY-MM-DD) to freeze the
    date, so evaluation results do not change depending on when they run.
    """
    fake = os.getenv("REFUND_FAKE_TODAY")
    return parse_date(fake) if fake else date.today()


def get_history(user: str) -> dict | None:
    """Return the user's raw purchase history, or None if the user does not exist."""
    return users_data.get(normalize(user)) or None


def parse_date(value: str) -> date:
    """Parse a purchase date. Accepts non-zero-padded dates such as '2026-9-24'."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def evaluate_items(
    history: dict, item_names: list[str], current_date: date
) -> tuple[dict[str, float], dict[str, str]]:
    """
    Split requested items into eligible and rejected.

    An item is eligible only if it is in the customer's history, its purchase
    date is not in the future, and it was bought at most REFUND_WINDOW_DAYS ago.

    Args:
        history (dict): The customer's raw purchase history from get_history().
        item_names (list[str]): Names of the items the customer wants refunded.
        current_date (date): The date to measure the refund window against.

    Returns:
        tuple: (eligible, rejected) where
            eligible: {item_name: price} for items that can be refunded
            rejected: {item_name: reason} for items that cannot, with a
                      customer-readable explanation
    """
    eligible: dict[str, float] = {}
    rejected: dict[str, str] = {}

    # Deduplicate while preserving order, so an item cannot be counted twice.
    for name in dict.fromkeys(normalize(n) for n in item_names):
        info = history.get(name)
        if info is None:
            rejected[name] = "This item was not purchased by this customer."
            continue

        # Negative means the purchase is dated after today, which is invalid data.
        days = (current_date - parse_date(info["date"])).days
        if days < 0:
            rejected[name] = (
                "The purchase date is in the future, so the purchase is invalid."
            )
        elif days > REFUND_WINDOW_DAYS:
            rejected[name] = (
                f"Purchased {days} days ago, which is beyond the "
                f"{REFUND_WINDOW_DAYS}-day refund window."
            )
        else:
            eligible[name] = info["price"]

    return eligible, rejected
