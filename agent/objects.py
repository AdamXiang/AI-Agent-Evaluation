"""
Pydantic schemas shared by the agent and its tools.

- Answer: the structured format of every final reply the agent sends.
- PurchaseItem: an item passed to calculate_right_refund.

Field descriptions are sent to the model, so they are written as instructions.
"""

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """The agent's final reply to the customer, used as the response_format."""

    answer: str = Field(description="The answer that is needed for the user question.")
    # Lets the caller know whether to wait for another customer message.
    # Once a refund decision has been made, there is nothing left to discuss.
    conversation: bool = Field(
        description=(
            "This is an indicator to know if the conversation should still continue "
            "or not. If the agent has made the decision about the refund this should "
            "be False, or if it needs more information then True."
        ),
        default=True,
    )


class PurchaseItem(BaseModel):
    """An item the customer wants refunded, identified by its name."""

    name: str = Field(description="Name of the item has been purchased")
