"""
Refund agent definition and an interactive console demo.

The agent is a LangChain `create_agent` graph that combines:
- the Claude chat model from models.py,
- the system prompt from prompts.py,
- three refund tools from tools.py,
- a structured response schema (`Answer`) from objects.py,
- an in-memory checkpointer, so a conversation keeps its history per thread_id.

The evaluation scripts import `agent` from this module. Run this file directly
to chat with the agent in the terminal:

    python agent/agent.py
"""

from uuid import uuid4

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from models import model
from objects import Answer
from prompts import agent_system_prompt
from tools import (
    calculate_right_refund,
    get_user_purchase_history,
    issue_refund_ticket,
)

# Load ANTHROPIC_API_KEY from .env before the model is used.
load_dotenv()

agent = create_agent(
    model=model,
    system_prompt=agent_system_prompt,
    # get_today_date is intentionally not registered: get_user_purchase_history
    # already returns days_since_purchase, and exposing an unneeded tool made the
    # agent call it anyway (flagged as a redundant step by StepEfficiencyMetric).
    tools=[
        calculate_right_refund,
        get_user_purchase_history,
        issue_refund_ticket,
    ],
    # Forces every final reply into the Answer schema (answer + conversation flag).
    # With LangChain's tool-based strategy this shows up in traces as a call to a
    # pseudo-tool named "Answer"; see the README for how this affects evaluation.
    response_format=Answer,
    # Keeps message history per thread_id, enabling multi-turn conversations.
    checkpointer=InMemorySaver(),
)


def run_conversation() -> None:
    """
    Run an interactive refund conversation in the terminal.

    Sends an opening request to the agent, prints each reply, and keeps asking
    for user input until the agent sets `conversation` to False in its
    structured response (a final refund decision was made).
    """
    # One thread_id per conversation, so the checkpointer keeps its history together.
    config = {"configurable": {"thread_id": str(uuid4())}}

    text = "I would like to request a refund"
    print(text)

    question = HumanMessage(content=text)
    conversation = True

    while conversation:
        response = agent.invoke({"messages": question}, config=config)
        structured_response = response["structured_response"]

        print(structured_response.answer)

        # The agent decides whether it needs more information from the customer.
        conversation = structured_response.conversation

        if conversation:
            user_input = input("reply: \n")
            question = HumanMessage(content=user_input)

    print("the job is done!")


if __name__ == "__main__":
    run_conversation()
