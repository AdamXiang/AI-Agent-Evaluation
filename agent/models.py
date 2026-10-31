"""
Chat model used by the refund agent.

Claude Haiku 5.5 is the cheapest current Claude model, which keeps the many
agent runs made during evaluation affordable. The API key is read from the
ANTHROPIC_API_KEY environment variable (loaded from .env).

To use a different model, change the name here; the rest of the agent is
model-agnostic.
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load ANTHROPIC_API_KEY from .env before the model client is created.
load_dotenv()

model = init_chat_model(model="claude-haiku-5-5")
