"""
LLM judge used by every DeepEval metric in this project.

The judge reads agent outputs and traces and decides scores, so its quality
caps how trustworthy the evaluation is. Claude Haiku 5.5 keeps costs low; a
stronger model (e.g. Sonnet) gives more reliable verdicts on borderline cases.
"""

from deepeval.models import AnthropicModel

eval_model = AnthropicModel("claude-haiku-5-5")
