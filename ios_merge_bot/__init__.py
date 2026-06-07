"""ios_merge_bot: autonomous DRL agent for an iOS auto-battler.

Closed-loop architecture:
    Perceive (screen capture + CV) -> Think (Gymnasium env + Transformer) -> Act (WDA touch).
"""

__version__ = "0.1.0"
