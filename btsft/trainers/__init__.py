"""
Trainer implementations for Blurred Thoughts Supervised Fine-Tuning (BT-SFT).

This package provides specialized trainer classes that implement the BT-SFT
methodology, combining traditional language model training with reward-based
learning to encourage structured thought processes in language models.

Available trainers:
    - BlurredThoughtsSFTTrainer: Main trainer implementation for BT-SFT
"""

from .blurred_thoughts import BlurredThoughtsSFTTrainer

__all__ = ['BlurredThoughtsSFTTrainer']
