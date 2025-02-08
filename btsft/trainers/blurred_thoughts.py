from transformers import Trainer
import torch
from typing import Optional, Tuple, Union, Dict, Any

class BlurredThoughtsSFTTrainer(Trainer):
    """
    A custom trainer implementation for Blurred Thoughts Supervised Fine-Tuning (BT-SFT).
    
    This trainer extends the Hugging Face Trainer class to implement the BT-SFT training
    methodology, which combines traditional language model training with a unique reward
    mechanism that encourages structured thought processes through <think> tags.
    
    The trainer implements two key components:
    1. Traditional language model loss for next token prediction
    2. A reward-based loss component that evaluates the model's adherence to the
       expected thought process structure
    
    Attributes:
        tokenizer: The tokenizer used for encoding/decoding text
        bf_beta (float): The beta parameter controlling the influence of the reward loss
        format_reward_func (callable): Function that calculates format adherence rewards
    """

    def __init__(
        self, 
        *args, 
        tokenizer: Optional[Any] = None, 
        bf_beta: float = 0.05, 
        format_reward_func: Optional[callable] = None, 
        **kwargs
    ):
        """
        Initialize the BT-SFT trainer.

        Args:
            tokenizer: Tokenizer for text encoding/decoding
            bf_beta: Weight factor for the reward loss component (default: 0.05)
            format_reward_func: Function that evaluates format adherence
            *args: Additional positional arguments passed to the parent Trainer
            **kwargs: Additional keyword arguments passed to the parent Trainer
        """
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.bf_beta = bf_beta
        self.format_reward_func = format_reward_func

    def compute_loss(
        self, 
        model: Any, 
        inputs: Dict[str, torch.Tensor], 
        num_items_in_batch: Optional[int] = None, 
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Compute the combined loss for BT-SFT training.

        This method combines two loss components:
        1. Standard language modeling loss from the base model
        2. Format adherence reward loss based on the presence and proper use of <think> tags

        Args:
            model: The neural network model being trained
            inputs: Dictionary containing the input tensors
            num_items_in_batch: Optional batch size override
            return_outputs: If True, return the model outputs along with the loss

        Returns:
            If return_outputs is False, returns the combined loss tensor.
            If return_outputs is True, returns a tuple of (loss, model_outputs).
        """
        outputs = self.model(
            inputs["input_ids"],
            labels=inputs["labels"],
            num_items_in_batch=num_items_in_batch,
            return_dict=True,
        )
        logits = outputs.logits
        loss = outputs.loss

        completions = self.tokenizer.batch_decode(
            logits.argmax(dim=-1), skip_special_tokens=False
        )

        rewards = self.format_reward_func(
            completions,
            self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=False),
        )
        rewards = torch.tensor(rewards).mean()
        rewards = 1 - rewards

        loss = loss + self.bf_beta * rewards

        return (loss, outputs) if return_outputs else loss
