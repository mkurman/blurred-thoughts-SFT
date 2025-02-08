from dataclasses import dataclass, asdict
from typing import Optional
import yaml
from pathlib import Path

@dataclass
class Config:
    """Configuration for training a language model with blurred thoughts SFT"""
    
    # Model configuration
    model_name: str = "llama"  # Name of the model architecture to use (e.g. llama)
    threshold: float = 0.2  # Probability threshold for masking tokens between <think> tags (0-1)
    bf_beta: float = 0.05  # Hyperparameter for partial blurred thoughts loss addition
    
    # Checkpoints and Tokenizer
    checkpoint: str = "mkurman/Llama-3.2-MedIT-SUN-2.5B-BT-GRPO"  # Path to model checkpoint to resume from
    trainer_checkpoint: Optional[str] = None  # Path to trainer state checkpoint to resume from
    base_model_checkpoint: Optional[str] = None  # Path to base model checkpoint for initialization
    tokenizer_name: Optional[str] = None  # Name or path of tokenizer to use. If None, uses checkpoint value
    
    #  Dataset
    dataset_train: str = "mkurman/simplescaling-s1K-R1"  # Path to training dataset file
    max_length: int = 512  # Maximum sequence length for training
    
    # Training parameters
    save_steps: int = 500  # Save checkpoint every N steps
    batch_size: int = 32  # Training batch size per device
    accumulation_iter: int = 12  # Number of batches for gradient accumulation
    epochs: int = 1  # Number of training epochs
    lr: float = 5e-5  # Learning rate
    warmup_steps: int = 500  # Number of warmup steps for learning rate scheduler
    weight_decay: float = 0.01  # Weight decay coefficient for AdamW optimizer
    
    # Directories
    logging_dir: str = "./logs"  # Directory to save training logs
    output_dir: str = "./results"  # Directory to save model checkpoints
    cache_dir: Optional[str] = None  # Directory to save cached datasets
    
    # Data handling
    train_test_split: float = 0.1  # Fraction of data to use for validation (0-1)
    seed: int = 42  # Random seed for reproducibility
    skip: int = 0  # Number of training examples to skip
    take: Optional[int] = None  # Maximum number of training examples to use
    num_workers: int = 24  # Number of workers for dataset processing
    
    # Other settings
    device: str = "cuda"  # Device to use for training (cuda/cpu)
    response_template: str = "<|start_header_id|>assistant<|end_header_id|>\n\n"  # Response template
    lora_rank: int = 64  # Rank of LoRA projection matrices

    def __post_init__(self):
        """Cast and validate configuration parameters after initialization"""
        # Cast numeric values to their proper types
        self.threshold = float(self.threshold)
        self.bf_beta = float(self.bf_beta)
        self.train_test_split = float(self.train_test_split)
        self.batch_size = int(self.batch_size)
        self.accumulation_iter = int(self.accumulation_iter)
        self.epochs = int(self.epochs)
        self.lr = float(self.lr)
        self.warmup_steps = int(self.warmup_steps)
        self.weight_decay = float(self.weight_decay)
        self.lora_rank = int(self.lora_rank)
        self.max_length = int(self.max_length)
        self.num_workers = int(self.num_workers)
        self.save_steps = int(self.save_steps)
        self.skip = int(self.skip)
        
        # Convert take to int if it's not None
        if self.take is not None:
            self.take = int(self.take)

        # Validation checks
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {self.threshold}")
        
        if not 0.0 <= self.bf_beta <= 1.0:
            raise ValueError(f"bf_beta must be between 0.0 and 1.0, got {self.bf_beta}")
            
        if not 0.0 <= self.train_test_split <= 1.0:
            raise ValueError(f"train_test_split must be between 0.0 and 1.0, got {self.train_test_split}")
            
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
            
        if self.accumulation_iter <= 0:
            raise ValueError(f"accumulation_iter must be positive, got {self.accumulation_iter}")
            
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
            
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
            
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {self.warmup_steps}")
            
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
            
        if self.lora_rank <= 0:
            raise ValueError(f"lora_rank must be positive, got {self.lora_rank}")
            
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
            
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {k: v for k, v in self.__dict__.items()}
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def to_dict(self) -> dict:
        """Convert config to a dictionary, filtering out None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}