import warnings
from argparse import ArgumentParser
from btsft.config import Config
from btsft.func.training import train

warnings.filterwarnings("ignore")

def main():
    parser = ArgumentParser(description="Train a language model with Blurred Thoughts SFT")
    parser.add_argument(
        "--config", 
        type=str,
        default="config/default.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = Config.from_yaml(args.config)
    
    # Start training with cleaned config dictionary
    train(**config.to_dict())

if __name__ == "__main__":
    main()