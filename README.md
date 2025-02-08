# Blurred Thoughts SFT

Blurred Thoughts SFT is a project designed to train a language model using a unique approach called "blurred thoughts." This method involves fine-tuning a model with a focus on generating structured thought processes, enhancing the quality of generated text.

**Blurred-Thoughts Supervised-Finetuning (BT-SFT)**

BT-SFT is a novel fine-tuning technique for language models, designed to foster diverse and creative responses. It employs a unique tokenization method, randomly masking tokens within `<think>` tags by setting their labels to -100, which prevents the model from strictly adhering to the training data. This encourages the model to generate more varied responses, aligned with its own probability distribution. Additionally, BT-SFT introduces a reward function that penalizes the model if it fails to follow the `<think> ... </think>` template, ensuring that responses remain coherent and structured while promoting creativity. This approach results in more diverse thought processes less constrained by the training data, leading to more creative and engaging text generation.

## Project Structure

```
blurred-thoughts-SFT
├── btsft
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── func
│   │       ├── __init__.py
│   │       ├── format_reward.py
│   │       ├── parameters.py
│   │       ├── mapping.py
│   │       └── training.py
├── setup.py
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd blurred-thoughts-SFT
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -e .
```
## Usage
Command Line Interface

Train a model using the default configuration:
```bash
btsft --config config/default.yaml
```
## Configuration
Create a custom YAML configuration file:

```yaml
# Model configuration
model_name: "llama"
threshold: 0.2 # Blurred thoughts threshold. I encourage you to experiment with this value to see how it affects the model's performance
bf_beta: 0.05 # Blurred thoughts beta parameter to control the reward function influence on the final loss. I encourage you to experiment with this value to see how it affects the model's performance

# Checkpoints
checkpoint: "path/to/checkpoint"
trainer_checkpoint: null
base_model_checkpoint: null

# Training parameters
tokenizer_name: "gpt2"
dataset_train: "path/to/dataset"
max_length: 512
batch_size: 32
accumulation_iter: 12
epochs: 1
lr: 5e-5
warmup_steps: 500
weight_decay: 0.01
```

Save the configuration file and run the training script with the `--config` argument:
```bash
btsft --config config/custom.yaml
```

## Key Features
- Blurred thoughts fine-tuning mechanism
- Support for various model architectures
- Configurable training parameters
- LoRA training support
- Distributed training support
- Integrated logging with Tensorboard and Weights & Biases
- Automatic mixed precision training

## Development
To contribute to the project:

1. Fork the repository on GitHub for your feature
2. Make your changes
3. Write or update tests
4. Submit a pull request

## License
This project is licensed under the MIT License. See the LICENSE file for details.