import re

def format_reward_func(completions: list[str], target: list[str], **kwargs) -> list[float]:
    """Calculate reward scores based on proper thought structure formatting.
    
    Checks if completions follow the format: <think>...</think><answer>...</answer>
    
    Args:
        completions: List of generated text completions to evaluate
        target: List of expected completions (used for length reference)
        **kwargs: Additional keyword arguments (unused)
        
    Returns:
        List of reward scores between 0-1, where:
        - 1.0: Completion has correct think/answer structure
        - 0.0: Completion has incorrect structure
    """
    rewards = []

    for completion, _ in zip(completions, target):
        try:
            regex = r"<think>([^<]*(?:<(?!/?think>)[^<]*)*)</think>([\s\S]*?)$"

            match = re.search(regex, completion, re.DOTALL)

            # if the format is not correct, reward is 0
            if match is None or len(match.groups()) != 2:
                rewards.append(0.0)
            else:
                rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards