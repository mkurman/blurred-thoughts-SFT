def map_iio(examples: dict) -> list[dict]:
    """Map instruction-input-output format to conversation format.
    
    Converts dataset examples from instruction/input/output fields into
    a list of role-based conversation messages.
    
    Args:
        examples: Dictionary containing instruction, input and output fields
        
    Returns:
        List of dictionaries with 'conversations' key containing list of messages with roles:
        - system: Contains instruction if present
        - user: Contains input if present  
        - assistant: Contains output if present
        Returns None if required fields are empty
    """
    tmp = []

    if "instruction" in examples and examples["instruction"] is not None:
        instruction = examples["instruction"]

        if len(instruction.strip()) != 0:
            tmp.append({"role": "system", "content": instruction})

    if "input" in examples and examples["input"] is not None:
        input_text = examples["input"]

        if len(input_text.strip()) == 0:
            return None

        tmp.append({"role": "user", "content": input_text})
    else:
        return None

    if "output" in examples and examples["output"] is not None:
        output_text = examples["output"]

        if len(output_text.strip()) == 0:
            return None

        tmp.append({"role": "assistant", "content": output_text})
    else:
        return None

    return tmp

def map_conversations(examples) -> list[dict]:
    """
    Map examples to conversation format.

    Args:
        examples: List of examples containing role and content fields

    Returns:
        List of dictionaries with 'conversations' key containing list of messages with roles:
        - system: Contains instruction if present
        - user: Contains input if present  
        - assistant: Contains output if present
        Returns None if required fields are empty
    
    """
    for idx, example in enumerate(examples):
        if "role" not in example:
            example["role"] = example["from"]

            if example["role"] == "human":
                example["role"] = "user"

            if example["role"] == "gpt":
                example["role"] = "assistant"

        if "content" not in example:
            example["content"] = example["value"]

        examples[idx] = {"role": example["role"], "content": example["content"].replace("<|begin_of_thought|>\n", "<think>").replace("\n<|end_of_thought|>\n", "</think>").replace("\n<|begin_of_solution|>\n", "").replace("\n\n<|end_of_solution|>", "")}

    return examples