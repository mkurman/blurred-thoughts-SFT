def get_parameters_count(model) -> tuple[int, int]:
    """Get total and trainable parameter counts for a model.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Tuple containing:
        - Total number of parameters
        - Number of trainable parameters
    """
    total_numel = sum(p.numel() for p in model.parameters())
    trainable_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_numel, trainable_numel