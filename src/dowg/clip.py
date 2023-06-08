import torch


def clip_gradient(grad, k):
    """Clips the top k% of the gradients in terms of absolute value.

    Args:
        grad (torch.Tensor): The gradients to be clipped.
        k (float): The percentage of gradients to keep.

    Returns:
        torch.Tensor: The clipped gradients.
    """
    if len(grad.size()) < 2 or grad.size(1) < 2:
        return grad

    if k >= 1.0:
        return grad

    # Compute the threshold
    #print(f"grad: {grad}, k: {k}, grad.size(1): {grad.size(1)}, int(grad.size(1) * (1 - k))): {int(grad.size(1) * (1 - k))}")
    threshold = torch.kthvalue(torch.abs(grad), int(grad.size(1) * (1 - k)), dim=1)[0]

    # Clip the gradients
    clipped_grad = torch.clamp(torch.abs(grad), max=threshold.unsqueeze(1))
    clipped_grad *= torch.sign(grad)

    return clipped_grad