import torch

from clip import clip_gradient


class CoordinateDoWG(torch.optim.Optimizer):
    """Implements CDoWG-- a coordinate-wise version of DoWG.

        Args:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            eps (float, optional): term added to the denominator to improve
                numerical stability (default: 1e-8)
        """

    def __init__(self, params, epsilon=1e-8, clip=0.5, *args, **kwargs):
        defaults = dict(r_epsilon=epsilon, lr=1, clip=clip)
        super(CoordinateDoWG, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            with torch.no_grad():
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    state = self.state[p]

                    # Initialize state variables
                    if 'x0' not in state:
                        state['x0'] = torch.clone(p).detach()
                    if 'rt2' not in state:
                        state['rt2'] = torch.zeros_like(p.data).add_(1e-8)
                    if 'vt' not in state:
                        state['vt'] = torch.zeros_like(p.data)

                    state['rt2'] = torch.max(state['rt2'], (p - state['x0']) ** 2)
                    rt2, vt = state['rt2'], state['vt']
                    vt.add_(rt2 * grad ** 2)
                    gt_hat = rt2 * clip_gradient(p, group['clip'])
                    denom = vt.sqrt().add_(group['epsilon'])
                    p.data.addcdiv_(gt_hat, denom, value=-1.0)
        return loss
    


