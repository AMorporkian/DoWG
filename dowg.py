import torch

class DoWG(torch.optim.Optimizer):
    def __init__(self, params, r_epsilon=1e-8, *args, **kwargs):
        defaults = dict(r_epsilon=r_epsilon, lr=0)  # Learning rate isn't used in this algorithm but kept here for compatibility
        super(DoWG, self).__init__(params, defaults)
    
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['v_prev'] = torch.zeros_like(p)
                    state['r_prev'] = torch.full_like(p, group['r_epsilon'])
                    state['x0'] = p.data.clone()

                v_prev, r_prev, x0 = state['v_prev'], state['r_prev'], state['x0']

                # Algorithm step
                state['step'] += 1

                # Update distance estimator
                r_t = torch.max(torch.norm(p.data - x0), r_prev)

                # Update weighted gradient sum
                v_t = v_prev + r_t.pow(2) * torch.norm(d_p).pow(2)

                # Set the stepsize
                eta_t = r_t.pow(2) / torch.sqrt(v_t)

                # Gradient descent step
                with torch.no_grad():
                    p.add_(d_p, alpha=-eta_t.mean().item())

                # Update the state
                state['v_prev'] = v_t
                state['r_prev'] = r_t

class DoWG8bit(DoWG):
    """An 8bit quantized implementation of DoWG."""
    def __init__(self, params, r_epsilon=1e-8, *args, **kwargs):
        super(DoWG8bit, self).__init__(params, r_epsilon, *args, **kwargs)
        self.params = params
        self.r_epsilon = r_epsilon
        self.args = args
        self.kwargs = kwargs
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def step(self, closure=None):
        # Quantize the model
        self.quant(self.params)
        super(DoWG8bit, self).step(closure)
        # Dequantize the model
        self.dequant(self.params)