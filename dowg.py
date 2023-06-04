import torch
from torch.cuda.amp import autocast

class DoWG(torch.optim.Optimizer):
    def __init__(self, params, r_epsilon=5e-7, clip=0.5, *args, **kwargs):
        defaults = dict(r_epsilon=r_epsilon, lr=1)  # Learning rate isn't used in this algorithm but kept here for compatibility
        super(DoWG, self).__init__(params, defaults)
        self.step_sizes = [r_epsilon]
        self.step_dx = 0
        self.clip = clip
    def step(self, closure=None):
        with autocast():
            loss = None
            if closure is not None:
                loss = closure()
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    d_p = p.grad
                    p.grad.data.clamp_(-self.clip, self.clip)
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
                    v_t = torch.addcmul(v_prev, r_t.pow(2), torch.norm(d_p).pow(2))
                    if not v_t.any(): 
                        continue # Avoid any division by 0.
                    # Set the stepsize
                    eta_t = r_t.pow(2) / (torch.sqrt(v_t) + 1e-6)
                    self.step_sizes.append(torch.mean(eta_t).item())
                    if state['step'] > 1:
                        self.step_dx = self.step_sizes[-1] - self.step_sizes[-2]
                # print(f"step size: {torch.mean(eta_t).item()}")

                    # Gradient descent step
                    #p.data.require_grad = True
                    p.data.addcmul_(-eta_t, d_p)
                    
                    # Update the state
                    state['v_prev'] = v_t
                    state['r_prev'] = r_t
                    del v_t, r_t, eta_t, d_p, v_prev, r_prev
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            return loss
class DoWG8bit(DoWG):
    """An 8bit quantized implementation of DoWG."""
    def __init__(self, params, r_epsilon=1e-5, *args, **kwargs):
        # This doesn't work very well right now.
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
        loss = super(DoWG8bit, self).step(closure)
        # Dequantize the model
        self.dequant(self.params)
        return loss
