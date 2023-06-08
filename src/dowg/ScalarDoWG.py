import torch

from clip import clip_gradient


class ScalarDoWG(torch.optim.Optimizer):
    def __init__(self, params, epsilon=1e-4, clip=0.5, *args, **kwargs):
        defaults = dict(r_epsilon=epsilon, lr=1, clip=clip)
        self.epsilon = epsilon
        super(ScalarDoWG, self).__init__(params, defaults)

    def step(self):
        state = self.state

        with torch.no_grad():
            device = self.param_groups[0]["params"][0].device

            if "rt2" not in state:
                state["rt2"] = torch.Tensor([self.epsilon]).to(device)
            if "vt" not in state:
                state["vt"] = torch.Tensor([0]).to(device)

            grad_sq_norm = torch.Tensor([0]).to(device)
            curr_d2 = torch.Tensor([0]).to(device)

            for idx, group in enumerate(self.param_groups):
                group_state = state[str(idx)]  # convert idx to a string
                if "x0" not in group_state:
                    group_state["x0"] = [torch.clone(p) for p in group["params"]]

                grad_sq_norm += torch.stack(
                    [(p.grad**2).sum() for p in group["params"]]
                ).sum()
                curr_d2 += torch.stack(
                    [
                        ((p - p0) ** 2).sum()
                        for p, p0 in zip(group["params"], group_state["x0"])
                    ]
                ).sum()

            state["rt2"] = torch.max(state["rt2"], curr_d2)
            state["vt"] += state["rt2"] * grad_sq_norm
            rt2, vt = state["rt2"], state["vt"]

            for group in self.param_groups:
                for p in group["params"]:
                    gt_hat = rt2 * clip_gradient(p.grad.data, group["clip"])
                    denom = torch.sqrt(vt).add_(group["epsilon"])
                    p.data.addcdiv_(gt_hat, denom, value=-1.0)
        return None
