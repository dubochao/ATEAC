
d_model=512

class TransformerOptimizer(object):
    def __init__(self, optimizer, warmup_steps=300):
        self.optimizer = optimizer
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.lr = self.init_lr
        self.step_num = 0
        self.stop_up = False
    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self._update_lr()
        self.optimizer.step()

    def _update_lr(self):
        self.step_num += 1
        self.min_lr = 1e-4
        self.lr = self.init_lr * min(self.step_num ** (-0.65), self.step_num * (self.warmup_steps ** (-1.5)))
        self.lr = max(self.lr, self.min_lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
