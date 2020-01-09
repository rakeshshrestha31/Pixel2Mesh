class CheckBest:
    def __init__(self, key, metric_name, is_loss=True):
        self.key = key
        self.metric_name = metric_name
        self.best = float('inf') if is_loss else \
                    -float('inf')
        self.is_best = lambda x: x < self.best if is_loss else \
                  lambda x: x > self.best

    def check_best(self, x):
        if self.is_best(x):
            self.best = x
            return True
        else:
            return False
