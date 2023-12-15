import torch


class RunningAverage:
    """
    Computes the running average
    """

    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.data = None

    def calculate(self, new_data: torch.Tensor):
        if self.data is None:
            self.data = torch.zeros_like(new_data).to(new_data.device)
        return self.alpha * self.data.detach() + (1 - self.alpha) * new_data

    def update(self, new_data: torch.Tensor):
        self.data = self.calculate(new_data)

    def get_value(self):
        if self.data is None:
            raise Exception("Data was not initialized!")
        return self.data


class RunningMean:
    def __init__(self):
        super(RunningMean, self).__init__()

        self.values = {}

    def update(self, values: dict):
        for k, v in values.items():
            if k not in self.values:
                self.values[k] = (v, 1)
            else:
                m, n = self.values[k]
                self.values[k] = ((n * m + v) / (n + 1), n + 1)

    def get_values(self):
        return {k: v[0] for k, v in self.values.items()}
