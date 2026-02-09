import numpy as np


class Oracle:

    def __init__(self, obj_func, noisy=False, noise_model=None,
                 noise_params=None, num_samples=1):
        if not callable(obj_func):
            raise TypeError("obj_func must be callable")

        self.obj_func = obj_func
        self.noisy = noisy
        self.noise_model = noise_model
        self.noise_params = noise_params or {}
        self.num_samples = num_samples

        # Track evaluation count
        self.evaluation_count = 0

        # Track evaluation history (optional, for debugging/analysis)
        self.evaluation_history = []

    def __call__(self, x):
        if self.noisy:
            return self._evaluate_noisy(x)
        else:
            return self._evaluate_deterministic(x)

    def _evaluate_deterministic(self, x):
        x = np.asarray(x)

        if x.ndim == 1:
            fx = self.obj_func(x)
            self.evaluation_count += 1
            self.evaluation_history.append((x.copy(), fx))
            return fx
        elif x.ndim == 2:
            fx = np.array([self.obj_func(xi) for xi in x])
            self.evaluation_count += len(x)
            for xi, fxi in zip(x, fx):
                self.evaluation_history.append((xi.copy(), fxi))
            return fx
        else:
            raise ValueError(f"x must be 1D (single point) or 2D (array of points), got {x.ndim}D")

    def _evaluate_noisy(self, x):
        # Placeholder: currently just calls deterministic evaluation
        # TODO: Implement noisy evaluation logic
        # Example future implementation:
        #   samples = []
        #   for _ in range(self.num_samples):
        #       fx = self.obj_func(x)
        #       noisy_fx = self._apply_noise(fx)
        #       samples.append(noisy_fx)
        #   self.evaluation_count += self.num_samples
        #   return np.mean(samples)

        return self._evaluate_deterministic(x)

    def _apply_noise(self, fx):
        # Placeholder: future implementation
        # if self.noise_model == 'gaussian':
        #     noise = np.random.normal(0, self.noise_params.get('std', 1.0))
        #     return fx + noise
        # elif self.noise_model == 'uniform':
        #     noise = np.random.uniform(
        #         -self.noise_params.get('range', 1.0),
        #         self.noise_params.get('range', 1.0)
        #     )
        #     return fx + noise
        # else:
        #     raise ValueError(f"Unknown noise model: {self.noise_model}")

        return fx

    def reset(self):
        self.evaluation_count = 0
        self.evaluation_history = []

    def get_evaluation_count(self):
        return self.evaluation_count

    def get_evaluation_history(self):
        return self.evaluation_history.copy()
