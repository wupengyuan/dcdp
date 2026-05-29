import numpy as np
import pdb

class NoiseGenerator:
    def __init__(self, noise_strength, correlation_factor=0.9):
        self.noise_strength = noise_strength
        self.correlation_factor = correlation_factor
        self.previous_noise = None

    def step(self, pred):
        # Generate random noise
        # noise_seed = np.random.randn(*pred) * self.noise_strength
        noise_seed = (np.random.rand(pred.shape[0], 1, pred.shape[2]) + 0.5) * np.random.choice([-1, 1], size=(pred.shape[0], 1, pred.shape[2]))
        action_step = (pred[:, 1:] - pred[:, :-1])
        noise_step = noise_seed.repeat(action_step.shape[1], axis=1) * action_step * self.noise_strength

        # If it's the first time step, there's no previous noise, so use the seed directly
        if self.previous_noise is None:
            self.previous_noise = noise_step
        else:
            # Combine the previous noise with new noise to create temporally correlated noise
            noise_step = self.correlation_factor * self.previous_noise + (1 - self.correlation_factor) * noise_step
            self.previous_noise = noise_step

        noise_cum = np.cumsum(noise_step, axis=1)

        return noise_cum
