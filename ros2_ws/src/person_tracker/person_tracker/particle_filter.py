"""
Particle Filter for person tracking.

State: [x, y, z, vx, vy, vz]  (position + velocity)

Motion model: constant velocity + Gaussian noise
Observation model: multi-modal Gaussian + robust (min weight floor)
Resampling: systematic, triggered by ESS threshold
Initialization: uniform distribution over search space
"""

import numpy as np


class ParticleFilter:
    def __init__(
        self,
        num_particles: int = 300,
        # Search space for initialization (world frame)
        x_range: tuple = (-5.0, 10.0),
        y_range: tuple = (-3.0, 3.0),
        z_range: tuple = (0.0, 2.0),
        # Noise (per step)
        pos_noise_xy: float = 0.05,
        pos_noise_z: float = 0.02,
        vel_noise_xy: float = 0.10,
        vel_noise_z: float = 0.05,
        # Observation model
        obs_sigma: float = 0.10,
        min_weight: float = 0.01,
        # Resampling
        ess_ratio_threshold: float = 0.5,
    ):
        self.n = num_particles
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

        self.pos_noise_xy = pos_noise_xy
        self.pos_noise_z = pos_noise_z
        self.vel_noise_xy = vel_noise_xy
        self.vel_noise_z = vel_noise_z

        self.obs_sigma = obs_sigma
        self.min_weight = min_weight
        self.ess_ratio_threshold = ess_ratio_threshold

        # State: (n, 6) [x, y, z, vx, vy, vz]
        self.particles = None
        self.weights = None

        self.rng = np.random.default_rng()

        self.initialized = False

    # ==========================
    # Initialization
    # ==========================
    def initialize_uniform(self):
        """Distribute particles uniformly over search space."""
        self.particles = np.zeros((self.n, 6), dtype=np.float32)
        # Position: uniform in search space
        self.particles[:, 0] = self.rng.uniform(
            self.x_range[0], self.x_range[1], self.n)
        self.particles[:, 1] = self.rng.uniform(
            self.y_range[0], self.y_range[1], self.n)
        self.particles[:, 2] = self.rng.uniform(
            self.z_range[0], self.z_range[1], self.n)
        # Velocity: zero (unknown initial motion)
        self.particles[:, 3:6] = 0.0

        # Uniform weights
        self.weights = np.ones(self.n, dtype=np.float32) / self.n

        self.initialized = True

    # ==========================
    # Prediction
    # ==========================
    def predict(self, dt: float):
        """Constant velocity motion model with Gaussian noise."""
        if not self.initialized:
            return

        # Position update: x_{t+1} = x_t + v_t * dt + noise
        self.particles[:, 0] += self.particles[:, 3] * dt
        self.particles[:, 1] += self.particles[:, 4] * dt
        self.particles[:, 2] += self.particles[:, 5] * dt

        # Add position noise
        self.particles[:, 0] += self.rng.normal(
            0, self.pos_noise_xy, self.n)
        self.particles[:, 1] += self.rng.normal(
            0, self.pos_noise_xy, self.n)
        self.particles[:, 2] += self.rng.normal(
            0, self.pos_noise_z, self.n)

        # Velocity: random walk (add noise)
        self.particles[:, 3] += self.rng.normal(
            0, self.vel_noise_xy, self.n)
        self.particles[:, 4] += self.rng.normal(
            0, self.vel_noise_xy, self.n)
        self.particles[:, 5] += self.rng.normal(
            0, self.vel_noise_z, self.n)

    # ==========================
    # Update
    # ==========================
    def update(self, observation: np.ndarray):
        """Update weights based on observation.
        
        observation: (3,) [x, y, z] measured position
        Multi-modal Gaussian likelihood + robust min weight.
        """
        if not self.initialized:
            return

        # Distance from each particle to observation
        # Only position components (0:3)
        diff = self.particles[:, :3] - observation
        dist_sq = np.sum(diff ** 2, axis=1)

        # Gaussian likelihood
        likelihood = np.exp(-dist_sq / (2 * self.obs_sigma ** 2))

        # Robust: apply min weight floor
        likelihood = np.maximum(likelihood, self.min_weight)

        # Multiply with prior weights
        self.weights *= likelihood

        # Normalize
        total = np.sum(self.weights)
        if total > 0:
            self.weights /= total
        else:
            # All weights zero (all particles far from obs)
            # Reset to uniform
            self.weights = np.ones(self.n, dtype=np.float32) / self.n

    # ==========================
    # Resampling
    # ==========================
    def effective_sample_size(self) -> float:
        """ESS = 1 / sum(w_i^2)."""
        return 1.0 / np.sum(self.weights ** 2)

    def resample_if_needed(self) -> bool:
        """Systematic resampling if ESS below threshold."""
        if not self.initialized:
            return False

        ess = self.effective_sample_size()
        if ess < self.n * self.ess_ratio_threshold:
            self.systematic_resample()
            return True
        return False

    def systematic_resample(self):
        """Systematic resampling."""
        # Cumulative sum of weights
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # ensure exact 1

        # Sample points evenly spaced
        u0 = self.rng.uniform(0, 1.0 / self.n)
        u = u0 + np.arange(self.n) / self.n

        # Find indices
        indices = np.searchsorted(cumsum, u)
        indices = np.clip(indices, 0, self.n - 1)

        # Resample
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n, dtype=np.float32) / self.n

    # ==========================
    # Estimation
    # ==========================
    def estimate(self) -> tuple:
        """Weighted mean state and position covariance.
        Returns: (position (3,), velocity (3,), cov (3,3))
        """
        if not self.initialized:
            return None, None, None

        # Weighted mean
        w = self.weights[:, np.newaxis]  # (n, 1)
        mean = np.sum(self.particles * w, axis=0)

        position = mean[:3]
        velocity = mean[3:6]

        # Position covariance
        pos_diff = self.particles[:, :3] - position
        cov = (pos_diff * w).T @ pos_diff

        return position, velocity, cov