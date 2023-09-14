"""Convenient specification of distributions as classes. 
    This allows a simple parameterization (see SimulatorConfig.py).
    This allows a simple sampling or fixation during simulation (see EbsDataSimulator.py)
"""
from abc import ABC, abstractmethod
import numpy as np
import scipy.stats as stats


class EbsDistribution(ABC):

    """Base class for 'parameterizable' distributions."""

    @abstractmethod
    def sample(self, size):
        """Draw a sample from the distribution.

        Args:
            size (int): Number of samples to draw

        Raises:
            NotImplementedError: Raised if base class is accessed.
        """
        raise NotImplementedError("Accessing the base class is not allowed.")


class Singleton(EbsDistribution):

    """Singleton distribution == provides only a single value.

    Attributes:
        fixed_value (float): Fixed and only value of the distribution.
    """

    # used for re-simulation
    fixed_value = None

    def __init__(self, fixed_value):
        """Ctor.

        Args:
            fixed_value (float): Fixed and only value of the distribution.
        """
        self.fixed_value = fixed_value

    def sample(self, size=1):
        """Draw a sample from the distribution.

        Args:
            size (int, optional): Number of samples to draw

        Returns:
            float: Fixed and only value of the distribution.
        """
        if size == 1:
            return self.fixed_value

        return np.ones(size) * self.fixed_value


class Normal(EbsDistribution):

    """Normal distribution.

    Attributes:
        mean (float): Mean of the normal
        std (float): Standard deviation of the normal
    """

    mean = None
    std = None

    def __init__(self, mean=0, std=1):
        """Ctor.

        Args:
            mean (int, optional): Mean of the normal
            std (int, optional): Standard deviation of the normal
        """
        self.mean = mean
        self.std = std

    def sample(self, size=1):
        """Draw a sample from the distribution.

        Args:
            size (int, optional): Number of samples to draw

        Returns:
            np.array<float>: Sampled values
        """
        return np.random.normal(self.mean, self.std, size=size)


class TruncNormal(EbsDistribution):

    """Truncated Normal distribution.

    Attributes:
        lower (double): Lower cut-off
        mu (double): Mean of the truncated normal.
        sigma (double): Sigma of the truncated normal.
        upper (double): Upper cut-off
    """

    lower = None
    upper = None
    mu = None
    sigma = None

    def __init__(self, lower_clip, upper_clip, mu, sigma):
        """Ctor.

        Args:
            lower (double): Lower cut-off
            mu (double): Mean of the truncated normal.
            sigma (double): Sigma of the truncated normal.
            upper (double): Upper cut-off
        """
        self.lower = lower_clip
        self.upper = upper_clip
        self.mu = mu
        self.sigma = sigma

    def sample(self, size=1):
        """Draw a sample from the distribution.

        Args:
            size (int, optional): Number of samples to draw

        Returns:
            array<float>: Sampled values
        """
        return stats.truncnorm.rvs(
            a=(self.lower - self.mu) / self.sigma,
            b=(self.upper - self.mu) / self.sigma,
            loc=self.mu,
            scale=self.sigma,
            size=size,
        )


class Pareto(EbsDistribution):

    """Pareto distribution

    Attributes:
        shape (double): Shape factor of the pareto distribution.
    """

    shape = None

    def __init__(self, shape_factor):
        """Ctor.

        Args:
            shape_factor (double): Shape factor of the pareto distribution.
        """
        self.shape = shape_factor

    def sample(self, size=1):
        """Draw a sample from the distribution.

        Args:
            size (int, optional): Number of samples to draw

        Returns:
            np.array<float>: Sampled values
        """
        return np.random.pareto(a=self.shape, size=size)


class TruncPareto(EbsDistribution):

    """Truncated Pareto distribution.

    Attributes:
        lower (double): Lower cut-off
        shape (double): Shape factor of the truncated pareto distribution
        upper (double): Upper cut-off
    """

    shape = None
    lower = None
    upper = None

    def __init__(self, shape_factor, lower_clip, upper_clip):
        """Ctor.

        Args:
            lower (double): Lower cut-off
            shape (double): Shape factor of the truncated pareto distribution
            upper (double): Upper cut-off
        """
        self.shape = shape_factor
        self.lower = lower_clip
        self.upper = upper_clip

    def sample(self, size=1):
        """Draw a sample from the distribution.

        Args:
            size (int, optional): Number of samples to draw

        Returns:
            np.array<float>: Sampled values

        Raises:
            ValueError: Raised if multiple samples are requested
        """
        if size != 1:
            raise ValueError("Currently only a sample size of 1 is supported.")

        sample_val = 0.0
        while True:
            sample_val = np.random.pareto(self.shape, size=1)

            if 0.0 <= sample_val <= 1.0:
                break

        return sample_val * 100.0


class Binomial(EbsDistribution):

    """Binomial distribution

    Attributes:
        prob (float): Probability of state "a"
    """

    prob = None

    def __init__(self, prob=0.5):
        """Ctor.

        Args:
            prob (float, optional): Probability of state "a"
        """
        self.prob = prob

    def sample(self, size=1):
        """Draw a sample from the distribution.

        Args:
            size (int, optional): Number of samples to draw

        Returns:
            np.array<float>: Sampled values
        """
        return np.random.binomial(n=1, p=self.prob, size=size)


class Uniform(EbsDistribution):

    """Uniform distribution

    Attributes:
        lower (float): Low end
        upper (float): Upper end
    """

    lower = None
    upper = None

    def __init__(self, lower_clip, upper_clip):
        """Ctor.

        Args:
            lower (float): Low end
            upper (float): Upper end
        """
        self.lower = lower_clip
        self.upper = upper_clip

    def sample(self, size=1):
        """Draw a sample from the distribution.

        Args:
            size (int, optional): Number of samples to draw

        Returns:
            np.array<float>: Sampled values
        """
        return np.random.uniform(low=self.lower, high=self.upper, size=size)
