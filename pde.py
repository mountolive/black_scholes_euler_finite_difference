#!/usr/bin/python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
import math

# PDE Classes


class ConventionDiffusionPDE(ABC):

    def __init__(self, option):
        self.option = option
        super().__init__()

    @abstractmethod
    def diff_coeff(self, t, x):
        pass

    @abstractmethod
    def conv_coeff(self, t, x):
        pass

    @abstractmethod
    def zero_coeff(self, t, x):
        pass

    @abstractmethod
    def source_coeff(self, t, x):
        pass

    @abstractmethod
    def boundary_left(self, t, x):
        pass

    @abstractmethod
    def boundary_right(self, t, x):
        pass

    @abstractmethod
    def init_cond(self, x):
        pass


class BlackScholesPDE(ConventionDiffusionPDE):

    def diff_coeff(self, t, x):
        return 0.5 * (self.option.sigma**2) * (x**2)

    def conv_coeff(self, t, x):
        return self.option.r * x

    def zero_coeff(self, t, x):
        return -self.option.r

    def source_coeff(self, t, x):
        return 0.0

    def boundary_left(self, t, x):
        """
        This is for a Call Option
        """
        return 0.0

    def boundary_right(self, t, x):
        """
        This is for a Call Option
        """
        return x - (self.option.k * math.exp((-self.option.r) * (self.option.T -
                                                                 t)))

    def init_cond(self, x):
        return self.option.payoff.operator(x)
