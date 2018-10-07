#!/usr/bin/python
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod

# PayOff classes


class Payoff(ABC):

    def __init__(self, k):
        self.k = k
        super().__init__()

    @abstractmethod
    def operator(self, s):
        pass


class PayoffCall(Payoff):

    def operator(self, s):
        return max(s - self.k, 0.0)


class PayoffPut(Payoff):

    def operator(self, s):
        return max(self.k - s, 0.0)