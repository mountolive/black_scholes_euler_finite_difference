#!/usr/bin/python
# -*- coding: utf-8 -*-
# Option class


class VanillaOption(object):

    def __init__(self, payoff, k, r, T, sigma):
        self.payoff = payoff
        self.k = k
        self.r = r
        self.T = T
        self.sigma = sigma
