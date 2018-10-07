#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
from abc import ABC, abstractmethod
from .pde import ConventionDiffusionPDE

# Finite Difference Method


class FDMBase(ABC):

    def __init__(self, x_dom, t_dom, j, n, pde):
        self.x_dom = x_dom
        self.t_dom = t_dom
        self.j = j
        self.n = n
        self.pde = pde
        self.x_values = []
        self.dx, self.dt = 0.0, 0.0
        self.prev_t, self.cur_t = 0.0, 0.0
        self.alpha, self.beta, self.gamma = 0.0, 0.0, 0.0
        self.new_result = []
        self.old_result = []
        super().__init__()

    @abstractmethod
    def calculate_step_size(self):
        pass

    @abstractmethod
    def set_initial_conditions(self):
        pass

    @abstractmethod
    def calculate_boundary_conditions(self):
        pass

    @abstractmethod
    def calculate_inner_domain(self):
        pass

    @abstractmethod
    def step_march(self):
        pass


class FDMEuler(FDMBase):

    def __init__(self, x_dom, t_dom, j, n, pde):
        super().__init__(x_dom, t_dom, j, n, pde)
        self.calculate_step_size()
        self.set_initial_conditions()

    def calculate_step_size(self):
        self.dx = self.x_dom / float(self.j - 1)
        self.dt = self.t_dom / float(self.n - 1)

    def set_initial_conditions(self):
        self.new_result = [0.0]*self.j
        for i in range(self.j):
            cur_spot = float(i) * self.dx
            self.old_result.append(self.pde.init_cond(cur_spot))
            self.x_values.append(cur_spot)

    def calculate_boundary_conditions(self):
        self.pde.boundary_left(self.prev_t, self.x_values[0])
        self.pde.boundary_right(self.prev_t, self.x_values[self.j - 1])

    def calculate_inner_domain(self):
        for i in range(1, self.j):
            dt_sig = self.dt * (self.pde.diff_coeff(self.prev_t,
                                                    self.x_values[0]))
            dt_sig_2 = self.dt * self.dx * 0.5 * self.pde.conv_coeff(
                self.prev_t, self.x_values[i])
            self.alpha = dt_sig - dt_sig_2
            self.beta = self._calculating_beta(dt_sig, i)
            self.gamma = dt_sig + dt_sig_2
            self.new_result[i] = self._calculating_new_result(i)

    def step_march(self):
        with open('fdm_result.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=' ',
                         quotechar='|', quoting=csv.QUOTE_MINIMAL)
            while self.cur_t < self.t_dom:
                self.cur_t = self.prev_t + self.dt
                self.calculate_boundary_conditions()
                self.calculate_inner_domain()
                for i in range(self.j):
                    csvwriter.writerow([self.x_values[i], self.prev_t,
                                        self.new_result[i]])
                self.old_result = self.new_result
                self.prev_t = self.cur_t

    def _calculating_beta(self, dt_sig, i):
        return self.dx**2 - (2.0 * dt_sig) + (self.dt * self.dx**2 *
                                              self.pde.zero_coeff(
                                                  self.prev_t,
                                                  self.x_values[i]))

    def _calculating_new_result(self, i):
        return (((self.alpha * self.old_result[i - 1]) +
                (self.beta * self.old_result[i]) +
                (self.gamma * self.old_result[i - 1])) /
                (self.dx * self.dt)) - (self.dt * self.pde.source_coeff(
                                        self.prev_t, self.x_values[i]))
