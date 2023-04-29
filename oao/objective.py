#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol, Union

from ax.service.ax_client import ObjectiveProperties


class Objective(Protocol):
    def __call__(self, parameters: dict) -> Union[float, dict]:
        ...

    def evaluate(self, parameters: dict) -> Union[float, dict]:
        ...


class NoiselessFormattedObjective:
    def __init__(
        self, objective: Objective, name: str, properties_kw: dict = {"minimize": True}
    ):
        self.objective = objective
        self.name = name
        self.properties = ObjectiveProperties(**properties_kw)

    def __call__(self, parameters: dict) -> dict:
        return self.evaluate(parameters)

    def evaluate(self, parameters: dict) -> dict:
        return {self.name: (self.objective.evaluate(parameters), None)}


class NoisyFormattedObjective:
    def __init__(
        self, objective: Objective, name: str, properties_kw: dict = {"minimize": True}
    ):
        self.objective = objective
        self.name = name
        self.properties = ObjectiveProperties(**properties_kw)

    def __call__(self, parameters: dict) -> dict:
        return self.evaluate(parameters)

    def evaluate(self, parameters: dict) -> dict:
        return {self.objective_name: self.objective.evaluate(parameters)}
