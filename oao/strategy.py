#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Optional

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy


@dataclass
class GenerationStepConfig:

    def construct(self) -> GenerationStep:
        raise NotImplementedError


@dataclass
class GenerationStrategyConfig:
    steps: list[GenerationStep]
    name: Optional[str] = None

    def construct(self) -> GenerationStrategy:
        return GenerationStrategy(
            steps=[step.construct() for step in self.steps], name=self.name
        )
