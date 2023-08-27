# -*- coding: utf-8 -*-


from typing import Optional, Union


class GridStrategy:
    def __init__(
        self,
        num_trials: Union[int, list[int]],
        max_parallelism: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        self.num_trials = num_trials
        self.max_parallelism = max_parallelism

    def __call__(self, *args, **kwargs):
        return lambda: self.num_trials
