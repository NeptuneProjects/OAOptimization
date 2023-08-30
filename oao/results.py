# -*- coding: utf-8 -*-

from ax.service.ax_client import AxClient
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render
import numpy as np
import pandas as pd

from oao.optimizer import Optimizer


def _extract_raw_results(client: AxClient) -> pd.DataFrame:
    left_df = client.get_trials_data_frame()
    right_df = pd.DataFrame(
        [trial.arm.parameters for trial in client.experiment.trials.values()]
    )
    return pd.concat([left_df, right_df], axis=1)


def _extract_times(time: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"time": time})


def _format_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["trial_index"])
    return df


def _get_best_trial(client: AxClient, minimize: bool = False) -> pd.DataFrame:
    objective_means = np.array(
        [[trial.objective_mean for trial in client.experiment.trials.values()]]
    )
    best_values, best_trials = _get_max_with_index(
        -objective_means if minimize else objective_means
    )
    best_parameters = [client.experiment.trials[i].arm.parameters for i in best_trials]
    df_values = pd.DataFrame(
        {
            "best_trial": best_trials,
            "best_value": -best_values if minimize else best_values,
        }
    )
    df_params = pd.DataFrame(best_parameters).rename(
        columns={k: f"best_{k}" for k in best_parameters[0].keys()}
    )
    return pd.concat([df_values, df_params], axis=1)


def _get_max_with_index(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_arr = np.maximum.accumulate(arr, axis=1).squeeze()
    max_index = np.zeros_like(max_arr, dtype=int)

    for i in range(1, len(max_arr)):
        if max_arr[i] > max_arr[i - 1]:
            max_index[i] = i
        else:
            max_index[i] = max_index[i - 1]

    return max_arr, max_index


def get_results(
    client: AxClient, times: list[float], minimize: bool = False
) -> pd.DataFrame:
    df_raw = _extract_raw_results(client)
    df_best = _get_best_trial(client, minimize=minimize)
    df_time = _extract_times(times)
    return _format_results(pd.concat([df_raw, df_best, df_time], axis=1))


def plot_results(df: pd.DataFrame, m) -> None:
    best_objective_plot = optimization_trace_single_method(
        y=df["best_value"].values[np.newaxis, ...],
        optimum=1.0,
    )
    render(best_objective_plot)
    render(plot_contour(m, param_x="rec_r", param_y="src_z", metric_name="bartlett"))
