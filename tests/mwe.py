from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import branin
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import numpy as np

from ax.plot.slice import plot_slice
from ax.utils.notebook.plotting import render

from ax.core.batch_trial import BatchTrial

N_WARMUP = 5
q = 4
N_TOTAL = N_WARMUP + 10
N_RESTARTS = 20
N_SAMPLES = 128

def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(2)])
    return {"branin": (branin(x), None)}

gs = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=N_WARMUP,
            min_trials_observed=N_WARMUP,
            max_parallelism=N_WARMUP,
            model_kwargs={"seed": 2023},
            model_gen_kwargs={}
        ),
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=N_TOTAL - N_WARMUP,
            max_parallelism=3,
            model_kwargs={
                "surrogate": Surrogate(
                    botorch_model_class=SingleTaskGP,
                    mll_class=ExactMarginalLogLikelihood
                ),
                "botorch_acqf_class": qExpectedImprovement,
            },
            model_gen_kwargs={
                "model_gen_options": {
                    "optimizer_kwargs": {
                        "num_restarts": N_RESTARTS,
                        "raw_samples": N_SAMPLES
                    }
                }
            }
        )
    ]
)

ax_client = AxClient(generation_strategy=gs)
ax_client.create_experiment(
    name="branin_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [-5.0, 10.0],
            "value_type": "float",
            "log_scale": False
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 15.0],
            "value_type": "float",
            "log_scale": False
        }
    ],
    objectives={"branin": ObjectiveProperties(minimize=True)}
)

# Normal sequential loop
# for i in range(N_TOTAL):
#     parameters, trial_index = ax_client.get_next_trial()
#     ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

# Batch optimization loop
for _ in range(N_WARMUP):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))


num_optim = N_TOTAL - N_WARMUP
num_batches = np.ceil(num_optim / q).astype(int)

optim_num = 0
for batch_num in range(num_batches):
    batch_size = (
        q
        if batch_num < num_batches - 1
        else num_optim - batch_num * q
    )

    gen_run = ax_client._gen_new_generator_run(n=q)

    for arm in gen_run.arms:
        if optim_num == num_optim:
            break
        params, trial_index = ax_client.attach_trial(arm.parameters)
        ax_client.complete_trial(trial_index, evaluate(params))
        optim_num += 1
