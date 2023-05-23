import setup
from setup import seed
from datetime import datetime
from experiment import experiment
from ray import tune
from args import args
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.optuna import OptunaSearch
from os import path
from ray.tune.logger import TBXLoggerCallback
from ray.tune.progress_reporter import CLIReporter
import numpy as np
import math

seed(args.seed)

def trail(params):
    import setup
    import d4rl
    
    from ray.tune.utils import wait_for_gpu
    wait_for_gpu(target_util= 1 - 1/15)

    experiment("gym-experiment-tune", {**vars(args), **params})



params = {
    "pref_loss_ratio": tune.quniform(0.05, 0.25, 0.05),
    "phi_norm_loss_ratio": tune.quniform(0.05, 0.25, 0.05),
    "in_tune": True,
    "w_lr": tune.choice([1e-3, 5e-4, 5e-3, 1e-2]),
    "dirpath": path.abspath('./')
}

tune.run(
    trail,
    metric="eval/return",
    mode="max",
    search_alg=OptunaSearch(),
    # scheduler=AsyncHyperBandScheduler(
    #     max_t=100, grace_period=math.floor(100 / 3)
    # ),
    name=f"ray-tune-{args.env}-{args.dataset}-{datetime.now().strftime('%m-%d:%H:%M:%S:%s')}",
    resources_per_trial={"cpu": 1 / 3, "gpu": 1 / 15},
    max_concurrent_trials=4,
    config=params,
    num_samples=12,
    verbose=1,
    progress_reporter=CLIReporter(max_report_frequency=300),
    sync_config=tune.SyncConfig(syncer=None),
    local_dir=path.abspath("./ray"),
    callbacks=[TBXLoggerCallback()]
)