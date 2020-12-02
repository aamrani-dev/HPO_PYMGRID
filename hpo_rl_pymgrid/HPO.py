from ray import tune
from ray.tune.schedulers import *
from ray.tune import suggest
from ray.tune.suggest import bayesopt
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import bohb
import ray
import time
import numpy as np
import os
from collections.abc import Iterable
from ray.tune.suggest import ConcurrencyLimiter

import sys

NB_GPU_BY_TRIAL=1
MEMORY=500* 1000*1000*1000


class CustomStopper:
    def __init__(self,max_iter):
        self.should_stop = False
        self.max_iter=max_iter
    def stop(self, trial_id, result):
        if not self.should_stop and result["training_iteration"] >= self.max_iter:
            self.should_stop = True
        return self.should_stop

def _hyperband_sched(hyperhyper_params):
    if "pruning_factor" in hyperhyper_params:
        reduction_factor=hyperhyper_params["pruning_factor"]
    else:
        reduction_factor=2
    return AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric=hyperhyper_params["metric"],
        mode=hyperhyper_params["mode"],
        max_t=hyperhyper_params["iter_max"],
        reduction_factor=reduction_factor
        )

def _tune_config(possible_values):
    config = {}
    for k, v in possible_values.items():
        if isinstance(v, Iterable):
            config[k] = tune.choice(v)
        else:
            config[k] = v
    return config

def _get_dim_desc(possible_values):
    dim_names=[]
    dim_limits=[]
    for k,v in possible_values.items():
        dim_names.append(k)

        if isinstance(v[0],str): # if categorical
            dim_limits.append(v)
        else: # elif real or integer
            dim_limits.append((np.min(v), np.max(v)))


    return dim_names, dim_limits


# Random Search Optimization using Tune
def RS(MyTrainableClass, results_folder, possible_values, hyperhyper_params):
    config = _tune_config(possible_values)

    stopper = {'training_iteration': hyperhyper_params['iter_max']}
    analysis = tune.run(
        local_dir=results_folder,
        name="exp_results",
        stop=stopper,
        run_or_experiment=MyTrainableClass,
        resources_per_trial={"gpu": NB_GPU_BY_TRIAL},
        num_samples=hyperhyper_params["samples"],
        config=config,
        reuse_actors=False,
        checkpoint_at_end=True)
    return analysis


# Hyberband 
def HB(MyTrainableClass, results_folder, possible_values, hyperhyper_params):
    config = _tune_config(possible_values)

    algo = HyperOptSearch(metric=hyperhyper_params["metric"], mode=hyperhyper_params["mode"])
    algo = ConcurrencyLimiter(algo, max_concurrent=hyperhyper_params["max_concurrent"])

    sched = _hyperband_sched(hyperhyper_params)
    stopper = {'training_iteration': hyperhyper_params['iter_max']}

    analysis = tune.run(
        local_dir=results_folder,
        name="exp_results",
        run_or_experiment=MyTrainableClass,
        search_alg=algo,
        scheduler=sched,
        config=config,
        num_samples=hyperhyper_params["samples"],
        stop=stopper,
        resources_per_trial={"gpu": NB_GPU_BY_TRIAL},
        checkpoint_at_end=True)
    
    return analysis

# Population Based Training
def PBT(MyTrainableClass, results_folder, possible_values, hyperhyper_params):
    mutation_variables = hyperhyper_params["mutation_variables"]

    # compute possible mutations
    possible_mutations = {}
    for var in mutation_variables:
        possible_mutations[var] = possible_values[var]

    config = _tune_config(possible_values)

    stopper = {'training_iteration': hyperhyper_params['iter_max']}
    sched = PopulationBasedTraining(
        time_attr="training_iteration",
        metric=hyperhyper_params["metric"],
        mode=hyperhyper_params["mode"],
        resample_probability= hyperhyper_params["resample_probability"],  # probability to draw new value
        hyperparam_mutations=possible_mutations,
        perturbation_interval = hyperhyper_params["perturbation_interval"])

    analysis = tune.run(
        reuse_actors=False,
        stop=stopper,
        local_dir=results_folder,
        name="exp",
        run_or_experiment=MyTrainableClass,
        scheduler=sched,
        resources_per_trial={"gpu": NB_GPU_BY_TRIAL},
        num_samples=hyperhyper_params["samples"],
        config=config,
        checkpoint_at_end=True)
    return analysis


#Bayesian Optimization mixed to Hyperband
def BOHB(MyTrainableClass, results_folder, possible_values, hyperhyper_params):
    # https://docs.ray.io/en/releases-0.8.2/tune-searchalg.html?highlight=BOHB#bohb
    # pip install hpbandster ConfigSpace

    # BOHB uses ConfigSpace for their hyperparameter search space
    from ray.tune.suggest.bohb import TuneBOHB
    import ConfigSpace as CS

    config_space = CS.ConfigurationSpace()

    for k, v in possible_values.items():
        csvar = CS.CategoricalHyperparameter(k, choices=v)
        config_space.add_hyperparameter(csvar)

    scheduler = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=hyperhyper_params["iter_max"],
        reduction_factor=hyperhyper_params["pruning_factor"],
        metric=hyperhyper_params["metric"], mode=hyperhyper_params["mode"])

    algo = TuneBOHB(config_space, max_concurrent=["max_concurrent"],
                    metric=hyperhyper_params["metric"], mode=hyperhyper_params["mode"])

    analysis = tune.run(
        local_dir=results_folder,
        name="exp",
        run_or_experiment=MyTrainableClass,
        search_alg=algo,
        scheduler=scheduler,
        resources_per_trial={"gpu": NB_GPU_BY_TRIAL},
        num_samples=hyperhyper_params["samples"],
        checkpoint_at_end=True)
    return analysis

class RAY_PROG_ABSTRACTION:
    def __init__(self, MyTrainableClass, is_multinode_mode):
        """
        :param RAY_PATH: <root>
        :param RAY_TASK_FOLDER_NAME: <root>/BO/
        :param TrainablePkg:
        :param is_multinode_mode:
        Tree:
        <RAY_PATH>/BO/
        <RAY_PATH>/set_ray_scripts/
        """
        #self.RAY_TASK_PATH = os.path.join(RAY_PATH  , RAY_TASK_FOLDER_NAME )
        self.init_out = None
        self.MyTrainableClass=MyTrainableClass

        if "RAY_HEAD_PATH" in os.environ:
            head_tcpip_path=os.environ["RAY_HEAD_PATH"]

        if is_multinode_mode and not os.path.exists(head_tcpip_path):
            raise ValueError("ERROR tcpip path not found in: ",head_tcpip_path)
    
        # INIT
        address = ""
        if is_multinode_mode:
            if os.path.exists(head_tcpip_path):
                with open(head_tcpip_path, "r") as f:
                    address = f.read()
        ray.shutdown()
        if address != "":
            print("HPC MODE. scheduler at address: ", address)
            #self.init_out = ray.init(address=address,object_store_memory=MEMORY)
            self.init_out = ray.init(address=address, ignore_reinit_error=True)
        else:
            print("SINGLE-NODE MODE")
            self.init_out = ray.init(local_mode=True)
    
        print("**** log ****")
        print(self.init_out)
        print(ray.cluster_resources())
        print("********")
    
    def RUN(self, out_folder, possible_values, config, strat_funct):
        """
        :param out_folder: e.g. "HB" save experiments in /src/HB/
        :param possible_values: dict hpo_name -> list of possible values
        :param config: HPO config
        :param strat_funct: HPO function pointer
        :return:
        """
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        start_time=time.time()
        analysis=strat_funct(self.MyTrainableClass, out_folder, possible_values, config)
        print("Exploration time: %0.2f sec." % round(time.time()-start_time,1))
        _procedure_when_finished(out_folder, analysis, config)

def _procedure_when_finished(results_folder, analysis, hyperhyper_params):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    filename_csv = "results.csv"

    best_config = analysis.get_best_config(metric=hyperhyper_params["metric"], mode=hyperhyper_params["mode"])
    print("BEST CONFIG: ", best_config)

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()
    with open(os.path.join(results_folder, filename_csv), 'a') as f:
        f.write(df.to_csv(header=True, index=False))

    best = analysis.get_best_trial(metric=hyperhyper_params["metric"], mode=hyperhyper_params["mode"])

    path = best.checkpoint.value
    # path=best.logdir+"/model"
    print("BEST: ", path)
