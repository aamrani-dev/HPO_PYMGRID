import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import numpy as np
import sys

PATH_PYMGRID = "/home/amine/total/PYMGRID/src/"
sys.path.append(PATH_PYMGRID)
PATH_PYMGRID = "/home/amine/total/PYMGRID/src/pymgrid"
sys.path.append(PATH_PYMGRID)

import gym
from PYMGRID.src.pymgrid.Environments.pymgrid_cspla import MicroGridEnv
from PYMGRID.src.pymgrid import MicrogridGenerator as mg
import ray
from ray.tune import Trainable
from ray.rllib.agents.a3c import a2c
import os
from pymgrid.Environments.Environment import Environment
from hpo_rl_pymgrid.mytrainable import  MyClassAbstract

class MyClass(MyClassAbstract):
    def _setup(self, config):
        
        # THIS FOLLOWING LIST CONTAINS THE HYPERPARAMETERS WE TUNE: 
        #   lr, deep, wide, train_batch_size
        print("SETUP : ", str(config))
        gpu_ids = ray.get_gpu_ids()
        print("GPU_IDS=",gpu_ids)
        
        rl_algo_config = a2c.A2C_DEFAULT_CONFIG.copy()
        rl_algo_config["log_level"] = "WARN"

        #Generating microgrid
        self.nb_steps = 24*30
        env = mg.MicrogridGenerator(path=PATH_PYMGRID, nb_microgrid=1)
        env.generate_microgrid(verbose=True)
        mg0 = env.microgrids[0]
        mg0.set_horizon(self.nb_steps)  # TODO: check if it is usefull     
        rl_algo_config["env_config"] = {"microgrid":mg0}

        # dqn parameters and hyperparameters
        rl_algo_config["soft_horizon"] = self.nb_steps
        rl_algo_config["num_workers"] = config["num_workers"]
        rl_algo_config["lr"] = config["lr"]
        rl_algo_config["train_batch_size"] = config["train_batch_size"]
        rl_algo_config["gamma"] = config["gamma"]
        rl_algo_config["min_iter_time_s"] = 5
        #os.environ["CUDA_VISIBLE_DEVICES"]='-1'


        # GPU CONFIGURATION

        # rl_algo_config["num_gpus"] = 1 # cannot allow to get <1 gpus
        # rl_algo_config["num_gpus_per_worker"]=1
        # rl_algo_config["num_workers"] = 1
        # rl_algo_config["eager"] = False

        self.trainer = a2c.A2CTrainer(env=MicroGridEnv, config=rl_algo_config)

        print("SETUP END")