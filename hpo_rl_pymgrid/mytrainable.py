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

from ray.tune import Trainable
import ray
import os
from pymgrid.Environments.Environment import Environment
from abc import ABC 

class MyClassAbstract(Trainable, ABC):
    def _train(self):
        print("TRAIN")
        info=self.trainer.train()
        reward=info["episode_reward_mean"]
        nb_days=float(self.nb_steps)/24
        mean_reward_by_day=reward/nb_days
        out={"reward": mean_reward_by_day}
        print("TRAIN END")
        return out

    def score(self,db_split_name):
        print("not implemented yet")
        #info=self.trainer
        #s=self._evaluate()
        return -1

    def save_checkpoint(self, checkpoint_dir):
        print("SAVE")
        file_path = checkpoint_dir
        print(checkpoint_dir)
        file_path = self.trainer.save(file_path)
        print(checkpoint_dir)
        print("SAVE END")
        return file_path

    def get_restore_path(self,path):
        """
        :param path: dir/checkpoint_3/ should return the model into /dir/checkpoint_3/checkpoint_3/checkpoint-3/
        :return: path to the model as string
        """
        path=os.path.normpath(path) # Eliminate last slash e.g. dir/checkpoint_3/ to dir/checkpoint_3
        check_id=path.split(os.sep)[-1] #e.g. checkpoint_3
        check_id_under=check_id.replace("_","-") #e.g. checkpoint-3
        checkpoint_leaf_path=os.path.join(path,check_id,check_id_under)
        return checkpoint_leaf_path

    def _restore(self, path):
        print("RESTORE")
        checkpoint_leaf_path=self.get_restore_path(path)
        print(checkpoint_leaf_path)
        print(path)
        self.trainer.restore(path)
        print("RESTORE END")


    def __softmax(self,x):
        exp_x=x.astype(np.float64)
        y=exp_x / np.sum(exp_x)
        return y.astype(np.float32)

    def predict(self,observation,stochasticity=True):
        state=[]

        agent=self.trainer.workers._local_worker
        policy_name="default_policy"
        clip_action = agent.policy_config["clip_actions"]
        proba_name="q_values" #"logits"

        preprocessed = agent.preprocessors[policy_name].transform(observation)
        filtered_obs = agent.filters[policy_name](preprocessed, update=False)
        result = agent.get_policy(policy_name).compute_single_action(filtered_obs, state, None, None, None, clip_actions=clip_action)

        raw_probs=result[2][proba_name]
        probs = self.__softmax(raw_probs)

        if stochasticity:
            action = result[0]
        else:
            action = np.argmax(probs)
        return action, probs
