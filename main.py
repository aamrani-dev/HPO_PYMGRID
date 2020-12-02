from hpo_rl_pymgrid import HPO 
import time
import ray
from ray import tune
import sys
import numpy as np
import argparse


# Construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--rl_algorithm", required=True)
parser.add_argument("--samples", required=True)
parser.add_argument("--iter_max", required=True)
parser.add_argument("--iter_min", required=True)
parser.add_argument("--hp_algo", required=True)
parser.add_argument("--resample_probability", required=False)
parser.add_argument("--perturbation_interval", required=False)
parser.add_argument("--max_concurrent", required=False)
parser.add_argument("--pruning_factor", required=False)

args, unknown = parser.parse_known_args()
args = vars(args)
print(args)
try:
	if args["rl_algorithm"] not in ["dqn"]:
		raise Exception("Requested RL algorithm is not supported")
	else:
		rl_algorithm = args["rl_algorithm"]
except:
	raise


# Search Space
if rl_algorithm == "dqn":
	from  hpo_rl_pymgrid.dqn import mytrainable_dqn as mytrainable
	from  hpo_rl_pymgrid.dqn import possible_values_dqn as search_space



#Params which will be used when calling tune.run method
common_config={
"samples":int(args["samples"]),
"metric":"reward",
"mode":"max",
"iter_max":int(args["iter_max"]),
"iter_min":int(args["iter_min"]),
"checkpoint_score_attr": "reward" 
}


algo = args["hp_algo"]

if algo == "PBT":
	if args["resample_probability"] != None:
		resample_probability = float(args["resample_probability"])
	else:
		resample_probability = 0.4
	if args["perturbation_interval"] != None:
		perturbation_interval = int(args["perturbation_interval"])
	else:
		perturbation_interval = 24
	
	config = {
	"resample_probability": resample_probability, 
	"perturbation_interval":perturbation_interval,
	"mutation_variables": search_space.mutation_variables
	}
	# copy common_config in config
	config.update(common_config)
	strat_funct = HPO.PBT

elif algo == "BOHB" or algo == "HB":
	if args["max_concurrent"] != None:
		max_concurrent = int(args["max_concurrent"])
	else:
		max_concurrent = 4

	if args["pruning_factor"] != None:
		pruning_factor = int(args["pruning_factor"])
	else:
		pruning_factor = 3

	config = {
	"max_concurrent":max_concurrent, 
	"pruning_factor":pruning_factor,
	}
	# copy common_config in config
	config.update(common_config)

	if algo == "HB":
		strat_funct = HPO.HB
	else:
		strat_funct = HPO.BOHB
else: 
	config = common_config
	strat_funct = HPO.RS

# Instantiate RAY
hpo=HPO.RAY_PROG_ABSTRACTION(mytrainable.MyClass, is_multinode_mode=False)
# Run hyperband
results_folder = "/tmp/"+algo+"/"
hpo.RUN(results_folder,search_space.possible_values, config, strat_funct)

