import argparse
from ray import tune

parser = argparse.ArgumentParser()
parser.add_argument("--dqn_type", required=True)

args, unknown = parser.parse_known_args()
args = vars(args)

try:
	if args["dqn_type"] not in ["dueling", "double_q"]: 
		raise Exception("Error: expected dqn type among [dueling, double_q] but received " + args["dqn_type"])
except:
	raise

possible_values={
 	"lr": tune.loguniform(0.0001, 0.1),
    "wide": [24, 60, 180, 360, 720],
    "deep": [1, 3, 5, 8],
    "type": args["dqn_type"],
    "train_batch_size": [167, 336, 720, 2160 ], 
    "gamma": [0.98, 0.99, 0.995]
}

#Required for PBT
mutation_variables =  ["lr","train_batch_size","gamma"]