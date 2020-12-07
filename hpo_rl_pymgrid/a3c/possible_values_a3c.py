import argparse
from ray import tune

possible_values={
 	"lr": tune.loguniform(0.0001, 0.1),
    "train_batch_size": [167, 336, 720, 1000], 
    "gamma": [0.98, 0.99, 0.995],
    "num_workers":[10, 15, 20]
}

#Required for PBT
mutation_variables =  ["lr","train_batch_size","gamma"]
