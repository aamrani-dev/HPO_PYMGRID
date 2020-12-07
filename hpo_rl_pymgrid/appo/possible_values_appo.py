from ray import tune

possible_values={
	"vtrace": True,
	"clip_param": [0.2, 0.3, 0.4], 
	"train_batch_size": [167, 336, 400],
	"lr": tune.loguniform(0.000001, 0.05),
	"num_workers": 10,
	# number of passes to make over each train batch
	"num_sgd_iter":[1]
}
#Required for PBT
mutation_variables =  ["clip_param","lr","train_batch_size","num_sgd_iter"]