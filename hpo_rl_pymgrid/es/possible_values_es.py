from ray import tune

possible_values={
	#noise standrad deviation
	"noise_stdev":[0.02,0.01],
	"episodes_per_batch": [1],
	#Batch size
	"train_batch_size": [100],
	#exploitation/exploration
	"eval_prob":[0.02]
}

#Required for PBT
mutation_variables =  ["noise_stdev","episodes_per_batch","train_batch_size","eval_prob"]