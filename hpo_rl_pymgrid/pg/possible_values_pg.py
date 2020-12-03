from ray import tune

possible_values={
	"lr": tune.loguniform(0.00001, 0.1),
}
#Required for PBT
mutation_variables =  ["lr"]