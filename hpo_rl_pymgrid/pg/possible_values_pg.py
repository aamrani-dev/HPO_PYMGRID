from ray import tune

possible_values={
	"lr": tune.loguniform(0.00001, 0.1),
	"horizon": [720, 2160, 4320, 8640]
}
#Required for PBT
mutation_variables =  ["lr", "horizon"]