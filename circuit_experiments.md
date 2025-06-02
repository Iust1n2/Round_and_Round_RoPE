# Circuit Experiments

1. We use `auto_circuit_experiment` from `circuit_discovery_utils.py` as a plug-in for our training. We want to run two circuits: 1. after phase A and 2. after phase B. 

2. Using `compute_circuit_overlap` we can get the overlap between the two. 

3. If we run for 10 epochs we have 20 circuits and 10 overlap statistics. We take the node or edge overlaps for circuit A and B and plot them against the task accuracy.