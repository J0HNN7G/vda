#!/bin/sh

# do this first
#source $HOME/Documents/diss/intuitive_physics/env/bin/activate


# experiment 1
python3 data_gen.py --data-folder predict_mass/ --rand-seed 43913 --num-sims 1 --mode "predict_mass" --num-balls 7

# experiment 2
python3 data_gen.py --data-folder predict_friction/ --rand-seed 53924 --num-sims 1 --mode "predict_friction" --num-balls 7

# experiment 3
python3 data_gen.py --data-folder predict_material/ --rand-seed 55947 --num-sims 1 --mode "predict_friction_mass_dependent" --num-balls 7

# experiment 4
python3 data_gen.py --data-folder predict_mass_friction/ --rand-seed 34638 --num-sims 1 --mode "predict_friction_mass_independent" --num-balls 7
