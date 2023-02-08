#!/bin/sh

source ~/Documents/diss/intuitive_physics/env/bin/activate

python generate_data.py --data-folder data/same_vis_same_phys/train --rand-seed 1415 --num-sims 9000 --same-vis --same-phys
python generate_data.py --data-folder data/same_vis_same_phys/val --rand-seed 2643 --num-sims 1000 --same-vis --same-phys
python generate_data.py --data-folder data/same_vis_same_phys/test --rand-seed 9502 --num-sims 200 --same-vis --same-phys

python generate_data.py --data-folder data/diff_vis_diff_phys/train --rand-seed 1415 --num-sims 9000
python generate_data.py --data-folder data/diff_vis_diff_phys/val --rand-seed 2643 --num-sims 1000
python generate_data.py --data-folder data/diff_vis_diff_phys/test --rand-seed 9502 --num-sims 200

python generate_data.py --data-folder data/same_vis_diff_phys/train --rand-seed 1415 --num-sims 9000 --same-vis
python generate_data.py --data-folder data/same_vis_diff_phys/val --rand-seed 2643 --num-sims 1000 --same-vis
python generate_data.py --data-folder data/same_vis_diff_phys/test --rand-seed 9502 --num-sims 200 --same-vis
