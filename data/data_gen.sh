#!/bin/sh

# do this first
#source $HOME/Documents/diss/intuitive_physics/env/bin/activate

python3 data_gen.py --data-folder same_vis_same_phys/train --rand-seed 1415 --num-sims 9000 --same-vis --same-phys
python3 data_gen.py --data-folder same_vis_same_phys/val --rand-seed 2643 --num-sims 9000 --same-vis --same-phys
python3 data_gen.py --data-folder same_vis_same_phys/test --rand-seed 9502 --num-sims 9000 --same-vis --same-phys

python3 data_gen.py --data-folder diff_vis_same_phys/train --rand-seed 1415 --num-sims 9000 --same-phys
python3 data_gen.py --data-folder diff_vis_same_phys/val --rand-seed 2643 --num-sims 9000 --same-phys
python3 data_gen.py --data-folder diff_vis_same_phys/test --rand-seed 9502 --num-sims 9000 --same-phys

python3 data_gen.py --data-folder same_vis_diff_phys/train --rand-seed 3384 --num-sims 9000 --same-vis
python3 data_gen.py --data-folder same_vis_diff_phys/val --rand-seed 2008 --num-sims 1000 --same-vis
python3 data_gen.py --data-folder same_vis_diff_phys/test --rand-seed 1004 --num-sims 200 --same-vis

python3 data_gen.py --data-folder diff_vis_diff_phys/train --rand-seed 6046 --num-sims 9000
python3 data_gen.py --data-folder diff_vis_diff_phys/val --rand-seed 8082 --num-sims 1000
python3 data_gen.py --data-folder diff_vis_diff_phys/test --rand-seed 1567 --num-sims 200