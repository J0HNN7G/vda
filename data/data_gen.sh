#!/bin/sh

# do this first
#source $HOME/Documents/diss/intuitive_physics/env/bin/activate

python3 data_gen.py --data-folder predict_friction/train --rand-seed 5924 --num-sims 9000 --mode "predict_friction"
python3 data_gen.py --data-folder predict_friction/val --rand-seed 6621 --num-sims 200 --mode "predict_friction"
python3 data_gen.py --data-folder predict_friction/test --rand-seed 7645 --num-sims 200 --mode "predict_friction"

python3 data_gen.py --data-folder predict_mass/train --rand-seed 4313 --num-sims 9000 --mode "predict_mass"
python3 data_gen.py --data-folder predict_mass/val --rand-seed 6212 --num-sims 200 --mode "predict_mass"
python3 data_gen.py --data-folder predict_mass/test --rand-seed 8911 --num-sims 200 --mode "predict_mass"

python3 data_gen.py --data-folder predict_friction_mass_independent/train --rand-seed 5947 --num-sims 9000 --mode "predict_friction_mass_independent"
python3 data_gen.py --data-folder predict_friction_mass_independent/val --rand-seed 2744 --num-sims 200 --mode "predict_friction_mass_independent"
python3 data_gen.py --data-folder predict_friction_mass_independent/test --rand-seed 5758 --num-sims 200 --mode "predict_friction_mass_independent"

python3 data_gen.py --data-folder predict_friction_mass_dependent/train --rand-seed 4504 --num-sims 9000 --mode "predict_friction_mass_dependent"
python3 data_gen.py --data-folder predict_friction_mass_dependent/val --rand-seed 3468 --num-sims 200 --mode "predict_friction_mass_dependent"
python3 data_gen.py --data-folder predict_friction_mass_dependent/test --rand-seed 4519 --num-sims 200 --mode "predict_friction_mass_dependent"

# python3 data_gen.py --data-folder same_vis_same_phys/train --rand-seed 1415 --num-sims 9000 --same-vis --same-phys
# python3 data_gen.py --data-folder same_vis_same_phys/val --rand-seed 2643 --num-sims 200 --same-vis --same-phys
# python3 data_gen.py --data-folder same_vis_same_phys/test --rand-seed 9502 --num-sims 200 --same-vis --same-phys

#python3 data_gen.py --data-folder diff_vis_same_phys/train --rand-seed 3096 --num-sims 9000 --same-phys
#python3 data_gen.py --data-folder diff_vis_same_phys/val --rand-seed 4713 --num-sims 200 --same-phys
#python3 data_gen.py --data-folder diff_vis_same_phys/test --rand-seed 6020 --num-sims 200 --same-phys

# python3 data_gen.py --data-folder same_vis_diff_phys/train --rand-seed 3384 --num-sims 9000 --same-vis
# python3 data_gen.py --data-folder same_vis_diff_phys/val --rand-seed 2008 --num-sims 200 --same-vis
# python3 data_gen.py --data-folder same_vis_diff_phys/test --rand-seed 1004 --num-sims 200 --same-vis

# python3 data_gen.py --data-folder diff_vis_diff_phys/train --rand-seed 6046 --num-sims 9000
# python3 data_gen.py --data-folder diff_vis_diff_phys/val --rand-seed 8082 --num-sims 200
# python3 data_gen.py --data-folder diff_vis_diff_phys/test --rand-seed 1567 --num-sims 200