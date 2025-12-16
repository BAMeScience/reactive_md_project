# reactive_md_project
Performing reactions in MD simulations

Currently only for one specific reaction: LiPF6 decomposition

To execute the code run:
PYTHONPATH=~/programs/reactive_md_project \
python -m reactive_md.main \
  --data /scratch/run1/mixture.data \
  --settings /scratch/run1/mixture.in.settings \
  --steps 10000
