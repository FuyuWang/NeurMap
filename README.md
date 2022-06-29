# NeurMap

NeurMap a deep reinforcement learning-based DNN mapper, 
to search the optimized mapping strategies for spatial accelerators.
This repository contains the source code for NeurMap.

### Setup ###
* Download the NeurMap source code
* Create virtual environment through anaconda
```
conda create --name NeurMapEnv python=3.8
conda activate NeurMapEnv
```
* Install packages
   
```
pip install -r requirements.txt
```

* Install [MAESTRO](https://github.com/maestro-project/maestro.git)
```
python build.py
```

### Run NeurMap ###

* Run NeurMap on TPU
```
./run_tpu.sh
```

* Run NeurMap on Eyeriss
```
./run_eyeriss.sh
```

