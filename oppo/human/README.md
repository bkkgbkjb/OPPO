# OFFLINE PREFERENCE-GUIDED POLICY OPTIMIZATION

This is source code for training an OFFLINE PREFERENCE-GUIDED POLICY OPTIMIZATION(OPPO) agent, with human-labeled preferences

## Commands

### Install

#### Method 1 (Manual)
```shell
# in a virtual environment
pip install -r requirements.txt

# if you encounter errors about gssapi install&build, please run following commands:
sudo apt install libkrb5-dev

# install robomimic and robosuite according to Preference Transformer
pip install git+https://github.com/ARISE-Initiative/robosuite.git@v1.3
pip install git+https://github.com/ARISE-Initiative/robomimic.git

# if encounter errors about numpy.ndarray size changed, please run following commands:
# and ignore errors about conflicts of mujoco-py version
pip install mujoco-py==2.0.2.9 --no-cache-dir --no-binary :all: --no-build-isolation # see more at https://github.com/openai/mujoco-py/issues/607

# then install d4rl(https://github.com/Farama-Foundation/D4RL) following their instructions
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl

```

#### Method 2 (Docker)

```shell
cd docker 
docker build -t oppo:human .
```


### Training
```shell
cd gym
# in a virtualenv
python experiment.py -h
```
