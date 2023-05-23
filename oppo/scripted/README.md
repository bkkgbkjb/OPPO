# OFFLINE PREFERENCE-GUIDED POLICY OPTIMIZATION

This is source code for training an OFFLINE PREFERENCE-GUIDED POLICY OPTIMIZATION(OPPO) agent, with scripted preference

## Commands

### Install

#### Method 1 (Manual)
```shell
# in a virtual environment
pip install -r requirements.txt

# then install d4rl(https://github.com/Farama-Foundation/D4RL) following their instructions
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl

```

#### Method 2 (Docker)

```shell
cd docker

docker build -t oppo:scripted .
```


### Training
```shell
cd gym
# in a virtualenv
python experiment.py -h
```
