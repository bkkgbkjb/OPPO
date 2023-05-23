export CUDA_VISIBLE_DEVICES=0

nohup python experiment.py --env hopper --dataset medium &>hm.log &

nohup python experiment.py --env hopper --dataset medium-expert &>hme.log &

nohup python experiment.py --env hopper --dataset medium-replay &>hmr.log &

nohup python experiment.py --env hopper --dataset random &>hr.log &

export CUDA_VISIBLE_DEVICES=1
nohup python experiment.py --env walker2d --dataset medium &>wm.log &

nohup python experiment.py --env walker2d --dataset medium-expert &>wme.log &

nohup python experiment.py --env walker2d --dataset medium-replay &>wmr.log &

nohup python experiment.py --env walker2d --dataset random &>wr.log &

export CUDA_VISIBLE_DEVICES=2
nohup python experiment.py --env halfcheetah --dataset medium &>cm.log &

nohup python experiment.py --env halfcheetah --dataset medium-expert &>cme.log &

nohup python experiment.py --env halfcheetah --dataset medium-replay &>cmr.log &

nohup python experiment.py --env halfcheetah --dataset random &>cr.log &