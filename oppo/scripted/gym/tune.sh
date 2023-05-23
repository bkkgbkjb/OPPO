export CUDA_VISIBLE_DEVICES=1

nohup python tune.py --env hopper --dataset medium &>hm.log &
sleep 3

nohup python tune.py --env hopper --dataset medium-expert &>hme.log &
sleep 3

nohup python tune.py --env hopper --dataset medium-replay &>hmr.log &
sleep 3

# nohup python tune.py --env hopper --dataset random &>hr.log &

export CUDA_VISIBLE_DEVICES=2
nohup python tune.py --env walker2d --dataset medium &>wm.log &
sleep 3

nohup python tune.py --env walker2d --dataset medium-expert &>wme.log &
sleep 3

nohup python tune.py --env walker2d --dataset medium-replay &>wmr.log &
sleep 3

# nohup python tune.py --env walker2d --dataset random &>wr.log &

export CUDA_VISIBLE_DEVICES=3
nohup python tune.py --env halfcheetah --dataset medium &>cm.log &
sleep 3

nohup python tune.py --env halfcheetah --dataset medium-expert &>cme.log &
sleep 3

nohup python tune.py --env halfcheetah --dataset medium-replay &>cmr.log &
sleep 3

# nohup python tune.py --env halfcheetah --dataset random &>cr.log &