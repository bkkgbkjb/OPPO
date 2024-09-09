export CUDA_VISIBLE_DEVICES=0

nohup python experiment.py --env hopper --dataset medium-replay --seed 1111 &>logs/hmr0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-replay --seed 2222 &>logs/hmr100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-replay --seed 3333 &>logs/hmr200.log &
sleep 2

nohup python experiment.py --env hopper --dataset medium-expert --seed 1111 &>logs/hme0.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-expert --seed 2222 &>logs/hme100.log &
sleep 2
nohup python experiment.py --env hopper --dataset medium-expert --seed 3333 &>logs/hme200.log &
sleep 2

# nohup python experiment.py --env hopper --dataset random &>hr.log &

export CUDA_VISIBLE_DEVICES=2
nohup python experiment.py --env walker2d --dataset medium-expert --seed 1111 &>logs/wme0.log &
sleep 2
nohup python experiment.py --env walker2d --dataset medium-expert --seed 2222 &>logs/wme100.log &
sleep 2
nohup python experiment.py --env walker2d --dataset medium-expert --seed 3333 &>logs/wme200.log &
sleep 2

nohup python experiment.py --env walker2d --dataset medium-replay --seed 1111 &>logs/wmr0.log &
sleep 2
nohup python experiment.py --env walker2d --dataset medium-replay --seed 2222 &>logs/wmr100.log &
sleep 2
nohup python experiment.py --env walker2d --dataset medium-replay --seed 3333 &>logs/wmr200.log &
sleep 2
