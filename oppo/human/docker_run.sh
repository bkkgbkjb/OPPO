cd gym
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin
# nohup tensorboard --logdir /app/oppo/gym/runs --bind_all --port 8888 &>/dev/null &

export CUDA_VISIBLE_DEVICES=0
python3 experiment.py --env Lift --dataset mh --seed 1111 --force-save-model True &>Lm1111.log &
sleep 2
python3 experiment.py --env Lift --dataset mh --seed 2222 --force-save-model True &>Lm2222.log &
sleep 2
python3 experiment.py --env Lift --dataset mh --seed 3333 --force-save-model True &>Lm3333.log &
sleep 2

export CUDA_VISIBLE_DEVICES=1
python3 experiment.py --env Lift --dataset ph --seed 1111 --force-save-model True &>Lp1111.log &
sleep 2
python3 experiment.py --env Lift --dataset ph --seed 2222 --force-save-model True &>Lp2222.log &
sleep 2
python3 experiment.py --env Lift --dataset ph --seed 3332 --force-save-model True &>Lp3333.log &
sleep 2


export CUDA_VISIBLE_DEVICES=2
python3 experiment.py --env Can --dataset mh --seed 1111 --force-save-model True &>Cm1111.log &
sleep 2
python3 experiment.py --env Can --dataset mh --seed 2222 --force-save-model True &>Cm2222.log &
sleep 2
python3 experiment.py --env Can --dataset mh --seed 3333 --force-save-model True &>Cm3333.log &
sleep 2

export CUDA_VISIBLE_DEVICES=3
python3 experiment.py --env Can --dataset ph --seed 1111 --force-save-model True &>Cp1111.log &
sleep 2
python3 experiment.py --env Can --dataset ph --seed 2222 --force-save-model True &>Cp2222.log &
sleep 2
python3 experiment.py --env Can --dataset ph --seed 3333 --force-save-model True &>Cp3333.log &
wait
