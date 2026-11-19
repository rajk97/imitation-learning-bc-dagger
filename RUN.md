# RUN.md — commands to reproduce all results

Run from the repository root with the `rob831` conda env (`conda activate rob831`). `--no_gpu` is used because the pinned torch 1.12.1 has no kernels for this GPU.

## Table 1 — expert returns (Q1.2)
```bash
python -c "import pickle,glob,numpy as np; [print(f, np.mean(r:=[p['reward'].sum() for p in pickle.load(open(f,'rb'))]), np.std(r)) for f in sorted(glob.glob('rob831/expert_data/*.pkl'))]"
```

## Table 2 — BC on Ant-v2 and Humanoid-v2 (Q1.3)
```bash
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --num_agent_train_steps_per_iter 5000 --eval_batch_size 5000 --video_log_freq -1 --no_gpu
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name bc_humanoid --n_iter 1 --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl --num_agent_train_steps_per_iter 5000 --eval_batch_size 5000 --video_log_freq -1 --no_gpu
```

## Figure 1 — BC training-steps sweep on Ant-v2 (Q1.4)
```bash
for N in 250 500 1000 2000 3000 5000 10000; do
  python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant_steps$N --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --num_agent_train_steps_per_iter $N --eval_batch_size 5000 --video_log_freq -1 --no_gpu
done
```

## Figure 2 — DAgger on Ant-v2 and Humanoid-v2 (Q2.2)
```bash
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name dagger_ant --n_iter 10 --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --num_agent_train_steps_per_iter 5000 --batch_size 1000 --eval_batch_size 5000 --video_log_freq -1 --no_gpu
python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Humanoid.pkl --env_name Humanoid-v2 --exp_name dagger_humanoid --n_iter 10 --do_dagger --expert_data rob831/expert_data/expert_data_Humanoid-v2.pkl --num_agent_train_steps_per_iter 5000 --batch_size 1000 --eval_batch_size 5000 --video_log_freq -1 --no_gpu
```

## Plots
```bash
python plot_results.py   # writes figures/fig1_bc_sweep.png and figures/fig2_dagger.png
```

## Videos
```bash
# train and save BC policies, then render expert vs. student rollouts to videos/
for ENV in Ant Humanoid; do
  python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/$ENV.pkl --env_name $ENV-v2 --exp_name video_bc_${ENV,,} --n_iter 1 --expert_data rob831/expert_data/expert_data_$ENV-v2.pkl --num_agent_train_steps_per_iter 5000 --eval_batch_size 5000 --video_log_freq -1 --save_params --no_gpu
done
python make_videos.py
```
