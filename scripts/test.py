from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="ray_test",
    sync_tensorboard=True
)

algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=8)
    .resources(num_gpus=0)
    .environment(env="CartPole-v1")
    .build()
)

for i in range(60):
    result = algo.train()
    algo.validate_env()
    print(pretty_print(result))

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        print(f"Checkpoint saved in directory {checkpoint_dir}")