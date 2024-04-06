from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

import gymnasium as gym

import wandb
from loguru import logger

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

logger.info("starting training")

for i in range(20):
    logger.info(f"training step {i}")
    result = algo.train()

    if i % 5 == 0:
        checkpoint_dir = algo.save().checkpoint.path
        logger.debug(f"Checkpoint saved in directory {checkpoint_dir}")

logger.success(f"succesfully trained network")

env = gym.make("CartPole-v1", render_mode="rgb_array")

obs, info = env.reset()

logger.debug(f"started rendering")

for i in range(500):
    action = algo.compute_single_action(
        observation=obs,
    )
    obs, reward, done, truncated, _ = env.step(action)
    image = env.render()
    
    wandb.log({"example": wandb.Image(image)})

    if done:
        logger.success(f"done!")
        break