from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms import Algorithm
from ray.tune.logger import pretty_print

import gymnasium as gym

import numpy as np
import cv2

import wandb
from loguru import logger

ENV = "MountainCar-v0"
SIZE = (600, 400)


def render_model(algorithm: Algorithm):
    result = cv2.VideoWriter('test.webm',  
                            cv2.VideoWriter_fourcc(*'VP90'), 
                            50, SIZE) 

    env = gym.make(ENV, render_mode="rgb_array")

    obs, info = env.reset()

    logger.debug(f"started rendering")

    frames = []
    image = None

    for i in range(1000):
        action = algorithm.compute_single_action(
            observation=obs,
            explore=False
        )
        obs, reward, done, truncated, _ = env.step(action)
        image = env.render()
        
        image_array = np.asanyarray(image, dtype=np.uint8).reshape(400 ,600 ,3)
        result.write(image_array)
        frames.append(image_array)

        if done:
            logger.success(f"done!")
            break

    result.release()

    wandb.log({"video": wandb.Video("test.webm", format="mp4")})


def main():
    wandb.init(
        # set the wandb project where this run will be logged
        project="ray_test",
        sync_tensorboard=True
    )

    algo = (
        DQNConfig()
        .rollouts(num_rollout_workers=8)
        .resources(num_gpus=0)
        .environment(env=ENV)
        .build()
    )

    logger.info("starting training")

    for i in range(120):
        logger.info(f"training step {i}")
        result = algo.train()
        if i % 5 == 0:
            checkpoint_dir = algo.save().checkpoint.path
            logger.debug(f"Checkpoint saved in directory {checkpoint_dir}")
            render_model(algo)

    logger.success(f"succesfully trained network")
    render_model(algo)
    wandb.finish()

if __name__ == "__main__":
    main()