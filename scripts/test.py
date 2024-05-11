import sys
import os

sys.path.append(os.getcwd())

from custom_enviroments.snake import SnekEnv

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.a2c.a2c import A2CConfig
from ray.rllib.algorithms.sac.sac import SACConfig

from ray.rllib.algorithms import Algorithm
from ray.tune.logger import pretty_print

import gymnasium as gym

import numpy as np
import cv2

from clearml import Task, Logger
from loguru import logger

ENV = SnekEnv
SIZE = (500, 500)
MAX_EPOCHS = 12000


def render_model(algorithm: Algorithm, iteration: int):
    result = cv2.VideoWriter('test.webm',  
                            cv2.VideoWriter_fourcc(*'VP90'), 
                            50, SIZE) 

    env = ENV()

    obs, info = env.reset()

    logger.debug(f"started rendering")

    frames = []
    image = None
    algorithm.evaluate
    for i in range(1000):
        action = algorithm.compute_single_action(
            observation=obs,
        )
        obs, reward, done, truncated, _ = env.step(action)
        image = env.render()
        
        image_array = np.asanyarray(image, dtype=np.uint8).reshape(500 ,500 ,3)
        result.write(image_array)
        frames.append(image_array)

        if done:
            logger.success(f"done!")
            break

    result.release()

    Logger.current_logger().report_media(
        title='autoput', 
        series='tada', 
        iteration=iteration,
        local_path="test.webm"
    )


def main():
    Task.init(
        # set the wandb project where this run will be logged
        project_name="test_project/gym_enviroments", task_name='Snake_PPO'
    )

    algo = (
        PPOConfig()
        .rollouts(num_rollout_workers=12)
        .resources(num_gpus=0)
        .training()
        .environment(env=ENV)
        .build()
    )

    logger.info("starting training")

    for i in range(MAX_EPOCHS):
        logger.info(f"training step {i}")
        result = algo.train()
        if i % 5 == 0:
            checkpoint_dir = algo.save().checkpoint.path
            logger.debug(f"Checkpoint saved in directory {checkpoint_dir}")
            render_model(algo, i)

    logger.success(f"succesfully trained network")
    render_model(algo, MAX_EPOCHS)

if __name__ == "__main__":
    main()