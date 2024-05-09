import sys
import os

sys.path.append(os.getcwd())

print(sys.path)

from custom_enviroments.snake import SnekEnv

from clearml import Task, Logger

Task.init(
    # set the wandb project where this run will be logged
    project_name="test_project/gym_enviroments", task_name='snake'
)


env = SnekEnv()
episodes = 50

for episode in range(episodes):
	done = False
	obs = env.reset()
	while True:
		random_action = env.action_space.sample()
		print("action",random_action)
		obs, reward, done, info = env.step(random_action)
		print('reward',reward)