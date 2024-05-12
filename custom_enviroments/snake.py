import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

SNAKE_LEN_GOAL = 30

def collision_with_apple(apple_position, score):
	apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
	score += 1
	return apple_position, score

def collision_with_boundaries(snake_head):
	if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
		return 1
	else:
		return 0

def collision_with_self(snake_position):
	snake_head = snake_position[0]
	if snake_head in snake_position[1:]:
		return 1
	else:
		return 0


class SnekEnv(gym.Env):

	def __init__(self,  render_mode=None, size=5):
		super(SnekEnv, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
		self.action_space = spaces.Discrete(4)
		# Example for using image as input (channel-first; channel-last also works):
		# self.observation_space = spaces.Box(low=-500, high=500,
		# 									shape=(5+SNAKE_LEN_GOAL,), dtype=np.float32)
		self.observation_space = spaces.Box(low=-500, high=500,
									shape=(5+2*SNAKE_LEN_GOAL,), dtype=np.float32)

	def step(self, action):
		self.prev_actions.append(action)
		# cv2.imshow('a',self.img) 
		# cv2.waitKey(1)
		self.img = np.zeros((500,500,3),dtype='uint8')
		# Display Apple
		cv2.rectangle(self.img,(self.apple_position[0],self.apple_position[1]),(self.apple_position[0]+10,self.apple_position[1]+10),(0,0,255),3)
		# Display Snake
		for position in self.snake_position:
			cv2.rectangle(self.img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)
		

		button_direction = action
		# Change the head position based on the button direction
		if button_direction == 1:
			self.snake_head[0] += 10
		elif button_direction == 0:
			self.snake_head[0] -= 10
		elif button_direction == 2:
			self.snake_head[1] += 10
		elif button_direction == 3:
			self.snake_head[1] -= 10

		apple_reward = 0
		# Increase Snake length on eating apple
		if self.snake_head == self.apple_position:
			self.apple_position, self.score = collision_with_apple(self.apple_position, self.score)
			self.snake_position.insert(0,list(self.snake_head))
			apple_reward = 1000

		else:
			self.snake_position.insert(0,list(self.snake_head))
			self.snake_position.pop()
		
		# On collision kill the snake and print the score
		if collision_with_boundaries(self.snake_head) == 1 or collision_with_self(self.snake_position) == 1:
			font = cv2.FONT_HERSHEY_SIMPLEX
			self.img = np.zeros((500,500,3),dtype='uint8')
			cv2.putText(self.img,'Your Score is {}'.format(self.score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
			# cv2.imshow('a',self.img)
			self.done = True

		euclidean_dist_to_apple = np.linalg.norm(np.array(self.snake_head) - np.array(self.apple_position))
		
		reward = 100/(1+euclidean_dist_to_apple)

		reward += apple_reward

		if self.done:
			reward = -1000
		info = {}


		head_x = self.snake_head[0]
		head_y = self.snake_head[1]

		snake_length = len(self.snake_position)
		apple_delta_x = self.apple_position[0] - head_x
		apple_delta_y = self.apple_position[1] - head_y

		# create observation:
		new_observation = []

		for position in self.snake_position:
			new_observation.append(position[0])
			new_observation.append(position[1])

		for _ in range(2 + 2*SNAKE_LEN_GOAL - len(new_observation)):
			new_observation.append(0)
		new_observation.append(apple_delta_x)
		new_observation.append(apple_delta_y)
		new_observation.append(snake_length)

		observation = [head_x, head_y, self.apple_position[0], self.apple_position[1], 0] + list(self.prev_actions)
		observation = np.array(observation)
		new_observation = np.array(new_observation)
	

		return new_observation, reward, self.done, False, info

	def render(self):
		return self.img

	def reset(self, *, seed=None, options=None):
		self.img = np.zeros((500,500,3),dtype='uint8')
		# Initial Snake and Apple position
		self.snake_position = [[250,250],[240,250],[230,250]]
		self.apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
		self.score = 0
		self.prev_button_direction = 1
		self.button_direction = 1
		self.snake_head = [250,250]

		self.prev_reward = 0

		self.done = False

		head_x = self.snake_head[0]
		head_y = self.snake_head[1]

		snake_length = len(self.snake_position)
		apple_delta_x = self.apple_position[0] - head_x
		apple_delta_y = self.apple_position[1] - head_y

		self.prev_actions = deque(maxlen = SNAKE_LEN_GOAL)  # however long we aspire the snake to be
		for i in range(SNAKE_LEN_GOAL):
			self.prev_actions.append(-1) # to create history

		new_observation = []

		for position in self.snake_position:
			new_observation.append(position[0])
			new_observation.append(position[1])

		for _ in range(2 + 2*SNAKE_LEN_GOAL - len(new_observation)):
			new_observation.append(0)

		new_observation.append(apple_delta_x)
		new_observation.append(apple_delta_y)
		new_observation.append(snake_length)
		
		observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
		observation = np.array(observation)

		# create observation:
		observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length] + list(self.prev_actions)
		observation = np.array(observation)
		info = {}
		new_observation = np.array(new_observation)

		return new_observation, info

