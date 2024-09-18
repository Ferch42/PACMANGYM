import os
import time
import numpy as np
from LTL import prog
import random

clear = lambda: os.system('cls')

letters = set({'A', 'B', 'C', 'D', 'E', 'F'})


class Patch():

	def __init__(self, bottom_coordinates, top_coordinates, sigma):

		self.BOTTOM_COORDINATES = bottom_coordinates
		self.TOP_COORDINATES = top_coordinates
		self.sigma = sigma

	def contains_coordinate(self, coordinates):
	
		coord_x, coord_y = coordinates
		bottom_x, bottom_y = self.BOTTOM_COORDINATES
		top_x, top_y = self.TOP_COORDINATES

		return coord_x>= bottom_x and coord_x <top_x and coord_y>= bottom_y and coord_y<top_y


class PatchEnv():


	def __init__(self):
	# Configuration Vars
		self.SIZE = 50
		self.AGENT_POS = (int(self.SIZE/2), int(self.SIZE/2))
		self.patches = [Patch((0,0), (20,20), set({'A'})), Patch((30,30), (40,40), set({'B'}))]
		self.sigma = set()
		self.GOAL = ("UNTIL", "TRUE", "A")
		self.patch_index = None

	
	def reset(self):

		self.AGENT_POS = (int(self.SIZE/2),int(self.SIZE/2))
		
		return self.get_discrete_representation(), {'GOAL': self.GOAL, 'P': self.sigma, 'E': 0}
		

	def render(self):

		horizontal_wall_str = '#'*(self.SIZE+2)
		
		#print(f'FLAG = {self.flag}')
		print(horizontal_wall_str)
		
		for i in range(self.SIZE):
			line_str = '#'
			for j in range(self.SIZE):
				
				flag = False
				for patch_number, patch in enumerate(self.patches):
					
					if i == self.AGENT_POS[0] and j == self.AGENT_POS[1]:
						line_str+='A'
						flag = True
						break

					elif patch.contains_coordinate((i,j)):
						line_str+= str(patch_number)
						flag = True
						
				if not flag:
					line_str +=' '

			line_str +='#'
			print(line_str)
		print(horizontal_wall_str)

	def step(self,action):

		
		AGENT_X, AGENT_Y = self.AGENT_POS
		# horizontal axis
		if action == 0:
			self.AGENT_POS = (AGENT_X+1,AGENT_Y)
		if action == 1:
			self.AGENT_POS = (AGENT_X-1,AGENT_Y)
		if action == 2:
			self.AGENT_POS = (AGENT_X,AGENT_Y+1)
		if action == 3:
			self.AGENT_POS = (AGENT_X,AGENT_Y-1)

		self.AGENT_POS = (max(min(self.SIZE-1,self.AGENT_POS[0]),0), max(min(self.SIZE-1,self.AGENT_POS[1]),0))

		patch_flag = False
		for patch_number, patch in enumerate(self.patches):

			if patch.contains_coordinate(self.AGENT_POS):
				self.patch_index = patch_number
				self.P = patch.sigma
				patch_flag = True
		
		if not patch_flag:
			self.P = set()
			self.patch_index = None

		reward = 0


		self.GOAL = prog(self.P, self.GOAL)

		if self.GOAL == True:
			reward = 1

		#return self.get_factored_representation(),reward, type(self.GOAL) == bool, {'GOAL': self.GOAL, 'P': P, 'E': event} 
		return self.get_discrete_representation(),reward, type(self.GOAL) == bool, {'GOAL': self.GOAL, 'P': self.P, 'E': 0} 


	def get_factored_representation(self):

		# AGENT coordinates
		AGENT_X, AGENT_Y = self.AGENT_POS
		
		return np.array([self.AGENT_POS])

	def get_discrete_representation(self):

		# AGENT coordinates
		AGENT_X, AGENT_Y = self.AGENT_POS

		return AGENT_X + AGENT_Y * self.SIZE




if __name__ == '__main__':
	
	env = PatchEnv()

	for i in range(100):

		env.render()
		time.sleep(0.5)
		state, reward, done, info = env.step(int(input(">")))
		print(state, info,reward)
		#clear()
		#print(env.reset())
		#reward, P = env.step(int(input()))
		#print(f'reward = {reward}')
		#print(f'P = {P}')
		#print(env.step(int(input())))

	
