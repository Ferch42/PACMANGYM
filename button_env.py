import os
import time
import numpy as np
from LTL import prog

clear = lambda: os.system('cls')

# Goal definition
T = ("UNTIL", "TRUE", ("AND", "TOUCHED", "FLAG"))
GOAL = ("UNTIL", "TRUE", ("AND", T, ("NEXT", T) )) 


class ButtonEnv():


	def __init__(self):
	# Configuration Vars
		self.SIZE = 10
		self.AGENT_POS = (self.SIZE/2,self.SIZE/2)
		self.GREEN_BTN_POS = (self.SIZE-1,6)
		self.RED_BTN_POS = (6,self.SIZE-6)
		self.BLUE_BTN_POS = (6,self.SIZE-2)
		self.flag = False
		self.GOAL = GOAL

	def reset(self):

		self.AGENT_POS = (self.SIZE/2,self.SIZE/2)
		self.PACMAN_POS = (0,0)
		self.flag = bool(np.random.randint(2))
		self.GOAL = GOAL
		return self.get_factored_representation()
		

	def render(self):

		horizontal_wall_str = '#'*(self.SIZE+2)
		
		print(f'FLAG = {self.flag}')
		print(horizontal_wall_str)
		
		for i in range(self.SIZE):
			line_str = '#'
			for j in range(self.SIZE):

				if i == self.AGENT_POS[0] and j == self.AGENT_POS[1]:
					line_str+='A'
				elif i == self.GREEN_BTN_POS[0] and j == self.GREEN_BTN_POS[1]:
					line_str+='G'
				elif i == self.RED_BTN_POS[0] and j == self.RED_BTN_POS[1]:
					line_str+='R'
				elif i == self.BLUE_BTN_POS[0] and j == self.BLUE_BTN_POS[1]:
					line_str+='B'

				else:
					line_str+= ' '

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

		# Move the pacman
		event = 0
		
		if self.AGENT_POS == self.GREEN_BTN_POS:
			# First event
			event = 1
			print(1)
			
		if self.AGENT_POS == self.RED_BTN_POS:
			# Second event
			event = 2
			print(2)

		if self.AGENT_POS == self.BLUE_BTN_POS:
			# Second event
			event = 3
			print(3)

		reward = 0
		P = self.get_propositions(event)

		self.GOAL = prog(P, self.GOAL)

		return self.get_factored_representation(),reward, type(self.GOAL) == bool, {'GOAL': self.GOAL, 'P': P, 'E': event} 


	def get_factored_representation(self):

		# AGENT coordinates
		AGENT_X, AGENT_Y = self.AGENT_POS
		# GREEN BTN coordinates
		GREEN_BTN_X, GREEN_BTN_Y = self.GREEN_BTN_POS
		# RED BTN coordinates
		RED_BTN_X, RED_BTN_Y = self.RED_BTN_POS
		# BLUE BTN coordinates
		BLUE_BTN_X, BLUE_BTN_Y = self.BLUE_BTN_POS

		return np.array([0,0,0])

	def get_propositions(self, event):

		P = set()
		
		if event==1:
			P.add('F1')

		return P


if __name__ == '__main__':
	
	env = ButtonEnv()

	for i in range(1000):

		env.render()
		time.sleep(0.5)
		env.step(np.random.randint(4))
		#print(env.reset())
		#reward, P = env.step(int(input()))
		#print(f'reward = {reward}')
		#print(f'P = {P}')
		#print(env.step(int(input())))