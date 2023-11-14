import os
import time
import numpy as np
from LTL import prog

clear = lambda: os.system('cls')

# Goal definition
BREAD_COKE_HAM = ("UNTIL", "TRUE", ("AND", "BREAD",("AND", "COKE", "HAM")))
BREAD_HAM = ("UNTIL", "TRUE", ("AND", "BREAD", "HAM"))
BREAD = ("UNTIL", "TRUE", "BREAD")

GOAL = ("UNTIL", "TRUE", ("AND", BREAD, ("UNTIL", "TRUE", ("AND", BREAD_HAM, ("UNTIL", "TRUE", BREAD_COKE_HAM))))) 

COKE_GOAL = ("UNTIL", "TRUE", "COKE")

class ButtonEnv():


	def __init__(self):
	# Configuration Vars
		self.SIZE = 20
		self.AGENT_POS = (int(self.SIZE/2),int(self.SIZE/2))
		# UPPER SIDE BUTTONS
		self.GREEN_BTN_POS = (int(self.SIZE/3),int(self.SIZE/4))
		self.RED_BTN_POS = (int(self.SIZE/3),int(self.SIZE/2))
		self.BLUE_BTN_POS = (int(self.SIZE/3),3*int(self.SIZE/4))
		# LOWER SIDE BUTTONS
		self.PURPLE_BTN_POS = (int(2*self.SIZE/3),int(self.SIZE/4))
		self.ORANGE_BTN_POS = (int(2*self.SIZE/3),int(self.SIZE/2))
		self.YELLOW_BTN_POS = (int(2*self.SIZE/3),3*int(self.SIZE/4))
		
		self.flag = False
		self.sigma = set()
		self.GOAL = COKE_GOAL

	def reset(self):

		self.AGENT_POS = (int(self.SIZE/2),int(self.SIZE/2))
		self.PACMAN_POS = (0,0)
		self.flag = bool(np.random.randint(2))
		self.GOAL = COKE_GOAL
		self.sigma = set()
		#return self.get_factored_representation()
		return self.get_discrete_representation(), {'GOAL': self.GOAL, 'P': self.sigma, 'E': 0}
		

	def render(self):

		horizontal_wall_str = '#'*(self.SIZE+2)
		
		#print(f'FLAG = {self.flag}')
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
				elif i == self.PURPLE_BTN_POS[0] and j == self.PURPLE_BTN_POS[1]:
					line_str+='P'
				elif i == self.ORANGE_BTN_POS[0] and j == self.ORANGE_BTN_POS[1]:
					line_str+='O'
				elif i == self.YELLOW_BTN_POS[0] and j == self.YELLOW_BTN_POS[1]:
					line_str+='Y'


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
			#print(1)
			
		if self.AGENT_POS == self.RED_BTN_POS:
			# Second event
			event = 2
			#print(2)

		if self.AGENT_POS == self.BLUE_BTN_POS:
			# Second event
			event = 3
			#print(3)

		reward = 0
		P = self.get_propositions(event)

		self.GOAL = prog(P, self.GOAL)

		if self.GOAL == True:
			reward = 1

		#return self.get_factored_representation(),reward, type(self.GOAL) == bool, {'GOAL': self.GOAL, 'P': P, 'E': event} 
		return self.get_discrete_representation(),reward, type(self.GOAL) == bool, {'GOAL': self.GOAL, 'P': P, 'E': event} 


	def get_factored_representation(self):

		# AGENT coordinates
		AGENT_X, AGENT_Y = self.AGENT_POS
		# GREEN BTN coordinates
		GREEN_BTN_X, GREEN_BTN_Y = self.GREEN_BTN_POS
		# RED BTN coordinates
		RED_BTN_X, RED_BTN_Y = self.RED_BTN_POS
		# BLUE BTN coordinates
		BLUE_BTN_X, BLUE_BTN_Y = self.BLUE_BTN_POS

		return np.array([self.AGENT_POS])

	def get_discrete_representation(self):

		# AGENT coordinates
		AGENT_X, AGENT_Y = self.AGENT_POS

		return AGENT_X + AGENT_Y * self.SIZE

	def get_propositions(self, event):

		# Definition of Lt function
		if event ==1: 
			self.sigma.add('BREAD')
		
		if event ==2: 
			self.sigma.add('HAM')

		if self.sigma == {'BREAD', 'HAM'} and event ==3: 
			self.sigma.add('COKE')

		return self.sigma.copy()


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