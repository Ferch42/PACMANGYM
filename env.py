import os
import time
import numpy as np
from LTL import prog

clear = lambda: os.system('cls')

# Goal definition
T = ("UNTIL", "TRUE", ("AND", "TOUCHED", "FLAG"))
GOAL = ("UNTIL", "TRUE", ("AND", T, ("NEXT", T) )) 


class PacmanEnv():


	def __init__(self):
	# Configuration Vars
		self.SIZE = 20
		self.AGENT_POS = (self.SIZE/2,self.SIZE/2)
		self.PACMAN_POS = (0,0)
		self.GREEN_BTN_POS = (self.SIZE-1,0)
		self.RED_BTN_POS = (0,self.SIZE-1)
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
				elif i == self.PACMAN_POS[0] and j == self.PACMAN_POS[1]:
					line_str+='P'
				elif i == self.GREEN_BTN_POS[0] and j == self.GREEN_BTN_POS[1]:
					line_str+='G'
				elif i == self.RED_BTN_POS[0] and j == self.RED_BTN_POS[1]:
					line_str+='R'

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
		pacman_target = self.RED_BTN_POS
		if not self.flag:
			pacman_target = self.AGENT_POS

		if np.random.uniform() < 0.5:
			dx, dy = self.get_pacman_action(pacman_target)

			self.PACMAN_POS = (self.PACMAN_POS[0] + dx, self.PACMAN_POS[1] + dy)
			self.PACMAN_POS = (max(min(self.SIZE-1,self.PACMAN_POS[0]),0), max(min(self.SIZE-1,self.PACMAN_POS[1]),0))

		event = 0
		
		if self.AGENT_POS == self.GREEN_BTN_POS:
			# First event
			event = 1
			self.flag = True
			
		if self.PACMAN_POS == self.RED_BTN_POS:
			# Second event
			event = 2
			self.flag = False

		reward = 0
		P = self.get_propositions()
		if self.PACMAN_POS == self.AGENT_POS:
			# Third event someone touched someone
			event = 3
			if self.flag:
				reward = 1
			else:
				reward = -1
			P =  self.get_propositions()
			
			self.AGENT_POS = (self.SIZE/2,self.SIZE/2)
			self.PACMAN_POS = (0,0)


		self.GOAL = prog(P, self.GOAL)

		return self.get_factored_representation(),reward, type(self.GOAL) == bool, {'GOAL': self.GOAL, 'P': P, 'E': event} 


	def get_factored_representation(self):

		# AGENT coordinates
		AGENT_X, AGENT_Y = self.AGENT_POS
		# PACMAN coordinates
		PACMAN_X, PACMAN_Y = self.PACMAN_POS
		# GREEN BTN coordinates
		GREEN_BTN_X, GREEN_BTN_Y = self.GREEN_BTN_POS
		# RED BTD coordinates
		RED_BTN_X, RED_BTN_Y = self.RED_BTN_POS

		return np.array([PACMAN_X - AGENT_X, PACMAN_Y - AGENT_Y, GREEN_BTN_X- AGENT_X, GREEN_BTN_Y - AGENT_Y, RED_BTN_X - AGENT_X, RED_BTN_Y - AGENT_Y, int(self.flag)])

	def get_propositions(self):

		P = set()

		if self.PACMAN_POS == self.AGENT_POS:
			P.add('TOUCHED')
		
		if self.flag:
			P.add('FLAG')

		return P
	

	def get_pacman_action(self, target):

		AGENT_X, AGENT_Y = target
		PACMAN_X, PACMAN_Y = self.PACMAN_POS

		if AGENT_X != PACMAN_X:
			# Move in the horizontal direction
			dx = AGENT_X - PACMAN_X
			return (dx/abs(dx),0)

		if AGENT_Y != PACMAN_Y:
			# Move in the vertical direction
			dy = AGENT_Y - PACMAN_Y
			return (0,dy/abs(dy))

		else:
			return (0,0)




if __name__ == '__main__':
	
	env = PacmanEnv()

	for i in range(1000):

		env.render()
		time.sleep(0.5)
		env.step(np.random.randint(4))
		#print(env.reset())
		#reward, P = env.step(int(input()))
		#print(f'reward = {reward}')
		#print(f'P = {P}')
		#print(env.step(int(input())))