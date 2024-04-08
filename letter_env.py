import os
import time
import numpy as np
from LTL import prog
import random

clear = lambda: os.system('cls')

P_GOAL = ("UNTIL", "TRUE", ('AND', 'P', ('UNTIL', 'TRUE', 'B')))

O_GOAL = ('UNTIL', 'TRUE', ('AND', 'P', ('UNTIL', 'TRUE', ('AND', 'B', ('UNTIL', 'TRUE', 'P')))))

FINAL_GOAL = O_GOAL

letters = ['G', 'P', 'B', 'O', 'Y', 'R']

goal_list = []
for l1 in letters:
    for l2 in letters:
        for l3 in letters:

            if l1 != l2 and l2!=l3 and l1!=l3:
                goal_list.append(eval(f"('UNTIL', 'TRUE', ('AND', '{l1}', ('UNTIL', 'TRUE', ('AND', '{l2}', ('UNTIL', 'TRUE', '{l3}')))))"))

#goal_list = goal_list[0:5]
print(len(goal_list))


class LetterEnv():


	def __init__(self):
	# Configuration Vars
		self.SIZE = 11
		self.AGENT_POS = (5,5)
		# UPPER SIDE BUTTONS
		self.GREEN_BTN_POS = (0,0)
		self.RED_BTN_POS = (0,5)
		self.BLUE_BTN_POS = (0,10)
		# LOWER SIDE BUTTONS
		self.PURPLE_BTN_POS = (10,0)
		self.ORANGE_BTN_POS = (10,5)
		self.YELLOW_BTN_POS = (10,10)
		
		self.flag = False
		self.sigma = set()
		self.GOAL = FINAL_GOAL
		self.propositions =  set(("LIGHT", "SOUND", "MONKEY"))
		self.index = 0

	def reset(self):

		self.AGENT_POS = (int(self.SIZE/2),int(self.SIZE/2))
		self.PACMAN_POS = (0,0)
		self.flag = bool(np.random.randint(2))
		self.GOAL = goal_list[self.index]
		self.index = (self.index+1)%(len(goal_list))
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
		event = "_"

		P = set()
		
		if self.AGENT_POS == self.GREEN_BTN_POS:
			# First event
			#event = 1
			P.add("G")
			#print(1)
			
		if self.AGENT_POS == self.RED_BTN_POS:
			# Second event
			#event = 2
			P.add("R")
			#print(2)

		if self.AGENT_POS == self.BLUE_BTN_POS:
			# Third event
			#event = 3
			P.add("B")
			#print(3)

		if self.AGENT_POS == self.PURPLE_BTN_POS:
			# Fourth event
			#event = 4
			P.add("P")
			#print(1)
			
		if self.AGENT_POS == self.ORANGE_BTN_POS:
			# Fifth event
			#event = 5
			P.add("O")
			#print(2)

		if self.AGENT_POS == self.YELLOW_BTN_POS:
			# Sixth event
			P.add("Y")
			#event = 6

		reward = 0

		if self.sigma != P:
			# Salient event occurred
			event = (str(self.sigma) + str(P)).replace('{', '').replace("}",'').replace('set()', '_').replace("'", "")

		self.sigma = P

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




if __name__ == '__main__':
	
	env = LetterEnv()

	for i in range(100):

		env.render()
		time.sleep(0.5)
		state, reward, done, info = env.step(int(input(">")))
		print(state, info,reward)
		#print(env.reset())
		#reward, P = env.step(int(input()))
		#print(f'reward = {reward}')
		#print(f'P = {P}')
		#print(env.step(int(input())))

	
