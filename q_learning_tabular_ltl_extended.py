from env import PacmanEnv
from letter_env import LetterEnv
import numpy as np
import time
from ltl_graph_generator import *


ALPHA = 0.1
GAMMA = 0.99
DECAY_RATE = 0.99995
EPSILON = 1

N_EPISODES = 100_000

q = {}

FORMULA_VALUE_DICT = {True : 0 , 
					  ('UNTIL', 'TRUE', 'P') : 1, 
					  ('UNTIL', 'TRUE', ('AND', 'B', ('UNTIL', 'TRUE', 'P'))): 0.9, 
					  ('UNTIL', 'TRUE', ('AND', 'P', ('UNTIL', 'TRUE', ('AND', 'B', ('UNTIL', 'TRUE', 'P'))))): 0.81}
					  
REWARD_SHAPPING = True


letters = ['G', 'P', 'B', 'O', 'Y', 'R']

goal_list = []
for l1 in letters:
	for l2 in letters:

		for l3 in letters:
			
			if l1 != l2 and l2!=l3 and l1!=l3:
				g_formula = eval(f"('UNTIL', 'TRUE', ('AND', '{l1}', ('UNTIL', 'TRUE', ('AND', '{l2}', ('UNTIL', 'TRUE', '{l3}')))))")
				FORMULA_VALUE_DICT[g_formula] = 0.81

				f_formula = eval(f"('UNTIL', 'TRUE', ('AND', '{l2}', ('UNTIL', 'TRUE', '{l3}')))")
				FORMULA_VALUE_DICT[f_formula] = 0.81

				h_formula = eval(f"('UNTIL', 'TRUE', '{l3}')")
				FORMULA_VALUE_DICT[h_formula] = 1
			


print(FORMULA_VALUE_DICT)


def Q(obs, g):

	global q

	#s = 0
	#for i in range(len(obs)):
	#	s+= (obs[i]+10)*(20**i)
	s = (obs, g)
	#print(s)
	if s not in q:
		q[s] = np.zeros(4) 

	return q[s]

def main():

	global EPSILON,q
	env = LetterEnv()
	obs = env.reset()
	print(env.GOAL)
	#w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = []
	times = []
	avg_rewards = []
	avg_timesteps = []

	for ep in range(N_EPISODES):

		s, info = env.reset()
		goal = info['GOAL']
		#print(f"SOLVING GOAL: {goal} ")

		
		t = 0
		r_total = 0
		while(True):

			# E-greedy
			a = np.argmax(Q(s, goal))
			if np.max(Q(s,goal))==0 or np.random.uniform()>EPSILON:
				a = np.random.randint(4)

			#print('in state', s,sigma, goal)
			ss, reward, done, info = env.step(a)
			next_goal = info['GOAL']


			r_total += reward

			r = reward

			if REWARD_SHAPPING:
				r = r +  0.9* FORMULA_VALUE_DICT[next_goal] - FORMULA_VALUE_DICT[goal]

			Q(s, goal)[a] = Q(s, goal)[a] + ALPHA * (r + (1- int(done))*GAMMA*np.max(Q(ss, next_goal)) - Q(s, goal)[a])

			s = ss
			goal = next_goal

			t+=1

			if done:
				#print('halolo')
				#print(f"DONE IN {np.mean(times[-100:])}")
				break

		rewards.append(r_total)
		times.append(t)
		EPSILON = max(EPSILON*DECAY_RATE,0.1)
		if ep%100==0:

			print(f'EPISODE: {ep}')
			print(f'REWARD AVG: {np.mean(rewards[-100:])}')
			avg_rewards.append(np.mean(rewards[-100:]))
			print(f'TIMESTEP AVG: {np.mean(times[-100:])}')
			avg_timesteps.append(np.mean(times[-100:]))
			print(EPSILON)



			cc = 0
			global q
			print('----------')
			for k in q:
				if q[k].sum()!= 0 :
					cc+=1
			#print(sorted(q.keys()))
			print(f"len_{len(q)}")
			print(f"Q_{cc}")
		"""
		if(np.mean(rewards[-100:])>0.9):

			print(f"DONE IN {ep} EPISODES")
			print(f"MEAN TIMESTEP =  {np.mean(times[-100:])}")
			break
		"""
	
	s, info = env.reset()
	goal = info['GOAL']

	with open('rewards.txt', 'w+') as f:
		f.write(str(avg_rewards))


	with open('timesteps.txt', 'w+') as f:
		f.write(str(avg_timesteps))
	



if __name__ == '__main__':
	main()