from env import PacmanEnv
from button_env import ButtonEnv
import numpy as np
import time
from ltl_graph_generator import *


ALPHA = 0.1
GAMMA = 0.99
DECAY_RATE = 0.9999
EPSILON = 1

N_EPISODES = 20_000

INFORMATIVE_GOAL_REWARD_SHAPPING_FLAG = True
MODEL_REWARD_SHAPPING_FLAG = False

q = {}



ENV_STATE_DICT = {(): 0, ('LIGHT',): 1, ('LIGHT', 'SOUND'): 2, ('SOUND',): 3, ('MONKEY', 'SOUND'): 4, ('LIGHT', 'MONKEY', 'SOUND'): 5, ('LIGHT', 'MONKEY'): 6, ('MONKEY',): 7}
TRANSITIONS = {(3, 4), (5, 4), (2, 2), (1, 0), (7, 7), (6, 5), (4, 5), (3, 3), (5, 6), (0, 1), (1, 2), (2, 1), (6, 7), (7, 6), (3, 2), (4, 4), (5, 5), (0, 0), (1, 1), (2, 3), (6, 6)}
	
def Q(obs, sigma, g):

	global q

	#s = 0
	#for i in range(len(obs)):
	#	s+= (obs[i]+10)*(20**i)
	s = (obs, str(sigma), g)
	#print(s)
	if s not in q:
		q[s] = np.zeros(4)

	return q[s]

def main():

	global EPSILON,q
	env = ButtonEnv()
	obs = env.reset()
	print(env.GOAL)
	#w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = []
	times = []

	print(INFORMATIVE_GOAL_REWARD_SHAPPING_FLAG)
	for ep in range(N_EPISODES):

		s, info = env.reset()

		sigma  = tuple(sorted(info['P']))
		goal = info['GOAL']
		#print('========================')
		#print('initial state', s,sigma, goal)
		goal_V = value_iteration_graph(*generate_graph(goal, env.propositions))
		#print(goal_V)
		model_goal_V = value_iteration_ltl_graph(*generate_ltl_env_graph(tuple(sigma), goal, ENV_STATE_DICT, TRANSITIONS ))
		#print(goal_V)
		#break

		
		t = 0
		r_total = 0
		while(t<500):

			# E-greedy
			a = np.argmax(Q(s,sigma, goal))
			if np.random.uniform() < EPSILON:
				a = np.random.randint(4)

			#print('in state', s,sigma, goal)
			ss, reward, done, info = env.step(a)
			next_sigma = tuple(sorted(info['P']))
			next_goal = info['GOAL']


			r_total += reward

			r = reward

			if INFORMATIVE_GOAL_REWARD_SHAPPING_FLAG:

				r = r + GAMMA * goal_V[next_goal] - goal_V[goal]

			elif MODEL_REWARD_SHAPPING_FLAG:

				r = r + GAMMA * model_goal_V[(next_goal, next_sigma)] - model_goal_V[(goal, sigma)]
				

			Q(s,sigma, goal)[a] = Q(s,sigma, goal)[a] + ALPHA * (r + (1- int(done))*GAMMA*np.max(Q(ss, next_sigma, next_goal)) - Q(s,sigma, goal)[a])
			#print( Q(s,sigma, goal)[a], r, r + (1- int(done))*GAMMA*np.max(Q(ss, next_sigma, next_goal)) - Q(s,sigma, goal)[a])
			#print(w[a])
			s = ss
			sigma = next_sigma
			goal = next_goal

			t+=1

			if done:
				#print('halolo')
				break

		rewards.append(r_total)
		times.append(t)
		EPSILON = max(EPSILON*DECAY_RATE,0.1)
		if ep%100==0:

			print(f'EPISODE: {ep}')
			print(f'REWARD AVG: {np.mean(rewards[-100:])}')
			print(f'TIMESTEP AVG: {np.mean(times[-100:])}')
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

		if(np.mean(rewards[-100:])>0.9):

			print(f"DONE IN {ep} EPISODES")
			print(f"MEAN TIMESTEP =  {np.mean(times[-100:])}")
			break
	
	s, info = env.reset()
	sigma  = tuple(sorted(info['P']))
	goal = info['GOAL']
	
	for _ in range(200):
		
		a = np.argmax(Q(s, sigma, goal))

		ss, reward, done, info = env.step(a)
		
		env.render()

		next_sigma = tuple(sorted(info['P']))
		next_goal = info['GOAL']
		s = ss
		sigma = next_sigma
		goal = next_goal	
		
		if done:
			break		
		
		time.sleep(0.33)









if __name__ == '__main__':
	main()