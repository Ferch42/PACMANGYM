from env import PacmanEnv
from letter_env import LetterEnv
import numpy as np
import time
from ltl_graph_generator import *


ALPHA = 0.1
GAMMA = 0.99
DECAY_RATE = 0.9999
EPSILON = 1

N_EPISODES = 20_000

q = {}


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

	for ep in range(N_EPISODES):

		s, info = env.reset()
		goal = info['GOAL']

		
		t = 0
		r_total = 0
		while(t<500):

			# E-greedy
			a = np.argmax(Q(s, goal))
			if np.random.uniform() < EPSILON:
				a = np.random.randint(4)

			#print('in state', s,sigma, goal)
			ss, reward, done, info = env.step(a)
			next_goal = info['GOAL']


			r_total += reward

			r = reward


			Q(s, goal)[a] = Q(s, goal)[a] + ALPHA * (r + (1- int(done))*GAMMA*np.max(Q(ss, next_goal)) - Q(s, goal)[a])

			s = ss
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
		"""
		if(np.mean(rewards[-100:])>0.9):

			print(f"DONE IN {ep} EPISODES")
			print(f"MEAN TIMESTEP =  {np.mean(times[-100:])}")
			break
		"""
	
	s, info = env.reset()
	goal = info['GOAL']
	
	for _ in range(200):
		
		a = np.argmax(Q(s, goal))

		ss, reward, done, info = env.step(a)
		
		env.render()

		next_goal = info['GOAL']
		s = ss
		goal = next_goal	
		
		if done:
			break		
		
		time.sleep(0.33)




if __name__ == '__main__':
	main()