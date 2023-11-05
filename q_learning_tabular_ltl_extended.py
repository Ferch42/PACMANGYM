from env import PacmanEnv
from button_env import ButtonEnv
import numpy as np
import time


ALPHA = 0.001
GAMMA = 0.99
DECAY_RATE = 0.9999
EPSILON = 1

N_EPISODES = 10_000


q = {}

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
	#w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = []
	times = []

	for ep in range(N_EPISODES):

		s, info = env.reset()

		sigma  = info['P']
		goal = info['GOAL']
		#print('========================')
		#print('initial state', s,sigma, goal)
		t = 0
		r_total = 0
		while(t<500):

			# E-greedy
			a = np.argmax(Q(s,sigma, goal))
			if np.random.uniform() < EPSILON:
				a = np.random.randint(4)

			#print('in state', s,sigma, goal)
			ss, reward, done, info = env.step(a)
			next_sigma = info['P']
			next_goal = info['GOAL']

			r = reward

			r_total += r

			Q(s,sigma, goal)[a] = Q(s,sigma, goal)[a] + ALPHA * (r + (1- int(done))*GAMMA*np.max(Q(ss, next_sigma, next_goal)) - Q(s,sigma, goal)[a])

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
				if q[k].sum()> 0 :
					cc+=1
			#print(sorted(q.keys()))
			print(f"len_{len(q)}")
			print(f"Q_{cc}")
	
	s, info = env.reset()
	sigma  = info['P']
	goal = info['GOAL']
	
	for _ in range(200):
		
		a = np.argmax(Q(s, sigma, goal))

		ss, reward, done, info = env.step(a)
		
		env.render()

		next_sigma = info['P']
		next_goal = info['GOAL']
		s = ss
		sigma = next_sigma
		goal = next_goal	
		
		if done:
			break		
		
		time.sleep(0.33)









if __name__ == '__main__':
	main()