from env import PacmanEnv
from button_env import ButtonEnv
import numpy as np
import time


ALPHA = 0.001
GAMMA = 0.99
DECAY_RATE = 0.999
EPSILON = 1

N_EPISODES = 10_000


q = {}

def Q(obs):

	global q

	#s = 0
	#for i in range(len(obs)):
	#	s+= (obs[i]+10)*(20**i)
	if obs not in q:
		q[obs] = np.zeros(4)

	return q[obs]

def main():

	global EPSILON,q
	env = ButtonEnv()
	obs = env.reset()
	#w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = []
	times = []

	for ep in range(N_EPISODES):

		s, _= env.reset()

		t = 0
		r_total = 0
		while(t<500):

			# E-greedy
			a = np.argmax(Q(s))
			if np.random.uniform() < EPSILON:
				a = np.random.randint(4)
			
			ss, reward, done, info = env.step(a)

			r = 0
			if info['E'] == 1:

				r = 1
			#if info['E'] == 3:
			#	r= -1
			#print(reward)
			r_total += r
			#q_target = r + GAMMA*(q(w,ss).max())*(1-r!=0)

			#q_a = q(w,ss)[a]
			
			#print((q_target- q_a))
			# Update rule
			#w[a] = w[a] + ALPHA* (q_target- q_a) * s

			Q(s)[a] = Q(s)[a] + ALPHA * (r + GAMMA*np.max(Q(ss)) - Q(s)[a])

			s = ss
			t+=1

			if r==1:
				#print('halolo')
				break

		rewards.append(r_total)
		times.append(t)
		EPSILON = max(EPSILON*DECAY_RATE,0.1)
		if ep%5==0:

			print(f'EPISODE: {ep}')
			print(f'REWARD AVG: {np.mean(rewards[-100:])}')
			print(f'TIMESTEP AVG: {np.mean(times[-100:])}')
			print(EPSILON)
	
	s, _ = env.reset()

	for _ in range(200):
		env.render()
		a = np.argmax(Q(s))

		ss, reward, done, info = env.step(a)
		print(reward)
		s = ss			
		time.sleep(0.33)









if __name__ == '__main__':
	main()