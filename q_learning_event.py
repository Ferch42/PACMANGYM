from env import PacmanEnv
from button_env import ButtonEnv
import numpy as np
import time


ALPHA = 0.1
GAMMA = 0.99
DECAY_RATE = 0.9999
EPSILON = 1

N_EPISODES = 10_000


q = {}

def Q(obs, event):

	global q

	#s = 0
	#for i in range(len(obs)):
	#	s+= (obs[i]+10)*(20**i)
	if (obs,event) not in q:
		q[(obs,event)] = np.zeros(4)

	return q[(obs,event)]

def main():

	global EPSILON,q
	env = ButtonEnv()
	obs = env.reset()
	#w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = []
	times = []

	for ep in range(N_EPISODES):

		s, _= env.reset()

		# initial state = 210
		#print(s)
		t = 0
		r_total = 0
		while(t<500):

			# 1 => 6 => 1 => 3
			# E-greedy
			#a = np.argmax(Q(s))
			a = np.random.randint(4)
			
			ss, reward, done, info = env.step(a)

			r = 0
			if info['E'] == 1:

				r = 0
				# FINAL STATE = 106
				#print(ss)

			if info['E'] == 3:

				r = 0
				# FINAL STATE = 306
				#print(ss)

			if info['E'] == 6:

				r = 0
				# FINAL STATE = 313
				#print(ss)


			#if info['E'] == 3:
			#	r= -1
			#print(reward)
			r_total += r
			#q_target = r + GAMMA*(q(w,ss).max())*(1-r!=0)

			#q_a = q(w,ss)[a]
			
			#print((q_target- q_a))
			# Update rule
			#w[a] = w[a] + ALPHA* (q_target- q_a) * s
			for e in (1,3,6):

				if info['E'] == e:
					r = 1
				else:
					r = 0

				Q(s, e)[a] = Q(s, e)[a] + ALPHA * (r + GAMMA*np.max(Q(ss, e)) - Q(s, e)[a])

			s = ss
			t+=1

			

		rewards.append(r_total)
		times.append(t)
		EPSILON = max(EPSILON*DECAY_RATE,0.1)

		if Q(210, 1).max()> 0 and Q(106, 6).max() > 0 and Q(313, 1).max() > 0 and Q(106, 3).max()> 0:
			
			print(f'DONE EPISODE: {ep}')
			break

		if ep%100==0:

			print(f'EPISODE: {ep}')
			print(f'REWARD AVG: {np.mean(rewards[-100:])}')
			print(f'TIMESTEP AVG: {np.mean(times[-100:])}')
			print(EPSILON)
	

if __name__ == '__main__':
	main()