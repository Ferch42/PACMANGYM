from env import PacmanEnv
import numpy as np
import time


ALPHA = 0.001
GAMMA = 0.99
DECAY_RATE = 0.9999
EPSILON = 1

N_EPISODES = 10_000


def q(w, s):

	return np.sum(w*s, axis = 1)

def main():

	global EPSILON
	env = PacmanEnv()
	obs = env.reset()
	w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = []
	times = []
	for ep in range(N_EPISODES):

		s = env.reset()

		t = 0
		r_total = 0
		while(t<500):

			# E-greedy
			a = np.argmax(q(w,s))
			if np.random.uniform() < EPSILON:
				a = np.random.randint(4)

			ss, reward, done, info = env.step(a)

			r = 0
			if info['E'] == 1:

				r = 1
			if info['E'] == 3:
				r= -1

			r_total += r
			q_target = r + GAMMA*(q(w,ss).max())*(1-r!=0)

			q_a = q(w,ss)[a]
			
			#print((q_target- q_a))
			# Update rule
			w[a] = w[a] + ALPHA* (q_target- q_a) * s
			#print(w[a])
			s = ss
			t+=1

			if r==1:
				#print('halolo')
				break

		rewards.append(r_total)
		times.append(t)
		EPSILON = max(EPSILON*DECAY_RATE,0.1)
		if ep%200==0:

			print(f'EPISODE: {ep}')
			print(f'REWARD AVG: {np.mean(rewards[-100:])}')
			print(f'TIMESTEP AVG: {np.mean(times[-100:])}')
			print(EPSILON)
			print(w)
			print(q(w,obs))
			print(f"Melhor acao: {np.argmax(q(w,obs))}")

	
	s = env.reset()

	for _ in range(200):
		env.render()
		a = np.argmax(q(w,s))

		ss, reward, done, info = env.step(a)
		s = ss			
		time.sleep(0.33)



















if __name__ == '__main__':
	main()