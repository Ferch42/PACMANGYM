from patch_env import PatchEnv
import numpy as np
import time
from event import Event
from ltl_graph_generator import *


ALPHA = 0.1
GAMMA = 0.99
DECAY_RATE = 0.99
EPSILON = 1
TIME_HORIZON = 20

N_EPISODES = 1200

q = {}
NS = {}

EVENT_DICT = {}

EMPTY_EVENT = Event(set(), set())


def Q(obs, g):

	global q
	s = (obs, g)
	#print(s)
	if s not in q:
		q[s] = np.zeros(4) 

	return q[s]


def main():

	global EPSILON,q, EMPTY_EVENT
	env = PatchEnv()
	obs = env.reset()
	print(env.GOAL)
	#w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = []
	times = []
	avg_rewards = []
	avg_timesteps = []

	
	print(N_EPISODES)
	for ep in range(N_EPISODES):

		s, info = env.reset()

		sigma  =info['P']
		goal = info['GOAL']

		t = 0
		r_total = 0
		
		while(True):

			# Random action policy
			if np.random.uniform() < 1-(ep/N_EPISODES):
				a = np.random.randint(4)
			else:
				a = Q(s, "[]['A']").argmax()


			ss, reward, done_ep, info = env.step(a)
			#print(goal)
			next_sigma = info['P']
			next_goal = info['GOAL']
			current_event = info['E']

			if current_event != EMPTY_EVENT:
				# New event detected
				EVENT_DICT[str(current_event)] = current_event

			
			for event_key, event in EVENT_DICT.items():

				done = 0
				ev_reward = 0
				if current_event == event:
					done = 1
					# Salient event detected
					if current_event == event:
						ev_reward = 1
				
			
				Q(s,event_key)[a] = Q(s,event_key)[a] + ALPHA * (ev_reward +  (1- done)*GAMMA*np.max(Q(ss,event_key))) - Q(s,event_key)[a]

				"""
				if done and ev_reward ==1:
					NS[(ss,ev)] = ss

				best_a = np.argmax(Q(s,ev))
				if a == best_a and Q(s,ev)[a]>0:
					NS[(s,ev)] = NS[(ss,ev)]
				"""			
			#done = r = int(current_event == event_goal)
			
			r_total += reward

			s = ss
			sigma = next_sigma
			goal = next_goal

			t+=1

			if t%10_000==0:

				print(f'EPISODE: {ep}')
				print(f'REWARD AVG: {np.mean(rewards[-100:])}')
				avg_rewards.append(np.mean(rewards[-100:]))
				print(f'TIMESTEP AVG: {np.mean(times[-100:])}')
				avg_timesteps.append(np.mean(times[-100:]))
				print(goal)
				print("Q value estimates")
				print(sum([x.sum() for x in q.values()]))
				print(EVENT_DICT.keys())	


			if done_ep:
				print(f"DONE IN {t}")
				#print('halolo')
				break

		rewards.append(r_total)
		times.append(t)
		EPSILON = max(EPSILON*DECAY_RATE,0.1)
		


if __name__ == '__main__':

	main()