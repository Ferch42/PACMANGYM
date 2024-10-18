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
F = {}
V = {}

EVENT_DICT = {}

EMPTY_EVENT = Event(set(), set())

def Q(obs, g):

	global q
	s = (obs, g)
	#print(s)
	if s not in q:
		q[s] = np.zeros(4) 

	return q[s]

def meta_planner(s, sigma, goal):

	global EVENT_DICT, F, V

	initial_state = [s, sigma, goal, []]
	queue = [initial_state]
	visited_set = set()

	done = False

	final_plan = None

	while(len(queue)>0):

		print(queue)
		node = queue.pop(0)
		s, sigma, goal, plan = node
		visited_set.add((s, str(list(sorted(sigma))), goal))

		if goal == True:
			done = True
			final_plan = plan

		for ev in [x for x in EVENT_DICT.values() if x.previous_sigma == sigma]:
			print(ev)
			next_goal = prog(ev.next_sigma, goal)

			try:
				ss = F[(s, str(ev), 0)].argmax()
			except:
				ss = None

			next_node = [ss, ev.next_sigma, next_goal, plan.copy() + [str(ev)]]
			try:

				if (ss, str(list(sorted(ev.next_sigma))), next_goal) not in visited_set and V[(s, str(ev), 0)]>0:

					queue.append(next_node)
			except:
				pass

		if done:
			break
	
	print(final_plan)



	
	return np.random.randint(4)


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
		if (s, "[]['A']", 0) in F.keys():
			print('------------------------')
			print("F")
			print(F[(s, "[]['A']", 0)].argmax())
			print(F[(s, "[]['A']", 0)][F[(s, "[]['A']", 0)].argmax()])	
			print(V[(s, "[]['A']", 0)])
		
		if ep%10==0:
			print("debugging metaplanner")
			meta_planner(s, sigma, goal)
		
		env_size = env.SIZE*env.SIZE
		while(True):

			# Random action policy
			if np.random.uniform() <1 : #1-(ep/N_EPISODES):
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
				
				if sigma == event.previous_sigma:
					# Validity constraint
					# Perceived sigma must be equal to the event initiation sigma
					done = 0
					ev_reward = 0
					if current_event != EMPTY_EVENT:
						done = 1
						# Salient event detected
						if current_event == event:
							ev_reward = 1
					
				
					Q(s,event_key)[a] = Q(s,event_key)[a] + ALPHA * (ev_reward +  (1- done)*GAMMA*np.max(Q(ss,event_key))) - Q(s,event_key)[a]

					best_a = np.argmax(Q(s,event_key))
					
					if a == best_a:
						
						if (s,event_key, TIME_HORIZON-1) not in F.keys():
							F[(s, event_key, TIME_HORIZON-1)] = np.zeros(env_size)
						
						target = np.zeros(env_size)
						target[ss] = 1
						
						F[(s, event_key, TIME_HORIZON-1)] = F[(s, event_key, TIME_HORIZON-1)] + 0.1 * (target  - F[(s, event_key, TIME_HORIZON-1)])

						for tt in reversed(range(TIME_HORIZON-1)):

							if (s,event_key, tt) not in F.keys():
								F[(s, event_key, tt)] = np.zeros(env_size)

							if done:
								target = np.zeros(env_size)
								target[ss] = 1
								F[(s, event_key , tt)] = F[(s, event_key , tt)] + 0.1 * (target  - F[(s, event_key , tt)])
								
							else:
								if (ss, event_key, tt+1) in F.keys():
									F[(s, event_key, tt)] = F[(s, event_key, tt)] + 0.1 *(F[(ss, event_key, tt+1)]- F[(s, event_key, tt)])

						for ttt in reversed(range(TIME_HORIZON)):
							if (s,event_key, ttt) not in V.keys():
								V[(s, event_key, ttt)] = 0
							
							if done and ev_reward == 1:
								V[(s, event_key, ttt)]  = V[(s, event_key, ttt)] + 0.1 *(1- V[(s, event_key, ttt)])
							
							if not done and ttt<TIME_HORIZON-1 and (ss, event_key, ttt+1) in V.keys():
								V[(s, event_key, ttt)]  = V[(s, event_key, ttt)] + 0.1 *(V[(ss, event_key, ttt+1)]- V[(s, event_key, ttt)])


				

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
				print('Final STATE')
				print(s)
				break

		rewards.append(r_total)
		times.append(t)
		EPSILON = max(EPSILON*DECAY_RATE,0.1)
		


if __name__ == '__main__':

	main()