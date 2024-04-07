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
NS = {}

EVENT_KNOWLEDGE_BASE = set()
WORLD_MODEL = set()

REGIONS_MAP = {}
REGIONS_LABELS = {}
REGIONS_TRANSITIONS = set()
region_count = 0


def add_state(s,l):

	global REGIONS_MAP, region_count
	if s not in REGIONS_MAP:
		REGIONS_MAP[s] = region_count
		REGIONS_LABELS[region_count] = l
		region_count = region_count + 1


def rename_region(r, possible_regions):

	if r in possible_regions:
		return min(possible_regions)
	else:
		return r

def merge_regions(r1, r2):

	global REGIONS_MAP, REGIONS_TRANSITIONS
	
	
	unified_region = min(r1,r2)

	transitions_to_remove = set()
	transitions_to_add = set()

	for transition in REGIONS_TRANSITIONS:

		if (transition[0] == r1 or transition[0] == r2) or (transition[1] == r1 or transition[1] == r2):

			transitions_to_remove.add(transition)
			transitions_to_add.add((rename_region(transition[0], (r1, r2)), rename_region(transition[1], (r1, r2))))

	
	for tr in transitions_to_remove:
		REGIONS_TRANSITIONS.remove(tr)
	for ta in transitions_to_add:
		REGIONS_TRANSITIONS.add(ta)
	for k in REGIONS_MAP.keys():

		if REGIONS_MAP[k] == r1 or REGIONS_MAP[k] == r2:
			REGIONS_MAP[k] = unified_region
	
def add_transition(s,l,ss,ll):

	global REGIONS_MAP, REGIONS_TRANSITIONS, REGIONS_LABELS

	add_state(s,l)
	add_state(ss,ll)

	REGIONS_TRANSITIONS.add((REGIONS_MAP[s], REGIONS_MAP[ss]))
	#time.sleep(1)	
	
	stable = False

	while not stable:

		initial_size = len(REGIONS_TRANSITIONS)

		region_numbers = set([x[0] for x in REGIONS_TRANSITIONS] + [x[1] for x in REGIONS_TRANSITIONS])

		for r1 in region_numbers:
			for r2 in region_numbers:
				if r1 != r2 and (r1, r2) in REGIONS_TRANSITIONS and (r2,r1) in REGIONS_TRANSITIONS and REGIONS_LABELS[r1] == REGIONS_LABELS[r2]:
					# Merge regions
					merge_regions(r1,r2)

		final_size = len(REGIONS_TRANSITIONS)

		if initial_size == final_size:
			stable = True

	


def stringfy_set(s):

	return str(s).replace('{', '').replace("}",'').replace('set()', '_').replace("'", "")

def Q(obs, event):

	global q

	#s = 0
	#for i in range(len(obs)):
	#	s+= (obs[i]+10)*(20**i)
	s = (obs, event)
	#print(s)
	if s not in q:
		q[s] = np.zeros(4)
		if event == -1:
			q[s] = q[s]+1

	return q[s]


def planner_action(s, goal):

	plan = plan_from_world_model(s,goal)

	if plan != None:
		
		return np.argmax(Q(s, plan[0]))
	
	else:
		return np.argmax(Q(s, -1))




def plan_from_world_model(s,formula):

	global REGIONS_MAP, REGIONS_TRANSITIONS, REGIONS_LABELS, q, NS

	region = REGIONS_MAP[s]
	initial_state = [formula, region, s, []]

	queue = [initial_state]
	visited_set = set()
	

	done = False
	while len(queue)>0:
		
		node = queue.pop(0)
		
		node_formula, node_region, state, plan = node

		visited_set.add((node_formula, node_region, state))
		
		if node_formula == True:
			done = True
			return plan


		for _ ,next_region in [transition for transition in REGIONS_TRANSITIONS if transition[0] == node_region]:
			
			next_formula= prog(REGIONS_LABELS[next_region], node_formula)
			event = (node_region, next_region)
			try:
				next_state = NS[(state, event)]
			except:
				next_state = None
			next_node = [next_formula, next_region, next_state, plan.copy() + [event]]

			
			if (next_formula, next_region, next_state) not in visited_set and (np.max(Q(state, event)) > 0):
				queue.append(next_node)


		if done:
			break
				
	
	return None


def main():

	global EPSILON,q, REGIONS_MAP, REGIONS_TRANSITIONS
	env = LetterEnv()
	obs = env.reset()
	print(env.GOAL)
	#w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = [0]
	times = [0]
	avg_rewards = [0]
	avg_timesteps = [0]

	
	print(N_EPISODES)
	for ep in range(N_EPISODES):

		s, info = env.reset()

		sigma  =info['P']
		goal = info['GOAL']

		t = 0
		r_total = 0
		add_state(s,sigma)
		print(f"SOLVING GOAL: {goal} ")
		
		while(t<500):

			# Random action policy
			#a = np.random.randint(4)
			# policy by the planner
			a = planner_action(s, goal)
			
			ss, reward, done_ep, info = env.step(a)
			#print(goal)
			next_sigma = info['P']
			next_goal = info['GOAL']
			#current_event = info['E']

			add_transition(s,sigma,ss,next_sigma)

			current_event = (REGIONS_MAP[s], REGIONS_MAP[ss])

			Q(s, -1)[a] = Q(s,-1)[a] + ALPHA * (0 +  GAMMA*np.max(Q(ss,-1))) - Q(s,-1)[a]

			
			for ev in REGIONS_TRANSITIONS:

				done = 0
				ev_reward = 0
				if current_event[0]!=current_event[1]:
					done = 1
					# Salient event detected
					if current_event == ev:
						ev_reward = 1
				
			
				Q(s,ev)[a] = Q(s,ev)[a] + ALPHA * (ev_reward +  (1- done)*GAMMA*np.max(Q(ss,ev))) - Q(s,ev)[a]


				if done and ev_reward ==1:
					NS[(ss,ev)] = ss

				best_a = np.argmax(Q(s,ev))
				if a == best_a and Q(s,ev)[a]>0:
					NS[(s,ev)] = NS[(ss,ev)]
			
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

				print('regions')
				print(set(REGIONS_MAP.values()))
				print(REGIONS_TRANSITIONS)
				print([x for x in REGIONS_LABELS.items() if x[1]!=set()])


			if done_ep:
				print(f"DONE IN {t}")
				#print('halolo')
				break

		rewards.append(r_total)
		times.append(t)
		EPSILON = max(EPSILON*DECAY_RATE,0.1)
		

	
	s, info = env.reset()
	sigma  =info['P']
	goal = info['GOAL']


	with open('rewards_option.txt', 'w+') as f:
		f.write(str(avg_rewards))


	with open('timesteps_option.txt', 'w+') as f:
		f.write(str(avg_timesteps))
	
	for _ in range(200):
		

		a = planner_action(s, goal)

		ss, reward, done, info = env.step(a)
		
		env.render()
		print(s)
		next_sigma =info['P']
		next_goal = info['GOAL']
		s = ss
		sigma = next_sigma
		goal = next_goal	
		
		if done:
			break		
		
		time.sleep(0.33)





if __name__ == '__main__':
	main()