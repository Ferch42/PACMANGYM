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

	return q[s]


def planner_action(s, goal):

	plan = plan_from_world_model(s,goal)

	if plan != None:
		
		return np.argmax(Q(s, plan[0]))
	
	else:
		return np.random.randint(4)




def plan_from_world_model(s,formula):

	global REGIONS_MAP, REGIONS_TRANSITIONS, REGIONS_LABELS, q, NS

	region = REGIONS_MAP[s]
	initial_state = (formula, region, s)

	queue = [initial_state]
	visited_set = set()
	
	parent_dict = {}

	done = False
	while len(queue)>0:
		
		node = queue.pop(0)
		visited_set.add(node)
		node_formula, node_region, state = node


		for _ ,next_region in [transition for transition in REGIONS_TRANSITIONS if transition[0] == node_region]:
			
			next_formula= prog(REGIONS_LABELS[next_region], node_formula)
			event = (node_region, next_region)
			try:
				next_state = NS[(state, event)]
			except:
				next_state = None
			next_node = (next_formula, next_region, next_state)

			
			if next_node not in visited_set and (np.max(Q(state, event)) > 0 or next_formula == True):
				queue.append(next_node)

				parent_dict[next_node] = (node,event)
			if next_formula == True:
				done = True
		if done:
			break
				
	if done:
		# Walk back from goal to start
		true_formula = None
		
		for k in parent_dict:
			if k[0]== True:
				true_formula = k
		S = true_formula
		#print(S)
		plan = []
		nodes = [S]
		
		while(S!= initial_state):

			parent, action = parent_dict[S]
			plan.insert(0, action)
			nodes.insert(0, parent)
			S = parent
		
		return plan
	
	else:
		return None


def main():

	global EPSILON,q, REGIONS_MAP, REGIONS_TRANSITIONS
	env = LetterEnv()
	#w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = []
	
	s, info = env.reset()

	sigma  =info['P']
	goal = info['GOAL']
	print(f"First goal :{goal}")

	t = 0
	r_total = 0
	add_state(s,sigma)

		
	while(t<1000_000):

		a = planner_action(s, goal)
			
		ss, reward, done_ep, info = env.step(a)
		#env.render()
		#time.sleep(1)
		next_sigma = info['P']
		next_goal = info['GOAL']

		add_transition(s,sigma,ss,next_sigma)

		current_event = (REGIONS_MAP[s], REGIONS_MAP[ss])

			
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
		if t%500==0:
			EPSILON = max(EPSILON*DECAY_RATE,0.1)
		
		if reward ==1:
			rewards.append(t)
			print(f"Done in t: {t}")
			print(f"Epsilon: {EPSILON}")
			print(f"New goal: {env.GOAL}")






if __name__ == '__main__':
	main()