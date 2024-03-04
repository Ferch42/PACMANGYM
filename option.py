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


def planner_action(s, sigma, goal, world_model):

	global q, NS, EPSILON
	plan, _ = plan_from_world_model(sigma,goal, world_model)

	if plan == None:
		return np.random.randint(4)
	
	# CHECK IF PLAN REACHES GOAL
	node = s
	plan_successful = True

	for event in plan:
		plan_successful = plan_successful and (np.max(Q(node, event)) > 0)
		if plan_successful:
			node = NS[(node,event)]
		else:
			break

	if plan_successful and np.random.uniform() >= EPSILON:
		return np.argmax(Q(s, plan[0]))
	
	else:
		return np.random.randint(4)




def plan_from_world_model(s,formula, world_model):

	sigma = stringfy_set(s)
	initial_state = (formula, sigma)

	queue = [initial_state]
	visited_set = set()
	
	parent_dict = {}

	done = False
	while len(queue)>0:
		
		node = queue.pop(0)
		visited_set.add(node)
		node_formula, node_sigma = node


		for _ ,next_sigma in [transition for transition in world_model if transition[0] == node_sigma]:
			
			next_formula= prog(set({next_sigma}), node_formula)
			next_node = (next_formula, next_sigma )
			
			if next_node not in visited_set:
				queue.append(next_node)

				parent_dict[next_node] = (node, stringfy_set(node_sigma)+ stringfy_set(next_sigma) )
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
			#print(S)
			parent, action = parent_dict[S]
			plan.insert(0, action)
			nodes.insert(0, parent)
			S = parent
		
		return plan, nodes
	
	else:
		return None, None


def main():

	global EPSILON,q
	env = LetterEnv()
	obs = env.reset()
	print(env.GOAL)
	#w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = []
	times = []
	print(N_EPISODES)
	for ep in range(N_EPISODES):

		s, info = env.reset()

		sigma  =info['P']
		goal = info['GOAL']

		t = 0
		r_total = 0

		
		while(t<500):

			# Random action policy
			#a = np.random.randint(4)
			# policy by the planner
			a = planner_action(s, sigma, goal, WORLD_MODEL)
			
			ss, reward, done_ep, info = env.step(a)
			next_sigma = info['P']
			next_goal = info['GOAL']
			current_event = info['E']

			if current_event != "_":
				# Salient event detected
				EVENT_KNOWLEDGE_BASE.add(current_event)
				WORLD_MODEL.add((stringfy_set(sigma), stringfy_set(next_sigma)))


		
			for ev in EVENT_KNOWLEDGE_BASE:

				done = 0
				ev_reward = 0
				if current_event!= "_":
					done = 1
					if current_event == ev:
						ev_reward = 1


				Q(s,ev)[a] = Q(s,ev)[a] + ALPHA * (ev_reward + (1- done)*GAMMA*np.max(Q(ss,ev))) - Q(s,ev)[a]

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

			if done_ep:
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
			#print(q.keys())
			print(f"Q_{cc}")

	
	s, info = env.reset()
	sigma  =info['P']
	goal = info['GOAL']
	
	for _ in range(200):
		

		a = planner_action(s, sigma, goal, WORLD_MODEL)

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