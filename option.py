from env import PacmanEnv
from button_env import ButtonEnv
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


def planner_action(s, sigma, goal):

	global q, NS, EPSILON
	plan, _ = plan_ltl_event(sigma, goal)

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


def main():

	global EPSILON,q
	env = ButtonEnv()
	obs = env.reset()
	print(env.GOAL)
	#w = np.zeros(shape = (4,obs.shape[0]))
	
	rewards = []
	times = []
	print(N_EPISODES)
	for ep in range(N_EPISODES):

		s, info = env.reset()

		sigma  = tuple(sorted(info['P']))
		goal = info['GOAL']
		#print(plan_ltl_event(sigma, goal))

		t = 0
		r_total = 0

		event_goal = np.random.randint(1,7)
		#print(event_goal)
		
		while(t<500):

			# Call the planner
			a = planner_action(s, sigma, goal)

			#print('in state', s,sigma, goal)
			ss, reward, done_ep, info = env.step(a)
			next_sigma = tuple(sorted(info['P']))
			next_goal = info['GOAL']
			current_event = info['E']

		
			for ev in range(1,7):

				done = 0
				ev_reward = 0
				if current_event!=0:
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
			print(f"Q_{cc}")

	
	s, info = env.reset()
	sigma  = tuple(sorted(info['P']))
	goal = info['GOAL']
	event_goal = np.random.randint(1,7)
	for _ in range(200):
		
		print(NS[(s,event_goal)])
		print(Q(s,event_goal))
		a = planner_action(s, sigma, goal)

		ss, reward, done, info = env.step(a)
		
		env.render()
		print(s)
		next_sigma = tuple(sorted(info['P']))
		next_goal = info['GOAL']
		s = ss
		sigma = next_sigma
		goal = next_goal	
		
		if done:
			break		
		
		time.sleep(0.33)





if __name__ == '__main__':
	main()