from LTL import powerset, prog
from functools import reduce
from button_env import update_symbolic_state_2
from time import sleep

def recursive_search(subformula, propositions, graph_transitions, graph_states_dict):


	new_formulas = set()
	for s in powerset(propositions):

		f = prog(s, subformula)
		if f not in graph_states_dict:
			new_formulas.add(f)

		if f not in graph_states_dict:
			i = reduce(max, graph_states_dict.values())
			graph_states_dict[f] = i+1

		graph_transitions.add((graph_states_dict[subformula], s, graph_states_dict[f]))


	for nf in new_formulas:
		recursive_search(nf,propositions, graph_transitions, graph_states_dict)



def generate_graph(formula, proprositions):

	i = 0 
	graph_states_dict = {formula : i}
	graph_transitions = set()

	recursive_search(formula, proprositions, graph_transitions, graph_states_dict)

	return graph_states_dict, graph_transitions


def value_iteration_graph(graph_states_dict,graph_transitions, epsilon = 0.00001, gamma = 0.99):

	V = {}
	for state in graph_states_dict.values():
		V[state] = 0

	terminal_state = graph_states_dict[True]

	r = lambda s,ss: 1 if ss==terminal_state and s!=terminal_state  else 0

	while True:

		delta = 0
		for s in graph_states_dict.values():

			V_target = reduce(max, [r(state,next_state) + gamma*V[next_state] for state, action,next_state in graph_transitions if s==state])
			delta = max(delta, abs(V[s] - V_target))
			V[s] = V_target

		if delta< epsilon:
			break

	V_state = dict((k,V[v]) for k,v in graph_states_dict.items())

	return V_state



def value_iteration_ltl_graph(graph_states_dict,graph_transitions, epsilon = 0.00001, gamma = 0.99):

	V = {}
	for state in graph_states_dict.values():
		V[state] = 0

	terminal_state_list = [ts for ts in graph_states_dict if ts[0]==True]
	assert(len(terminal_state_list) == 1)

	terminal_state_item  = terminal_state_list.pop(0)

	terminal_state = graph_states_dict[terminal_state_item]

	r = lambda s,ss: 1 if ss==terminal_state and s!=terminal_state  else 0

	while True:

		delta = 0
		for s in graph_states_dict.values():

			V_target = reduce(max, [r(state,next_state) + gamma*V[next_state] for state,next_state in graph_transitions if s==state])
			delta = max(delta, abs(V[s] - V_target))
			V[s] = V_target

		if delta< epsilon:
			break

	V_state = dict((k,V[v]) for k,v in graph_states_dict.items())

	return V_state


def generate_ltl_env_graph(sigma,formula,state_dict ,transitions):

	
	inverse_state_dict = dict((v,k) for k,v in state_dict.items())
	ltl_state_dict = {}
	ltl_transitions = set()

	ltl_sigma = (formula, sigma)

	ltl_state_dict[ltl_sigma] = 0

	new_states_list = [ltl_sigma]
	i=0

	while True:

		new_states = new_states_list.pop(0)

		for e in [transition for transition in transitions if transition[0] == state_dict[new_states[1]]]:

			next_sigma = inverse_state_dict[e[1]]
			next_formula = prog(next_sigma, new_states[0])
			
			new_ltl_sigma = (next_formula, next_sigma)

			if new_ltl_sigma not in ltl_state_dict:

				i+=1 
				ltl_state_dict[new_ltl_sigma] = i

				if type(new_ltl_sigma[0]) != bool:
					new_states_list.append(new_ltl_sigma)

			ltl_transitions.add((ltl_state_dict[new_states], ltl_state_dict[new_ltl_sigma]))

			if type(new_ltl_sigma[0]) == bool:
				ltl_transitions.add((ltl_state_dict[new_ltl_sigma], ltl_state_dict[new_ltl_sigma]))

		if len(new_states_list) == 0:
			break

	return ltl_state_dict, ltl_transitions



def plan_ltl_event(sigma,formula):

	initial_state = (formula, sigma)

	queue = [initial_state]
	visited_set = set()
	
	parent_dict = {}

	done = False
	while len(queue)>0:
		
		node = queue.pop(0)
		visited_set.add(node)
		node_formula, node_sigma = node
		

		for action in range(1,7):

			next_formula= prog(update_symbolic_state_2(node_sigma, action), node_formula)
			next_node = (next_formula, update_symbolic_state_2(node_sigma, action) )
			
			if next_node not in visited_set:
				queue.append(next_node)
				parent_dict[next_node] = (node, action)
			if next_formula == True:
				done = True
		if done:
			break
				
	
	# Walk back from goal to start
	true_formula = None

	for k in parent_dict:
		if k[0]== True:
			true_formula = k
	S = k
	print(S)
	plan = []
	nodes = [S]
	while(S!= initial_state):
		parent, action = parent_dict[S]
		plan.insert(0, action)
		nodes.insert(0, parent)
		S = parent
	
	return plan, nodes



		






if __name__ == '__main__':
	
	ENV_STATE_DICT = {(): 0, ('LIGHT',): 1, ('LIGHT', 'SOUND'): 2, ('SOUND',): 3, ('MONKEY', 'SOUND'): 4, ('LIGHT', 'MONKEY', 'SOUND'): 5, ('LIGHT', 'MONKEY'): 6, ('MONKEY',): 7}
	TRANSITIONS = {(3, 4), (5, 4), (2, 2), (1, 0), (7, 7), (6, 5), (4, 5), (3, 3), (5, 6), (0, 1), (1, 2), (2, 1), (6, 7), (7, 6), (3, 2), (4, 4), (5, 5), (0, 0), (1, 1), (2, 3), (6, 6)}
	LIGHT_GOAL = ("UNTIL", "TRUE", ("AND", "LIGHT", ("UNTIL", "TRUE", ("AND", "SOUND", ("UNTIL", "TRUE", ("AND", ("NOT", "LIGHT"), ("UNTIL", "TRUE", "MONKEY")))))))

	#print(generate_ltl_env_graph((), ("UNTIL", "TRUE", "MONKEY"),ENV_STATE_DICT, TRANSITIONS ))

	#generate_ltl_env_graph((), LIGHT_GOAL,ENV_STATE_DICT, TRANSITIONS )

	#print(value_iteration_ltl_graph(*generate_ltl_env_graph((), LIGHT_GOAL,ENV_STATE_DICT, TRANSITIONS )))

	#print(generate_ltl_env_graph((), LIGHT_GOAL,ENV_STATE_DICT, TRANSITIONS ))

	print(plan_ltl_event((), LIGHT_GOAL))

