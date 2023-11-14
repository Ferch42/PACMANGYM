from LTL import powerset, prog
from functools import reduce

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


def value_iteration_graph(graph_states_dict,graph_transitions, epsilon = 0.00001, gamma = 0.9):

	V = {}
	for state in graph_states_dict.values():
		V[state] = 1

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
	return V








