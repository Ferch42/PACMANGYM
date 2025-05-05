import time
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import random


STOP_PROB = 1/5
GAMMA = 0.99
Q_FLAG = True

T_MAX= 37

N = 51
INITIAL_STATE = (25,25)

POINTS = {
    '🤖' : INITIAL_STATE,
    '🍜' : (0,14),
    '🥗' : (35,35),
    '🍸' : (40,25),
    '🍹' : (10,25),
    '🌮' : (15,35),
    '🍣' : (40,44),
    '🥓' : (5,31),
    '🍤': (-1,-1)
}

ACTION_DICT = {
    0: (0,1),
    1: (1,0),
    2: (-1,0),
    3: (0,-1)
}

IMPOSSIBLE_GOALS = ['🍤']
GOALS = ['🍜' ,'🥗' ,'🍸' ,'🍹' ,'🌮','🍣','🥓'] + IMPOSSIBLE_GOALS
N_GOALS = len(GOALS)
GOAL_INDEX = {g:i for i,g in enumerate(GOALS)}

TRAJECTORY = []

# TRIALS DICT

REWARDS_DICT = {}
TRIALS_DICT = {}

# POSSIBLE PATHS 

POSSIBLE_PATHS = (('🍜',), ('🥗' ,'🍸'), ('🥗' ,'🍣'),('🌮', '🍹'), ('🌮', '🥓'))

#POSSIBLE_PATHS = (('🍜',), ('🥗' ,'🍸'), ('🥗' ,'🍣'))


def possible_events(partial_path):
    
    global POSSIBLE_PATHS

    path_size = len(partial_path)

    mathing_paths = [x for x in POSSIBLE_PATHS if x[0:path_size] == partial_path]
    aux = []

    for m_p in mathing_paths:
        if len(m_p)>path_size:
            aux.append(m_p[path_size])
    
    return list(set(aux)) + IMPOSSIBLE_GOALS




q = dict()

def Q(state, n_actions = 4):
    
    if state not in q:
        q[state] = np.zeros(n_actions)

    return q[state]

def reset():

    global TRAJECTORY

    POINTS['🤖'] = INITIAL_STATE
    TRAJECTORY = [POINTS['🤖']]


def episilon_greedy_meta_policy(s, episilon = 0.1):

    global N_GOALS
    #s = (s, tuple(events), t)
    events = s[1]
    possible_events_list = possible_events(events)
    

    best_a  = Q(s, n_actions = N_GOALS).argmax()

    """
    if Q(s, n_actions= N_GOALS).max()>0:
        print(s)
        print(Q(s, n_actions= N_GOALS))
    """

    if np.random.uniform()< episilon or Q(s, n_actions = N_GOALS).max() == 0:
        # random action
        a = random.choice(possible_events_list)
        return a
    
    return GOALS[best_a]



def render():

    global POINTS, STOP_PROB, ACTION_DICT, N, T_MAX
    
    grid = '#'*(N+2) + '\n'
    for i in range(N):
        line = '#'
        for j in range(N):
            
            flag = False
            for letter,point in POINTS.items():
                if point == (i,j):
                    line += letter
                    flag = True

            if not flag:
                line += ' '
        line += '#\n'
        grid+=line
    grid += '#'*(N+2)

    print(grid)


def step(action, goal):

    global POINTS, STOP_PROB, ACTION_DICT, TRAJECTORY,N

    X, Y = POINTS['🤖']
    
    if np.random.uniform()>=STOP_PROB:
        D_X, D_Y = ACTION_DICT[action]

        X = min(max(X+D_X, 0), N-1)
        Y = min(max(Y+D_Y, 0), N-1)

    POINTS['🤖'] = (X,Y)

    TRAJECTORY.append(POINTS['🤖'])
    
    reward = int(POINTS['🤖'] == POINTS[goal])
    
    return POINTS['🤖'], reward


def run_policy(goal, t):

    
    ss = POINTS['🤖']
    for i in range(t):
        
        ss , reward = step(Q((POINTS['🤖'], goal)).argmax(), goal)

        if reward==1:
            return ss,(i+1), True
    
    return ss, t, False




def main():

    global TRAJECTORY, REWARDS_DICT, TRIALS_DICT, POSSIBLE_PATHS, q, Q_FLAG, IMPOSSIBLE_GOALS
    
    goal = '🍜'
    reset()
    for i in tqdm(range(1_000_000)):

        s = POINTS['🤖']
        action = np.random.randint(4)
        ss, _ = step(action, goal)
        
        for g in GOALS:
            
            reward = int(POINTS['🤖'] == POINTS[g])
            Q((s,g))[action] = Q((s,g))[action] + 0.1 * (reward + GAMMA * (1-reward) * np.max(Q((ss,g))) - Q((s,g))[action])
    
        
    reset()

    total_time = 0
    events = []
    goal_completed = False

    while(not goal_completed):

        #s = POINTS['🤖']
        pe = possible_events(tuple(events)) 
        event_done = False
        p = None
        t = 0

        while(not event_done):
        
            # ADDING THE MAX TIME
            t+=1
            
            for e in pe:
                _ ,_ , done = run_policy(e, t)
                total_time += t

                if done:
                    event_done = True
                    p = e
                    print(f'EVENT DONE IN {t}')
                    print(f"EVENT {p}")
                    
                
        events.append(p)
        for possible_path in POSSIBLE_PATHS:
            if set(events) == set(possible_path):
                
                goal_completed = True
                print(events)
                print(f"Done in {total_time} timesteps")
                




if __name__=='__main__':

    main()
    #render()
    #POSSIBLE_PATHS = (('🍜',), ('🥗' ,'🍸'), ('🥗' ,'🍣'),('🌮', '🍹'), ('🌮', '🥓'))

    #print(possible_events(tuple()))