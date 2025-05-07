import time
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import random


STOP_PROB = 0.1
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
    '🥐' :(20,20),
    '🍤': (25,37),
    '🍓': (43,15)
}

ACTION_DICT = {
    0: (0,1),
    1: (1,0),
    2: (-1,0),
    3: (0,-1)
}


REVERSE_ACTION_DICT = {
    (0,1) : 0,
    (1,0) : 1,
    (-1,0) : 2,
    (0,-1) : 3,
    (1,1) : 0,
    (0,0) :0,
    (-1, -1): 2,
    (1,-1): 1,
    (-1, 1): 2
}


IMPOSSIBLE_GOALS = ['🍤']
GOALS = ['🍜' ,'🥗' ,'🍸' ,'🍹' ,'🌮','🍣','🥓', '🥐','🍤', '🍓'] #+ IMPOSSIBLE_GOALS
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
    time.sleep(1)
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
        
        #render()
        ss , reward = step(Q((POINTS['🤖'], goal)).argmax(), goal)

        if reward==1:
            return ss,(i+1), True
    
    return ss, t, False


def check_sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        return 0



def run_universal_policy(goal, t):

    global POINTS
    
    ss = POINTS['🤖']
    POINTS['G'] = goal
    for i in range(t):
        
        action = REVERSE_ACTION_DICT[(check_sign(goal[0] -ss[0]), check_sign(goal[1] - ss[1]))]
        #render()
        ss , reward = step(action, 'G')

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

    task = random.sample(GOALS,3)
    total_time = 0
    events = []
    goal_completed = False

    buffer_history = set()

    perceived_options_set = set()
    print(task)

    while(not goal_completed):
        
        sx,sy = POINTS['🤖']
        # Check for perception of event
        for g in task:
            gx,gy = POINTS[g]
            if (abs(sx-gx) <=5) and (abs(sy-gy) <=5):

                buffer_history.add(((sx,sy), g))
                perceived_options_set.add(g)
        
        # perform exploratory action
        action = np.random.randint(4)
        ss, _ = step(action, goal)
        total_time +=1

        if len(perceived_options_set) ==3:
            goal_completed = True
            for g in task:
                gx,gy = POINTS[g]
                if (abs(sx-gx) <=5) and (abs(sy-gy) <=5): 
                    _,elapsed_time, _ = run_policy(g, 100)

                    total_time += elapsed_time

                else:

                    goal_state = [x for x in buffer_history if x[1] == g][0][0]
                    _,elapsed_time, _ = run_universal_policy(goal_state, 100)
                    total_time += elapsed_time
                    _,elapsed_time, _ = run_policy(g, 100)
                    total_time += elapsed_time
            
            print(total_time)


if __name__=='__main__':

    main()
    #render()
    #POSSIBLE_PATHS = (('🍜',), ('🥗' ,'🍸'), ('🥗' ,'🍣'),('🌮', '🍹'), ('🌮', '🥓'))

    #print(possible_events(tuple()))