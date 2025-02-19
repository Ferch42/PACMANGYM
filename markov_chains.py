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
    '🥓' : (5,31)
}

ACTION_DICT = {
    0: (0,1),
    1: (1,0),
    2: (-1,0),
    3: (0,-1)
}

GOALS = ['🍜' ,'🥗' ,'🍸' ,'🍹' ,'🌮','🍣','🥓']
N_GOALS = len(GOALS)
GOAL_INDEX = {g:i for i,g in enumerate(GOALS)}

TRAJECTORY = []

# TRIALS DICT

REWARDS_DICT = {}
TRIALS_DICT = {}

# POSSIBLE PATHS 

#POSSIBLE_PATHS = (('🍜',), ('🥗' ,'🍸'), ('🥗' ,'🍣'),('🌮', '🍹'), ('🌮', '🥓'))

POSSIBLE_PATHS = (('🍜',), ('🥗' ,'🍸'), ('🥗' ,'🍣'))


def possible_events(partial_path):
    
    global POSSIBLE_PATHS

    path_size = len(partial_path)

    mathing_paths = [x for x in POSSIBLE_PATHS if x[0:path_size] == partial_path]
    aux = []

    for m_p in mathing_paths:
        if len(m_p)>path_size:
            aux.append(m_p[path_size])
    
    return list(set(aux))




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


def update_trial_dict(trail, goal, done):

    global TRIALS_DICT, REWARDS_DICT
    #print(trail)

    for i in range(len(trail)-1):
        key = (trail[i], goal, len(trail)- i - 1)
        #print(key)
        if key not in REWARDS_DICT:
            TRIALS_DICT[key] = 0
            REWARDS_DICT[key] = 0.9
            
            
        TRIALS_DICT[key] +=1

        if done:
            REWARDS_DICT[key] = REWARDS_DICT[key] + 0.1 * (1-REWARDS_DICT[key])
        else:
            REWARDS_DICT[key] = REWARDS_DICT[key] + 0.1 * (0-REWARDS_DICT[key])


    for j in range(len(trail) -1):

        for k in range(len(trail) -2 - j):
            #print(k, len(trail)- j - 2)

            key = (trail[k], goal, len(trail)- j-k - 2)
            #print(key)
                
            if key not in REWARDS_DICT:
                TRIALS_DICT[key] = 0
                REWARDS_DICT[key] = 0.9
           
            TRIALS_DICT[key] +=1
            REWARDS_DICT[key] = REWARDS_DICT[key] + 0.1 * (0-REWARDS_DICT[key])




def main():

    global TRAJECTORY, REWARDS_DICT, TRIALS_DICT, POSSIBLE_PATHS, q, Q_FLAG
    
    goal = '🍜'
    reset()
    for i in tqdm(range(1_000_000)):

        s = POINTS['🤖']
        action = np.random.randint(4)
        ss, _ = step(action, goal)
        
        for g in GOALS:

            reward = int(POINTS['🤖'] == POINTS[g])
            Q((s,g))[action] = Q((s,g))[action] + 0.1 * (reward + GAMMA * (1-reward) * np.max(Q((ss,g))) - Q((s,g))[action])
    
    
    q_base_policies = q.copy()
    
    trial_list = []
    first_goal_reached_list = []
    for trial in range(100):
        q = q_base_policies.copy()
        REWARDS_DICT = {}
        reward_list = np.zeros(int(10_000/100))
        reached_goal = False

        for episode in range(10_000):
            
            reset()
            s = POINTS['🤖']

            t = 0

            high_level_activations = []
            
            goal_completed = False
            if Q_FLAG: 

                events = []
                while(t< T_MAX):

                    s = POINTS['🤖']
                    h_s = (s, tuple(events), T_MAX - t)
                    p = episilon_greedy_meta_policy(h_s)
                    ss, time_spent, done = run_policy(p, T_MAX - t)
                    high_level_activations.append((p, t, done, time_spent))
                    t += time_spent
                    
                    events.append(p)

                    for possible_path in POSSIBLE_PATHS:

                        if tuple(events) == possible_path and done:
                            t += T_MAX
                            goal_completed = True
                
            else:

                plan = POSSIBLE_PATHS[episode%3]

                events = []
                for p in plan:
                    s = POINTS['🤖']
                    h_s = (s, tuple(events), T_MAX - t)
                    
                    ss, time_spent, done = run_policy(p, T_MAX - t)
                    high_level_activations.append((p, t, done, time_spent))
                    t += time_spent

                    events.append(p)

                    for possible_path in POSSIBLE_PATHS:

                        if tuple(events) == possible_path and done:
                            t += T_MAX
                            goal_completed = True
                            if not reached_goal:
                                first_goal_reached_list.append(episode)
                                reached_goal = True
            
            if goal_completed:
                #reward_list.append(1)
                reward_list[int(episode/100)] += 1
            
            else:
                #reward_list.append(0)
                pass
            
            for h in high_level_activations:

                update_trial_dict(TRAJECTORY[h[1]: h[3]+ h[1]+1], h[0], h[2])



            act_events = []
            for ii in range(len(high_level_activations)):
                
                h = high_level_activations[ii]
                TT_MAXX = T_MAX - h[1]
                
                for et in range(h[3]):

                    reward = 1

                    for jj in range(ii, len(high_level_activations)):
                        h2 = high_level_activations[jj]

                        reward_key = 0
                        if h[0]==h2[0]:
                            ss = TRAJECTORY[h2[1] + et]
                            reward_key = (ss, h2[0], h2[3] - et)
                        
                        else:
                            ss = TRAJECTORY[h2[1]]
                            reward_key = (ss, h2[0], h2[3])
                        
                        if reward_key[-1] >0:
                            reward = reward * REWARDS_DICT[reward_key]
                        else:
                            reward = 0



                    s = TRAJECTORY[h[1] + et]
                    k = (s, tuple(act_events), TT_MAXX-et)


                    Q(k, n_actions= N_GOALS)[GOAL_INDEX[h[0]]]  = Q(k, n_actions= N_GOALS)[GOAL_INDEX[h[0]]] + 0.1 * (reward- Q(k, n_actions= N_GOALS)[GOAL_INDEX[h[0]]])

                act_events.append(h[0])

        trial_list.append(reward_list)
        print(f"RUN {trial} AVERAGE RETURN = {np.array(reward_list).mean()}")
    trial_list = np.array(trial_list)
    print(trial_list.shape)
    trial_list = np.mean(trial_list, axis = 0)
    plt.plot(trial_list)
    #plt.show()

    trial_list_2 = []
    Q_FLAG = False

    for trial in range(100):
        q = q_base_policies.copy()
        REWARDS_DICT = {}
        reward_list = np.zeros(int(10_000/100))

        for episode in range(10_000):
            
            reset()
            s = POINTS['🤖']

            t = 0

            high_level_activations = []
            
            goal_completed = False
            if Q_FLAG: 

                events = []
                while(t< T_MAX):

                    s = POINTS['🤖']
                    h_s = (s, tuple(events), T_MAX - t)
                    p = episilon_greedy_meta_policy(h_s)
                    ss, time_spent, done = run_policy(p, T_MAX - t)
                    high_level_activations.append((p, t, done, time_spent))
                    t += time_spent
                    
                    events.append(p)

                    for possible_path in POSSIBLE_PATHS:

                        if tuple(events) == possible_path and done:
                            t += T_MAX
                            goal_completed = True
                
            else:

                plan = POSSIBLE_PATHS[episode%3]

                events = []
                for p in plan:
                    s = POINTS['🤖']
                    h_s = (s, tuple(events), T_MAX - t)
                    
                    ss, time_spent, done = run_policy(p, T_MAX - t)
                    high_level_activations.append((p, t, done, time_spent))
                    t += time_spent

                    events.append(p)

                    for possible_path in POSSIBLE_PATHS:

                        if tuple(events) == possible_path and done:
                            t += T_MAX
                            goal_completed = True
            
            if goal_completed:
                #reward_list.append(1)
                reward_list[int(episode/100)] += 1
            
            else:
                #reward_list.append(0)
                pass
            
            for h in high_level_activations:

                update_trial_dict(TRAJECTORY[h[1]: h[3]+ h[1]+1], h[0], h[2])



            act_events = []
            for ii in range(len(high_level_activations)):
                
                h = high_level_activations[ii]
                TT_MAXX = T_MAX - h[1]
                
                for et in range(h[3]):

                    reward = 1

                    for jj in range(ii, len(high_level_activations)):
                        h2 = high_level_activations[jj]

                        reward_key = 0
                        if h[0]==h2[0]:
                            ss = TRAJECTORY[h2[1] + et]
                            reward_key = (ss, h2[0], h2[3] - et)
                        
                        else:
                            ss = TRAJECTORY[h2[1]]
                            reward_key = (ss, h2[0], h2[3])
                        
                        if reward_key[-1] >0:
                            reward = reward * REWARDS_DICT[reward_key]
                        else:
                            reward = 0



                    s = TRAJECTORY[h[1] + et]
                    k = (s, tuple(act_events), TT_MAXX-et)


                    Q(k, n_actions= N_GOALS)[GOAL_INDEX[h[0]]]  = Q(k, n_actions= N_GOALS)[GOAL_INDEX[h[0]]] + 0.1 * (reward- Q(k, n_actions= N_GOALS)[GOAL_INDEX[h[0]]])

                act_events.append(h[0])

        trial_list_2.append(reward_list)
        print(f"RUN {trial} AVERAGE RETURN = {np.array(reward_list).mean()}")
    trial_list_2 = np.array(trial_list_2)
    print(trial_list_2.shape)
    trial_list_2 = np.mean(trial_list_2, axis = 0)
    plt.plot(trial_list, label="META PLANNER")
    plt.plot(trial_list_2, label="RANDOM APPROACH")
    plt.show()
    




if __name__=='__main__':

    main()
    #render()
    #POSSIBLE_PATHS = (('🍜',), ('🥗' ,'🍸'), ('🥗' ,'🍣'),('🌮', '🍹'), ('🌮', '🥓'))

    #print(possible_events(tuple()))