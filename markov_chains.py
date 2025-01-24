import time
import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt


STOP_PROB = 0.2
GAMMA = 0.99

N = 51

POINTS = {
    '🤖' : (25,25),
    '🍜' : (0,0),
    '🥗' : (35,35),
    '🍸' : (40,25),
    '🍹' : (10,25),
    '🌮' : (15,35),
}

ACTION_DICT = {
    0: (0,1),
    1: (1,0),
    2: (-1,0),
    3: (0,-1)
}

GOALS = ['🍜' ,'🥗' ,'🍸' ,'🍹' ,'🌮' ]
TRAJECTORY = []


q = dict()

def Q(state):
    
    if state not in q:
        q[state] = np.zeros(4)

    return q[state]

def reset():

    global TRAJECTORY

    POINTS['🤖'] = (25,25)
    TRAJECTORY = [POINTS['🤖']]


def render():

    global POINTS, STOP_PROB, ACTION_DICT, N
    
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

    global TRAJECTORY
    
    goal = '🥗'
    reset()
    for i in tqdm(range(1_000_000)):

        s = POINTS['🤖']
        action = np.random.randint(4)
        ss, _ = step(action, goal)
        
        for g in GOALS:

            reward = int(POINTS['🤖'] == POINTS[g])
            Q((s,g))[action] = Q((s,g))[action] + 0.1 * (reward + GAMMA * (1-reward) * np.max(Q((ss,g))) - Q((s,g))[action])

        if i%100_000==0:

            print(f"The value is {sum([x.sum() for x in q.values()])}")
            print(f"The value is {Q(((25,25),goal))}")

    reset()

    ep_len = []
    e = 0
    for i in range(100_000):
        #os.system('cls')
        #render()
        #time.sleep(1)
        _, reward = step(Q((POINTS['🤖'], goal)).argmax(), goal)
        e+=1
        if reward == 1:
            ep_len.append(e)
            e = 0
            reset()

    print(ep_len)

    #counts, bins = np.histogram(ep_len, bins=200)
    #plt.stairs(counts, bins, fill=True)
    #plt.show()


    reset()

    T_MAX= 200
    t = 0
    high_level_activations = []
    
    #plan = ['🥗', '🍸', '🌮']
    plan = ['🥗'] 
    
    for p in plan:
        
        high_level_activations.append((p, t))
        
        ss, time_spent, done = run_policy(p, T_MAX - t)
        t += time_spent
        
    print("++++++++++++++++++++++++++++++++++")
    print(TRAJECTORY)
    print(high_level_activations)

    SUCCESS_DICT = {}
    TRIALS_DICT = {}

    for episode in range(10_000):
        
        reset()
        ss, time_spent, done = run_policy('🥗', 100)

        #print('----------------------')
        for i in range(len(TRAJECTORY)-1):
            key = (TRAJECTORY[i], '🥗', len(TRAJECTORY)- i - 1)
            #print(key)
            if key not in TRIALS_DICT:
                TRIALS_DICT[key] = 0
                SUCCESS_DICT[key] = 0
            
            
            TRIALS_DICT[key] +=1

            if done:
                SUCCESS_DICT[key] +=1
            
        for j in range(len(TRAJECTORY) -1):

            for k in range(len(TRAJECTORY) -2 - j):
                #print(k, len(TRAJECTORY)- j - 2)

                key = (TRAJECTORY[k], '🥗', len(TRAJECTORY)- j-k - 2)
                #print(key)
                
                if key not in TRIALS_DICT:
                    TRIALS_DICT[key] = 0
                    SUCCESS_DICT[key] = 0

                TRIALS_DICT[key] +=1


    PROB_DICT = {k: SUCCESS_DICT[k]/(TRIALS_DICT[k]) for k in TRIALS_DICT.keys()}
    print(PROB_DICT)




if __name__=='__main__':
    main()