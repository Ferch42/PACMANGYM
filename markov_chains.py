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

q = dict()

def Q(state):
    
    if state not in q:
        q[state] = np.zeros(4)

    return q[state]

def reset():

    POINTS['🤖'] = (25,25)


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

    global POINTS, STOP_PROB, ACTION_DICT, N

    X, Y = POINTS['🤖']
    
    if np.random.uniform()>=STOP_PROB:
        D_X, D_Y = ACTION_DICT[action]

        X = min(max(X+D_X, 0), N-1)
        Y = min(max(Y+D_Y, 0), N-1)

    POINTS['🤖'] = (X,Y)
    
    reward = int(POINTS['🤖'] == POINTS[goal])
    
    return POINTS['🤖'], reward


def run_policy(goal, t):

    
    ss = POINTS['🤖']
    for i in range(t):

        ss , reward = step(Q((POINTS['🤖'], goal)).argmax(), goal)

        if reward==1:
            return ss,t-(i+1), True
    
    return ss, t, False


def main():

    goal = '🥗'
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

    counts, bins = np.histogram(ep_len, bins=200)
    plt.stairs(counts, bins, fill=True)
    plt.show()



if __name__=='__main__':
    main()