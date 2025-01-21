import numpy as np

STOP_PROB = 0.1

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


def step(action):

    global POINTS, STOP_PROB, ACTION_DICT, N

    X, Y = POINTS['🤖']
    
    if np.random.uniform()>=STOP_PROB:
        D_X, D_Y = ACTION_DICT[action]

        X = min(max(X+D_X, 0), N-1)
        Y = min(max(Y+D_Y, 0), N-1)

    POINTS['🤖'] = (X,Y)


for i in range(100):

    render()
    print(POINTS['🤖'])
    a = int(input("Digite a acao: "))
    
    step(a)