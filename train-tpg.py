# imports and helper methods
from tpg.tpg_trainer import TpgTrainer
from tpg.tpg_agent import TpgAgent

from osim.env import ProstheticsEnv

import multiprocessing as mp
import time
import random
import psutil
import os
import pickle
import datetime


"""
Run each agent in this method for parallization.
Args:
    args: (TpgAgent, envName, scoreList, numEpisodes, numFrames)
"""
def runAgent(args):
    agent = args[0]
    scoreList = args[1] 
    
    # skip if task already done by agent
    if agent.taskDone():
        print('Agent #' + str(agent.getAgentNum()) + ' can skip.')
        scoreList.append((agent.getUid(), agent.getOutcomes()))
        return

    env = ProstheticsEnv(visualize=False)

    score = 0
    
    state = env.reset()
    state.extend([0]*19)
    numRandFrames = random.randint(0,10)
    curAction = [0.2]*19 # start with all muscles barely activated
    for i in range(300): # frame loop
        if i < numRandFrames:
            _, _, isDone, _ = env.step(env.action_space.sample())
            continue

        act = agent.act(state)
        for i in range(19):
            curAction[i] += act[i]
            if curAction[i] < 0:
                curAction[i] = 0
            elif curAction[i] > 1:
                curAction[i] = 1
        # feedback from env
        state, reward, isDone, debug = env.step(curAction)
        state.extend(curAction) # feedback action, because sequence is important
        
        score += reward # accumulate reward in score
        if isDone:
            break # end early if losing state
            
    print('Agent #' + str(agent.getAgentNum())
          + ' | Score: ' + str(score))
        
    env.close()
    agent.reward(score)
    scoreList.append((agent.getUid(), agent.getOutcomes()))
    
# https://stackoverflow.com/questions/42103367/limit-total-cpu-usage-in-python-multiprocessing/42130713
def limit_cpu():
    p = psutil.Process(os.getpid())
    p.nice(10)


trainer = TpgTrainer(actions=19, actionRange=(-0.4,0.4,0.02), teamPopSizeInit=360)

processes = 3
pool = mp.Pool(processes=processes, initializer=limit_cpu)
man = mp.Manager()

allScores = [] # track all scores each generation

tStart = time.time()

logFileName = 'train-log-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.txt'

while True: # do generations with no end
    scoreList = man.list()
    
    pool.map(runAgent, 
        [(agent, scoreList)
        for agent in trainer.getAllAgents(skipTasks=[], noRef=True)])
    
    # apply scores
    trainer.applyScores(scoreList)
    scoreStats = trainer.generateScoreStats()
    allScores.append((trainer.curGen, scoreStats['min'], scoreStats['max'], scoreStats['average']))

    trainer.evolve() # go into next gen
    
    # save model after every gen
    with open('saved-model-1.pkl','wb') as f:
        pickle.dump(trainer,f)

    # save best agent each generation
    with open('best-agent.pkl','wb') as f:
        pickle.dump(trainer.getBestAgent(), f)
        
    print(chr(27) + "[2J")
    print('Time So Far (Seconds): ' + str(time.time() - tStart))
    print('Results: ', str(allScores))
    
    with open(logFileName, 'a') as f:
            f.write(str(trainer.curGen) + ' | '
                + str(scoreStats['min']) + ' | ' 
                + str(scoreStats['max']) + ' | '
                + str(scoreStats['average']) + '\n')



