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
from obstrans import obsTrans
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-c', '--cont', action='store_true', dest='cont', default=False)
(options, args) = parser.parse_args()


"""
Run each agent in this method for parallization.
Args:
    args: (TpgAgent, envName, scoreList, numEpisodes, numFrames)
"""
def runAgent(args):
    agent = args[0]
    scoreList = args[1]
    step = args[2]
    
    # skip if task already done by agent
    if agent.taskDone():
        print('Agent #' + str(agent.getAgentNum()) + ' can skip.')
        scoreList.append((agent.getUid(), agent.getOutcomes()))
        return

    env = ProstheticsEnv(visualize=False)

    score = 0
    
    state = env.reset(project=False)
    state = obsTrans(state)
    state.extend([0]*19)
    curAction = [0]*19
    for i in range(300): # frame loop
        act = agent.act(state)
        if not isinstance(act, list):
            act = [0]*19
            print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHHHhhhhhhhhhhhhhhh!!!!!!!!')
        for i in range(19):
            if act[i] > 0.333:
                curAction[i] += step
            elif act[i] < -0.333:
                curAction[i] -= step

            if curAction[i] < 0:
                curAction[i] = 0
            elif curAction[i] > 1:
                curAction[i] = 1

        # feedback from env
        state, reward, isDone, debug = env.step(curAction, project=False)
        state = obsTrans(state)
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

if options.cont:
    with open('saved-model-1.pkl', 'rb') as f:
        trainer = pickle.load(f)
else:
    trainer = TpgTrainer(actions=19, actionRange=(-1.0,1.0,0.35), teamPopSize=360, maxProgramSize=256)

processes = 7
pool = mp.Pool(processes=processes, maxtasksperchild=2)
man = mp.Manager()

allScores = [] # track all scores each generation

tStart = time.time()

logFileName = 'train-log-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + '.txt'

while True: # do generations with no end
    scoreList = man.list()
    
    pool.map(runAgent, 
        [(agent, scoreList, 0.05)
        for agent in trainer.getAllAgents(noRef=True)])
    
    # apply scores
    trainer.applyScores(scoreList)
    trainer.evolve(fitShare=False, tasks=[]) 
    scoreStats = trainer.scoreStats
    allScores.append((trainer.curGen, scoreStats['min'], scoreStats['max'], scoreStats['average']))
    
    # save model after every gen
    with open('saved-model-1.pkl','wb') as f:
        pickle.dump(trainer,f)

    # save best agent each generation
    #with open('best-agent.pkl','wb') as f:
    #    pickle.dump(trainer.getBestAgent(tasks=[]), f)
        
    print(chr(27) + "[2J")
    print('Time So Far (Seconds): ' + str(time.time() - tStart))
    print('Results: ', str(allScores))
    
    with open(logFileName, 'a') as f:
            f.write(str(trainer.curGen) + ' | '
                + str(scoreStats['min']) + ' | ' 
                + str(scoreStats['max']) + ' | '
                + str(scoreStats['average']) + '\n')



