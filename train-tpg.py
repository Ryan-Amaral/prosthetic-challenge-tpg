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
    
    state = env.reset(project=False)
    state = obsTrans(state)
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

def obsTrans(obs):
    newObs = [obs['joint_pos']['ground_pelvis'],obs['joint_pos']['hip_r'],obs['joint_pos']['knee_r'],obs['joint_pos']['ankle_r'],obs['joint_pos']['hip_l'],obs['joint_pos']['knee_l'],obs['joint_pos']['ankle_l'],obs['joint_pos']['subtalar_l'],obs['joint_pos']['mtp_l'],obs['joint_pos']['back'],obs['joint_pos']['back_0'],obs['joint_vel']['ground_pelvis'],obs['joint_vel']['hip_r'],obs['joint_vel']['knee_r'],obs['joint_vel']['ankle_r'],obs['joint_vel']['hip_l'],obs['joint_vel']['knee_l'],obs['joint_vel']['ankle_l'],obs['joint_vel']['subtalar_l'],obs['joint_vel']['mtp_l'],obs['joint_vel']['back'],obs['joint_vel']['back_0'],obs['joint_acc']['ground_pelvis'],obs['joint_acc']['hip_r'],obs['joint_acc']['knee_r'],obs['joint_acc']['ankle_r'],obs['joint_acc']['hip_l'],obs['joint_acc']['knee_l'],obs['joint_acc']['ankle_l'],obs['joint_acc']['subtalar_l'],obs['joint_acc']['mtp_l'],obs['joint_acc']['back'],obs['joint_acc']['back_0'],obs['body_pos']['pelvis'],obs['body_pos']['femur_r'],obs['body_pos']['pros_tibia_r'],obs['body_pos']['pros_foot_r'],obs['body_pos']['femur_l'],obs['body_pos']['tibia_l'],obs['body_pos']['talus_l'],obs['body_pos']['calcn_l'],obs['body_pos']['toes_l'],obs['body_pos']['torso'],obs['body_pos']['head'],obs['body_vel']['pelvis'],obs['body_vel']['femur_r'],obs['body_vel']['pros_tibia_r'],obs['body_vel']['pros_foot_r'],obs['body_vel']['femur_l'],obs['body_vel']['tibia_l'],obs['body_vel']['talus_l'],obs['body_vel']['calcn_l'],obs['body_vel']['toes_l'],obs['body_vel']['torso'],obs['body_vel']['head'],obs['body_acc']['pelvis'],obs['body_acc']['femur_r'],obs['body_acc']['pros_tibia_r'],obs['body_acc']['pros_foot_r'],obs['body_acc']['femur_l'],obs['body_acc']['tibia_l'],obs['body_acc']['talus_l'],obs['body_acc']['calcn_l'],obs['body_acc']['toes_l'],obs['body_acc']['torso'],obs['body_acc']['head'],obs['body_pos_rot']['pelvis'],obs['body_pos_rot']['femur_r'],obs['body_pos_rot']['pros_tibia_r'],obs['body_pos_rot']['pros_foot_r'],obs['body_pos_rot']['femur_l'],obs['body_pos_rot']['tibia_l'],obs['body_pos_rot']['talus_l'],obs['body_pos_rot']['calcn_l'],obs['body_pos_rot']['toes_l'],obs['body_pos_rot']['torso'],obs['body_pos_rot']['head'],obs['body_vel_rot']['pelvis'],obs['body_vel_rot']['femur_r'],obs['body_vel_rot']['pros_tibia_r'],obs['body_vel_rot']['pros_foot_r'],obs['body_vel_rot']['femur_l'],obs['body_vel_rot']['tibia_l'],obs['body_vel_rot']['talus_l'],obs['body_vel_rot']['calcn_l'],obs['body_vel_rot']['toes_l'],obs['body_vel_rot']['torso'],obs['body_vel_rot']['head'],obs['body_acc_rot']['pelvis'],obs['body_acc_rot']['femur_r'],obs['body_acc_rot']['pros_tibia_r'],obs['body_acc_rot']['pros_foot_r'],obs['body_acc_rot']['femur_l'],obs['body_acc_rot']['tibia_l'],obs['body_acc_rot']['talus_l'],obs['body_acc_rot']['calcn_l'],obs['body_acc_rot']['toes_l'],obs['body_acc_rot']['torso'],obs['body_acc_rot']['head'],obs['forces']['abd_r'],obs['forces']['add_r'],obs['forces']['hamstrings_r'],obs['forces']['bifemsh_r'],obs['forces']['glut_max_r'],obs['forces']['iliopsoas_r'],obs['forces']['rect_fem_r'],obs['forces']['vasti_r'],obs['forces']['abd_l'],obs['forces']['add_l'],obs['forces']['hamstrings_l'],obs['forces']['bifemsh_l'],obs['forces']['glut_max_l'],obs['forces']['iliopsoas_l'],obs['forces']['rect_fem_l'],obs['forces']['vasti_l'],obs['forces']['gastroc_l'],obs['forces']['soleus_l'],obs['forces']['tib_ant_l'],obs['forces']['ankleSpring'],obs['forces']['pros_foot_r_0'],obs['forces']['foot_l'],obs['forces']['HipLimit_r'],obs['forces']['HipLimit_l'],obs['forces']['KneeLimit_r'],obs['forces']['KneeLimit_l'],obs['forces']['AnkleLimit_r'],obs['forces']['AnkleLimit_l'],obs['forces']['HipAddLimit_r'],obs['forces']['HipAddLimit_l'],[obs['muscles']['abd_r']['activation']],[obs['muscles']['abd_r']['fiber_length']],[obs['muscles']['abd_r']['fiber_velocity']],[obs['muscles']['abd_r']['fiber_force']],[obs['muscles']['add_r']['activation']],[obs['muscles']['add_r']['fiber_length']],[obs['muscles']['add_r']['fiber_velocity']],[obs['muscles']['add_r']['fiber_force']],[obs['muscles']['hamstrings_r']['activation']],[obs['muscles']['hamstrings_r']['fiber_length']],[obs['muscles']['hamstrings_r']['fiber_velocity']],[obs['muscles']['hamstrings_r']['fiber_force']],[obs['muscles']['bifemsh_r']['activation']],[obs['muscles']['bifemsh_r']['fiber_length']],[obs['muscles']['bifemsh_r']['fiber_velocity']],[obs['muscles']['bifemsh_r']['fiber_force']],[obs['muscles']['glut_max_r']['activation']],[obs['muscles']['glut_max_r']['fiber_length']],[obs['muscles']['glut_max_r']['fiber_velocity']],[obs['muscles']['glut_max_r']['fiber_force']],[obs['muscles']['iliopsoas_r']['activation']],[obs['muscles']['iliopsoas_r']['fiber_length']],[obs['muscles']['iliopsoas_r']['fiber_velocity']],[obs['muscles']['iliopsoas_r']['fiber_force']],[obs['muscles']['rect_fem_r']['activation']],[obs['muscles']['rect_fem_r']['fiber_length']],[obs['muscles']['rect_fem_r']['fiber_velocity']],[obs['muscles']['rect_fem_r']['fiber_force']],[obs['muscles']['vasti_r']['activation']],[obs['muscles']['vasti_r']['fiber_length']],[obs['muscles']['vasti_r']['fiber_velocity']],[obs['muscles']['vasti_r']['fiber_force']],[obs['muscles']['abd_l']['activation']],[obs['muscles']['abd_l']['fiber_length']],[obs['muscles']['abd_l']['fiber_velocity']],[obs['muscles']['abd_l']['fiber_force']],[obs['muscles']['add_l']['activation']],[obs['muscles']['add_l']['fiber_length']],[obs['muscles']['add_l']['fiber_velocity']],[obs['muscles']['add_l']['fiber_force']],[obs['muscles']['hamstrings_l']['activation']],[obs['muscles']['hamstrings_l']['fiber_length']],[obs['muscles']['hamstrings_l']['fiber_velocity']],[obs['muscles']['hamstrings_l']['fiber_force']],[obs['muscles']['bifemsh_l']['activation']],[obs['muscles']['bifemsh_l']['fiber_length']],[obs['muscles']['bifemsh_l']['fiber_velocity']],[obs['muscles']['bifemsh_l']['fiber_force']],[obs['muscles']['glut_max_l']['activation']],[obs['muscles']['glut_max_l']['fiber_length']],[obs['muscles']['glut_max_l']['fiber_velocity']],[obs['muscles']['glut_max_l']['fiber_force']],[obs['muscles']['iliopsoas_l']['activation']],[obs['muscles']['iliopsoas_l']['fiber_length']],[obs['muscles']['iliopsoas_l']['fiber_velocity']],[obs['muscles']['iliopsoas_l']['fiber_force']],[obs['muscles']['rect_fem_l']['activation']],[obs['muscles']['rect_fem_l']['fiber_length']],[obs['muscles']['rect_fem_l']['fiber_velocity']],[obs['muscles']['rect_fem_l']['fiber_force']],[obs['muscles']['vasti_l']['activation']],[obs['muscles']['vasti_l']['fiber_length']],[obs['muscles']['vasti_l']['fiber_velocity']],[obs['muscles']['vasti_l']['fiber_force']],[obs['muscles']['gastroc_l']['activation']],[obs['muscles']['gastroc_l']['fiber_length']],[obs['muscles']['gastroc_l']['fiber_velocity']],[obs['muscles']['gastroc_l']['fiber_force']],[obs['muscles']['soleus_l']['activation']],[obs['muscles']['soleus_l']['fiber_length']],[obs['muscles']['soleus_l']['fiber_velocity']],[obs['muscles']['soleus_l']['fiber_force']],[obs['muscles']['tib_ant_l']['activation']],[obs['muscles']['tib_ant_l']['fiber_length']],[obs['muscles']['tib_ant_l']['fiber_velocity']],[obs['muscles']['tib_ant_l']['fiber_force']],obs['misc']['mass_center_pos'],obs['misc']['mass_center_vel'],obs['misc']['mass_center_acc']]
    trueObs = []
    for i in range(len(newObs)):
        for j in range(len(newObs[i])):
            trueObs.append(newObs[i][j])
    return trueObs

trainer = TpgTrainer(actions=19, randSeed=1, actionRange=(-0.4,0.4,0.02), teamPopSizeInit=360)

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



