from osim.env import ProstheticsEnv

env = ProstheticsEnv(visualize=False)
observation = env.reset()
score = 0
for i in range(300):
    observation, reward, done, info = env.step([.5]*19)
    score += reward
    if done:
        break
print('reward: ' + str(reward))
env.close()
