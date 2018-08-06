from osim.env import ProstheticsEnv

env = ProstheticsEnv(visualize=True)
observation = env.reset()
for i in range(300):
    observation, reward, done, info = env.step([.5]*19)
    if done:
        break
env.close()