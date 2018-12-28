# prosthetic-challenge-tpg
My work for a competition on crowdAI: https://www.crowdai.org/challenges/neurips-2018-ai-for-prosthetics-challenge. I only did the first phase of the competition, and I did not obtain decent results until after the competition. I used the Tangled Program Graphs algorithm: https://github.com/Ryan-Amaral/PyTPG.

## Results Visualization
Results from a run that was stopped prematurely, due to a desire to save Google Cloud Platform credits (and the run was done after the competition was already over).

![Results Graph](https://github.com/Ryan-Amaral/prosthetic-challenge-tpg/blob/master/pros-res.jpg)

## The Files that matter
### train-tpg.py
Trains a TPG model on the task with multiprocessing.
### train-log-...
The results log of the most recent (best) run.
