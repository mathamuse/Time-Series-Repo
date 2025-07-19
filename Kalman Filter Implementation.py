############################## INCLUSIONS AND HEADERS #########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#################### UPDATE FUNCTION ############################
## Position Update function
def pos_update(ape,cobs,alpha,step):
    print("we are at POS UPDATE step",step," the previous prediction was:",ape)
    print("The current observation is", cobs)
    val = ape + alpha*(cobs-ape)
    print("we estimate the POS to be ",val, "with alpha ",alpha)
    return val

# velocity update function
def vel_update(ape,appos,cobs,beta,dt,step):
    print("we are at VEL UPDATE step",step," the previous prediction was:",ape)
    print("The current observation is", cobs)
    val = ape + (beta*(cobs-appos)/dt)
    print("we estimate the VEL to be ",val, "with beta ",beta)
    return val

#################### PREDICT FUNCTION ##########################
def pos_predict(app,apv,dt):
    val = app+(dt*apv)
    return val

def vel_predict(app,apv,dt):
    return apv

#################### DEFINING OUR SYSTEM ########################
steps = 51
dt=1;
alpha = 0.9;
beta = 0.1;
idx=np.arange(1,steps,1)
#print(idx)
############# CREATE A FUNCTION TO FOLLOW #######################
trueval = 30000 + (40)*idx
#print(trueval)

# the noisy distribution of observations #
obs = trueval + np.random.normal(0,20,steps-1)
#print(obs)

pos=[];
vel=[];
pos.insert(0,30000);
vel.insert(0,40)
pred_pos = pos_predict(pos[0], vel[0], dt)
vel_pred = vel_predict(pos[0],vel[0],dt)
print(pred_pos,vel_pred)
#'''
############## RUNNING THE FILTER ################
for i in range(1,steps-1):
    pos.insert(i, pos_update(pred_pos, obs[i], alpha, i))
    vel.insert(i,vel_update(vel_pred,pos[i-1] , obs[i], beta, dt, i))
    pred_pos = pos_predict(pos[i],vel[i], dt)
    vel_pred = vel_predict(pos[i], vel[i], dt)   

############# PLOTTING OUTPUT ###############

plt.plot(trueval,label ="Trueval")
plt.plot(obs,label='Obs')
plt.plot(pos,label='State')
plt.legend()
plt.show()

error_pos = pos-trueval
error_obs=obs-trueval
zero1 = trueval-trueval
plt.plot(zero1)
plt.plot(error_pos,label='Filter Error')
plt.plot(error_obs,label='Noise')
plt.legend()
plt.show()
#'''
