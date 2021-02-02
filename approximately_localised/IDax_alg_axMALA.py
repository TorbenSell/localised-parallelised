#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 15:34:52 2018

@author: torbensell
"""


"""

This is an approximate MCMC algorithm using parallelisation.

"""

from multiprocessing import Process, Value, Array, Lock, Manager
import numpy as np
import time
import sys
import IDax_scenario as sc
from skimage.metrics import structural_similarity as ssim


"""
This approximate MCMC algorithm is given a starting vector and a vector with
observations to calculate the likelihood. Optional arguments are m (the root of
the dimension of the state space), and t_max (maximum runtime given to the
algorithm).
"""

finish = Value('i', 0)
w = 4

''' Create list with possible S-W corners '''
possible_corners_list = []  
for j in range(int(sc.m/sc.delta_inner)):
    for i in range(int(sc.m/sc.delta_inner)):
        possible_corners_list.append(tuple((int(i*sc.delta_inner),int(j*sc.delta_inner))))

master_list = []
for k in range(len(possible_corners_list)):
    pick = possible_corners_list[k]
    
    ''' regions_to_block stores the corner values of those regions that should be touched while working on pick. '''
    regions_to_block = []
    for j in range(3):
        for i in range(3):
            regions_to_block.append(tuple((int(pick[0]+(i-1)*sc.delta_inner)%sc.m,int(pick[1]+(j-1)*sc.delta_inner)%sc.m)))
    
    indices_resample_region = []
    for j in range(sc.delta_inner):
        for i in range(sc.delta_inner):
            indices_resample_region.append(int(pick[0]+i)%sc.m+sc.m*(int(pick[1]+j)%sc.m))
    
    indices_work_region = []
    for j in range(sc.delta_outer):
        for i in range(sc.delta_outer):
            indices_work_region.append(int(pick[0]-sc.delta_side+i)%sc.m+sc.m*(int(pick[1]-sc.delta_side+j)%sc.m))
    indices_cond_region = [x for x in indices_work_region if x not in indices_resample_region]
    
    master_list.append((pick,regions_to_block,np.array(indices_resample_region),np.array(indices_cond_region),np.array(indices_work_region)))
blocked_regions_list = []


"""
Main MCMC function, which is called by the main programme.
"""

def MCMC(u_current,obs_data, m, t_max = 60):
    global finish, blocked_regions_list, master_list, w
   
    u = Array('d',u_current)
    
    '''Initialise list of samples and mean'''    
    samples = []
    no_samples = int(t_max/30) # one sample every 30 seconds
    average = np.zeros(m**2)
    
    finish = Value('i', 0)
    acceptance_stats = Array('d',np.zeros(2)) #number of acceptances, and rejections.
    
    '''Print start time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\nApproximate MCMC algorithm with ' + str(w) + ' workers was started: ' + str(time.ctime()))
    f.write('\n\nInitial loglikelikehood: ' + str(round(sc.loglikelihood(np.array(u[:]),obs_data,m),5)) + ', initial logprior: ' + str(round(sc.logprior(np.array(u[:]),m),5)) + '.')
    f.close()
    
    t_start = time.time()
    
    lock = Lock()
    with Manager() as manager:
        master_list = manager.list(master_list)
        
        ''' sampling '''
        for j in range(no_samples):
            blocked_regions_list = manager.list([])
            finish.value = 0
            workers = []
            for i in range(w):
                workers.append(Process(target=local_MCMC,args=(lock,u,obs_data,finish,acceptance_stats,blocked_regions_list, master_list,m))) 
                workers[i].start()
            time.sleep(t_max/no_samples)
            finish.value = 1
            for i in range(w):
                workers[i].join(timeout=10.0)
            for i in range(w):
                if workers[i].is_alive() == True:
                    print('error')
            samples.append(np.array(u[:]))
            average += np.array(u[:])/no_samples
            progressBar(j+1, no_samples)
    
    t_end = time.time()
          
    samples = np.array(samples)
            
#    np.save('ID/good_sample_axMALA.npy',u)
    
    '''Print time needed in total and end time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\n\nMCMC algorithm terminated: ' + str(time.ctime()) + '\nTotal runtime was ' + str(t_end-t_start) + ' seconds.')
    f.write('\n\nFinal loglikelikehood: ' + str(round(sc.loglikelihood(np.array(u[:]),obs_data,m),5)) + ', final logprior: ' + str(round(sc.logprior(np.array(u[:]),m),5)) + '.\n\n\n')
    f.close()
    
    return(samples,average,acceptance_stats[:])
    
""" MCMC for MSE and SSIM calculation """
def MCMC_mse(u_current,obs_data, m, t_max = 60):
    global finish, blocked_regions_list, master_list, w
   
    u = Array('d',u_current)
    
    '''Initialise mean'''    
    mean = np.zeros(m**2)
    mean += u_current
    
    t_i = 0 # index for MSE estimation
    MSE_steps = int(t_max/MSE_time)
    MSE = np.zeros(MSE_steps) # store the MSEs here
    SSIM = np.zeros(MSE_steps) # store the SSIMs here
    
    finish = Value('i', 0)
    acceptance_stats = Array('d',np.zeros(2)) #number of acceptances, and rejections.
    
    '''Print start time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\nApproximate MCMC algorithm with ' + str(w) + ' workers was started: ' + str(time.ctime()))
    f.write('\n\nInitial loglikelikehood: ' + str(round(sc.loglikelihood(np.array(u[:]),obs_data,m),5)) + ', initial logprior: ' + str(round(sc.logprior(np.array(u[:]),m),5)) + '.')
    f.close()
    
    t_start = time.time()
    
    lock = Lock()
    with Manager() as manager:
        master_list = manager.list(master_list)
        
        ''' sampling '''
        while t_i<MSE_steps:
            MSE[t_i] = np.linalg.norm((mean-posterior_mean)**2)
            SSIM[t_i] = ssim(mean.reshape(m,m), posterior_mean.reshape(m,m),data_range=posterior_mean.max() - posterior_mean.min())
            progressBar(t_i+1, MSE_steps) 
            
            blocked_regions_list = manager.list([])
            finish.value = 0
            workers = []
            for i in range(w):
                workers.append(Process(target=local_MCMC,args=(lock,u,obs_data,finish,acceptance_stats,blocked_regions_list, master_list,m))) 
                workers[i].start()
            time.sleep(MSE_time)
            finish.value = 1
            for i in range(w):
                workers[i].join(timeout=10.0)
                
            mean = online_mean_estimation(np.array(u[:]),mean,t_i,m)
            t_i += 1
            
    np.save('ID/good_sample_axMALA.npy',u)
    
    t_end = time.time()
    
    '''Print time needed in total and end time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\n\nMCMC algorithm terminated: ' + str(time.ctime()) + '\nTotal runtime was ' + str(t_end-t_start) + ' seconds.')
    f.write('\n\nFinal loglikelikehood: ' + str(round(sc.loglikelihood(np.array(u[:]),obs_data,m),5)) + ', final logprior: ' + str(round(sc.logprior(np.array(u[:]),m),5)) + '.\n\n\n')
    f.close()
    
    np.save('ID/mean_axMALA.npy',mean)
    np.save('ID/MSE_axMALA.npy',MSE)
    np.save('ID/SSIM_axMALA.npy',SSIM)
    
    return acceptance_stats[:]
    
    
"""
Parallelised MCMC step
"""  
    
def local_MCMC(lock,u_current,y,finish,acceptance_stats,blocked_regions, master_list,m):
    '''
    This is the routine for a single worker!
    '''
    delta = 0.00002 # Tuning parameter
    with lock:
        reserved_regions,indices_resample_region,indices_cond_region,indices_work_region = new_region(blocked_regions, master_list) # Initial work region
    
    u_curr = np.zeros(len(indices_resample_region))
    u_cond = np.zeros(len(indices_cond_region))
    
    while(finish.value == 0):
        y_work = y[indices_work_region]
        
        ''' Propose '''
        for i in range(len(indices_resample_region)):
            u_curr[i] = u_current[indices_resample_region[i]]
        for i in range(len(indices_cond_region)):
            u_cond[i] = u_current[indices_cond_region[i]]
        
        grad_log_post_loc = sc.grad_ll_loc(u_curr,u_cond,y_work,m)+sc.grad_prior_loc(u_curr,u_cond,m)
        u_proposal = sc.propose_loc(u_curr,grad_log_post_loc,delta,m)
        u_proposal = np.array(u_proposal)
        
        ''' Evaluate likelihood (potentially approximate) '''
        loglikelihood_prop = sc.loglikelihood_local(u_proposal,u_cond,y_work,m)
        loglikelihood_current = sc.loglikelihood_local(u_curr,u_cond,y_work,m)
        likelihood_diff = loglikelihood_prop-loglikelihood_current
        
        ''' Evaluate prior (potentially approximate) '''
        logprior_prop = sc.logprior_local(u_proposal,u_cond,m)
        logprior_current = sc.logprior_local(u_curr,u_cond,m)
        prior_diff = logprior_prop-logprior_current
        
        ''' Compute transition probabilities '''
        grad_log_post_loc_prop = sc.grad_ll_loc(u_proposal,u_cond,y_work,m)+sc.grad_prior_loc(u_proposal,u_cond,m)
        q_1 = sc.log_q_loc(u_curr,u_proposal,grad_log_post_loc_prop,delta,m)
        q_2 = sc.log_q_loc(u_proposal,u_curr,grad_log_post_loc,delta,m)  
        q_diff = q_1-q_2
    
        ''' Acccept or reject '''
        acceptance_probability = min(1,np.exp(likelihood_diff+prior_diff+q_diff))
        if np.random.uniform() < acceptance_probability:
            ''' Proposal accepted. '''
            acceptance_stats[0]+=1
            for i in range(len(indices_resample_region)):
                u_current[indices_resample_region[i]] = u_proposal[i]
                
        else:
            ''' Proposal rejected. '''
            acceptance_stats[1]+=1
        
        ''' Sample new region '''
        with lock:
            reserved_regions,indices_resample_region,indices_cond_region,indices_work_region = next_region(reserved_regions,blocked_regions, master_list) 
            

def new_region(blocked_regions, master_list):
    ''' Pick new regions '''  
    i=0
    corner = (-1,-1)
    while corner == (-1,-1):
        if master_list[i][0] in blocked_regions:
            i += 1
        else:
            corner,reserved_regions,indices_resample_region,indices_cond_region,indices_work_region = master_list.pop(i)
            master_list.append((corner,reserved_regions,indices_resample_region,indices_cond_region,indices_work_region))
    
    ''' Block surrounding regions '''        
    blocked_regions.extend(reserved_regions)
    return reserved_regions,indices_resample_region,indices_cond_region,indices_work_region
      

def next_region(reserved_regions,blocked_regions, master_list):
    ''' Remove old blockings '''        
    for i in range(len(reserved_regions)):
        blocked_regions.remove(reserved_regions[i])
        
    ''' Pick new regions '''    
    i=0
    corner = (-1,-1)
    while corner == (-1,-1):
        if master_list[i][0] in blocked_regions:
            i += 1
        else:
            corner,reserved_regions,indices_resample_region,indices_cond_region,indices_work_region = master_list.pop(i)
            master_list.append((corner,reserved_regions,indices_resample_region,indices_cond_region,indices_work_region))
    
    ''' Block surrounding regions '''  
    blocked_regions.extend(reserved_regions)
    return reserved_regions,indices_resample_region,indices_cond_region,indices_work_region


        
"""
Function plotting the progress.
"""
def progressBar(value, endvalue, bar_length=40):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()
    
#@numba.njit()
def online_mean_estimation(x,x_hat,i,m):
    delta_x = np.zeros(m**2)
    delta_x = x-x_hat
    x_hat = x_hat+delta_x/(i+2) # x_hat is initialised at first sample, and i is initialised at 0
    return x_hat

try:
    MSE_time = 15 # number of seconds between estimation of MSE/SSIM
    posterior_mean = np.load('ID/posterior_mean_estimate.npy')
except FileNotFoundError:
    print('No posterior mean found.')
    posterior_mean = np.load('ID/good_sample_MALA.npy')