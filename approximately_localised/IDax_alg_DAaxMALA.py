#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue June 26 16:47:26 2018

@author: torbensell
"""


"""

This is a MCMC algorithm adapted to the example of solving an inverse problem,
here the Groundwater flow example with boundary conditions as in the talk
given by Beskos. 

"""

from multiprocessing import Process, Value, Array
import numpy as np
import time
import sys
import copy
import IDax_scenario as sc
from skimage.metrics import structural_similarity as ssim


"""
This MCMC algorithm is given a starting vector and a vector with observations
to calculate the likelihood. Optional arguments are m (the root of the
dimension of the state space), dimY (the number of observations), N_max (the
number of steps to be executed, the last N/2 samples are later returned).
"""

w = 4
N_current = 0

''' Create list with possible S-W corners '''
possible_corners_list = []  
for j in range(int(sc.m/sc.delta_inner)):
    for i in range(int(sc.m/sc.delta_inner)):
        possible_corners_list.append(tuple((int(i*sc.delta_inner),int(j*sc.delta_inner))))

master_list = []
for k in range(len(possible_corners_list)):
    pick = possible_corners_list[k]
    
    indices_resample_region = []
    for j in range(sc.delta_inner):
        for i in range(sc.delta_inner):
            indices_resample_region.append(int(pick[0]+i)%sc.m+sc.m*((pick[1]+j)%sc.m))
    
    indices_work_region = []
    for j in range(sc.delta_outer):
        for i in range(sc.delta_outer):
            indices_work_region.append(int(pick[0]-sc.delta_side+i)%sc.m+sc.m*(int(pick[1]-sc.delta_side+j)%sc.m))
    indices_cond_region = [x for x in indices_work_region if x not in indices_resample_region]
    
    master_list.append((pick,np.array(indices_resample_region),np.array(indices_cond_region),np.array(indices_work_region)))



"""
Main MCMC function, which is called by the main programme.
"""

def MCMC(u_current,obs_data,m, t_max = 60):
    
    global w
    
    '''Initialise list of samples and mean'''    
    samples = []
    no_samples = int(t_max/30) # one sample every 30 seconds
    j=0
    average = np.zeros(m**2)
    
    loglikelihood_current = sc.loglikelihood(u_current,obs_data,m)
    logprior_current = sc.logprior(u_current,m)
    acceptance_stats = np.zeros(3) #number of acceptances, rejections at first, rejections at second stage.
    
    '''Print start time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\nParallel DA MCMC algorithm with ' + str(w) + ' workers was started: ' + str(time.ctime()))
    f.write('\n\nInitial loglikelikehood: ' + str(round(loglikelihood_current,5)) + ', initial logprior: ' + str(round(logprior_current,5)) + '.')
    f.close()
    
    t_start = time.time()
    while(time.time()-t_start<t_max):
        u_current,loglikelihood_current,logprior_current,acceptance_stats = MCMC_step(u_current,loglikelihood_current,logprior_current,obs_data,acceptance_stats,m)
        progressBar(int(time.time()-t_start),t_max)
        if (time.time()-t_start)>j*t_max/no_samples:    
            samples.append(u_current)
            average+=u_current/no_samples
            j+=1
          
    samples = np.array(samples)
            
    np.save('ID/good_sample_DAaxMALA.npy',u_current)
    
    '''Print time needed in total and end time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\n\nMCMC algorithm terminated: ' + str(time.ctime()) + '\nTotal runtime was ' + str(time.time()-t_start) + ' seconds.')
    f.write('\n\nFinal loglikelikehood: ' + str(round(loglikelihood_current,5)) + ', final logprior: ' + str(round(logprior_current,5)) + '.\n\n\n')
    f.close()
    
    return(samples,average,acceptance_stats)
    
    
""" MCMC for MSE and SSIM calculation """
def MCMC_mse(u_current,obs_data,m, t_max = 60):
    
    global w
    
    '''Initialise mean'''   
    j=0
    mean = np.zeros(m**2)
    mean += u_current
    
    t_i = 0 # index for MSE estimation
    MSE_steps = int(t_max/MSE_time)
    MSE = np.zeros(MSE_steps) # store the MSEs here
    SSIM = np.zeros(MSE_steps) # store the SSIMs here
    
    loglikelihood_current = sc.loglikelihood(u_current,obs_data,m)
    logprior_current = sc.logprior(u_current,m)
    acceptance_stats = np.zeros(3) #number of acceptances, rejections at first, rejections at second stage.
    
    '''Print start time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\nParallel DA MCMC algorithm with ' + str(w) + ' workers was started: ' + str(time.ctime()))
    f.write('\n\nInitial loglikelikehood: ' + str(round(loglikelihood_current,5)) + ', initial logprior: ' + str(round(logprior_current,5)) + '.')
    f.close()
          
    t_start = time.time()
    while t_i<MSE_steps:
        u_current,loglikelihood_current,logprior_current,acceptance_stats = MCMC_step(u_current,loglikelihood_current,logprior_current,obs_data,acceptance_stats,m)
        mean = online_mean_estimation(u_current,mean,j,m)
        # Calculate MSE every MSE_time seconds
        if time.time()-t_start>MSE_time*t_i:
            MSE[t_i] = np.linalg.norm((mean-posterior_mean)**2)
            SSIM[t_i] = ssim(mean.reshape(m,m), posterior_mean.reshape(m,m),data_range=posterior_mean.max() - posterior_mean.min())
            t_i += 1
            progressBar(t_i, MSE_steps) 
        j+=1

#    np.save('ID/good_sample_DAaxMALA.npy',u_current)
    
    '''Print time needed in total and end time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\n\nMCMC algorithm terminated: ' + str(time.ctime()) + '\nTotal runtime was ' + str(time.time()-t_start) + ' seconds.')
    f.write('\n\nFinal loglikelikehood: ' + str(round(loglikelihood_current,5)) + ', final logprior: ' + str(round(logprior_current,5)) + '.\n\n\n')
    f.close()
    
    np.save('ID/mean_DAaxMALA.npy',mean)
    np.save('ID/MSE_DAaxMALA.npy',MSE)
    np.save('ID/SSIM_DAaxMALA.npy',SSIM)
    
    return acceptance_stats
    
    
    
"""
Parallelised MCMC step
"""  
    
def MCMC_step(u_current,loglikelihood_current,logprior_current,y,acceptance_stats,m=60):
    
    ''' SAMPLE REGIONS '''
    regions_list = []
    regions_list = sample_regions(w,m)
    
    
    ''' GET LOCAL PROPOSALS USING MULTIPLE PROCESSORS '''
    workers = []
    q_ratio_approx = []
    post_ratio_approx = []
    u_resampled_values = []
    
    for i in range(w): 
        q_ratio_approx.append(Value('d', 0.0))
        post_ratio_approx.append(Value('d', 0.0))
        u_resampled_values.append(Array('d', range(sc.delta_inner**2)))
        workers.append(Process(target=MCMC_local,args=(post_ratio_approx[i],q_ratio_approx[i],u_resampled_values[i],regions_list[i],u_current,loglikelihood_current,logprior_current,y,m)))
        workers[i].start()
    
    for i in range(w):
        workers[i].join(timeout=10.0)
        if workers[i].is_alive() == True:
            print('error')
        
    ''' MERGE LOCAL PROPOSALS TO ONE '''
    u_proposal = copy.deepcopy(u_current)
    for i in range(w):    
        u_proposal[regions_list[i][1]] = np.array(u_resampled_values[i][:])
    
    MH_ratio = 0
    for i in range(w):
        MH_ratio += q_ratio_approx[i].value+post_ratio_approx[i].value
    acceptance_probability_1st_stage = min(1,np.exp(MH_ratio))
    
    rand = np.random.uniform()
    
    if rand < acceptance_probability_1st_stage:
    
        '''CALCULATE EXACT PRIOR '''        
        logprior_proposal = sc.logprior(u_proposal,m)
        prior_ratio = logprior_proposal-logprior_current
    
        ''' CALCULATE EXACT LIKELIHOOD '''
        loglikelihood_proposal = sc.loglikelihood(u_proposal,y,m)
        likelihood_ratio = loglikelihood_proposal-loglikelihood_current
        
        ratio_second_stage = 0
        for i in range(w):
            ratio_second_stage += post_ratio_approx[i].value
    
        ratio_second_stage = prior_ratio+likelihood_ratio-ratio_second_stage
        acceptance_probability_2nd_stage = min(1,np.exp(ratio_second_stage))
        
        rand = np.random.uniform()
        if rand < acceptance_probability_2nd_stage:
            ''' Proposal accepted. '''
            acceptance_stats[0]+=1
            return u_proposal,loglikelihood_proposal,logprior_proposal,acceptance_stats
        
        else:
            ''' Proposal rejected at second stage. '''
            acceptance_stats[2]+=1
            return u_current,loglikelihood_current,logprior_current,acceptance_stats
    else:
        ''' Proposal rejected at first stage. '''
        acceptance_stats[1]+=1
        return u_current,loglikelihood_current,logprior_current,acceptance_stats
    
    

"""
This is a systematic way to define the work regions.
"""
def sample_regions(w,m):
    global N_current
    regions_list = []
    difference = int((m/sc.delta_inner)**2/w)
    number_of_possibilities = int((m/sc.delta_inner)**2)
    
    ''' Add all corners '''
    for i in range(w):       
        corner,indices_resample_region,indices_cond_region,indices_work_region = master_list.pop((difference+N_current)%number_of_possibilities)
        regions_list.append((corner,indices_resample_region,indices_cond_region,indices_work_region))
        master_list.append((corner,indices_resample_region,indices_cond_region,indices_work_region))
        
    N_current += 1
    return regions_list

    
"""
The next function is a local MCMC step, which can run in parallel if the given 
regions don't overlap.
"""
    
def MCMC_local(post_ratio_approx,q_ratio_approx,u_resampled_values,regions_list,u_current,loglikelihood_current,logprior_current,y,m,time_elapsed_for_approximations_p=0, time_elapsed_for_resampling_p=0):
    
    delta = 0.00003 
    
    ''' RESAMPLING '''
    indices_resample_region = regions_list[1]
    indices_work_region = regions_list[3]
    indices_cond_region = regions_list[2]
    
    '''Actual resampling'''
    u_curr = u_current[indices_resample_region]
    u_cond = u_current[indices_cond_region]
    y_work = y[indices_work_region]
    grad_ll = sc.grad_ll_loc(u_curr,u_cond,y_work,m)
    grad_prior = sc.grad_prior_loc(u_curr,u_cond,m)
    grad_log_post_loc = grad_ll+grad_prior
    u_resampled_values[:] = sc.propose_loc(u_curr,grad_log_post_loc,delta,m)
 
    
    ''' CALCULATE APPROXIMATE LIKELIHOOD '''
    
    '''Calculate local likelikihood'''    
    loglikelihood_local_approx = sc.loglikelihood_local(np.array(u_resampled_values[:]),u_cond,y_work,m)
    loglikelihood_current_local = sc.loglikelihood_local(u_current[indices_resample_region],u_cond,y_work,m)
    
    '''Calculate local prior'''    
    logprior_local_approx = sc.logprior_local(np.array(u_resampled_values[:]),u_cond,m)
    logprior_current_local = sc.logprior_local(u_current[indices_resample_region],u_cond,m)
    
    post_ratio_approx.value = logprior_local_approx+loglikelihood_local_approx-logprior_current_local-loglikelihood_current_local
    
    ''' Calc q_loc-ratio ''' 
    grad_ll = sc.grad_ll_loc(np.array(u_resampled_values[:]),u_cond,y_work,m)
    grad_prior = sc.grad_prior_loc(np.array(u_resampled_values[:]),u_cond,m)
    grad_log_post_loc_proposal = grad_ll+grad_prior
    q_1 = sc.log_q_loc(u_current[indices_resample_region],np.array(u_resampled_values[:]),grad_log_post_loc_proposal,delta,m)
    q_2 = sc.log_q_loc(np.array(u_resampled_values[:]),u_current[indices_resample_region],grad_log_post_loc,delta,m)
    q_ratio_approx.value = q_1-q_2

    

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