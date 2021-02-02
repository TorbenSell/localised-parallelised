#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 13:46:25 2018

@author: torbensell
"""


"""

This is an approximate MCMC algorithm using parallelisation.

"""

import numpy as np
import time
import sys
import ID_scenario as sc
from skimage.metrics import structural_similarity as ssim
import numba as nb



"""
This MCMC algorithm is given a starting vector and a vector with observations
to calculate the likelihood. Optional arguments are m (the root of the
dimension of the state space), dimY (the number of observations), t_max (the
maximum running time), the last N/2 samples are later returned.
"""


"""
Main MCMC function, which is called by the main programme.
"""

def MCMC(u_current,obs_data, m, t_max = 60):
    
    '''Initialise list of samples and mean'''    
    samples = []
    no_samples = int(t_max/15) # one sample every 15 seconds
    j=0
    average = np.zeros(m**2)
    loglikelihood_current = sc.loglikelihood(u_current,obs_data,m)
    logprior_current = sc.logprior(u_current,m)
    acceptance_stats = np.zeros(2) #number of acceptances, rejections.
        
    '''Print start time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\nVanilla MCMC algorithm was started: ' + str(time.ctime()))
    f.write('\n\nInitial loglikelikehood: ' + str(round(loglikelihood_current,5)) + ', initial logprior: ' + str(round(logprior_current,5)) + '.')
    f.close()
    
    t_start = time.time()
    
    while(time.time()-t_start<t_max):
        u_current,loglikelihood_current,logprior_current,acceptance_stats = MCMC_step(u_current,loglikelihood_current,logprior_current,obs_data,acceptance_stats,m)
        progressBar(int(time.time()-t_start),t_max)
        if (time.time()-t_start)>j*t_max/no_samples:    
#            samples.append(u_current)
            average+=u_current/no_samples
            j+=1
          
#    samples = np.array(samples)
#    average = average/len(samples)
    np.save('ID/good_sample_MALA.npy',u_current)
    np.save('ID/posterior_mean_estimate.npy',average)
    
    t_end = time.time()
    
    '''Print time needed in total and end time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\n\nMCMC algorithm terminated: ' + str(time.ctime()) + '\nTotal runtime was ' + str(t_end-t_start) + ' seconds.')
    f.write('\n\nFinal loglikelikehood: ' + str(round(loglikelihood_current,5)) + ', final logprior: ' + str(round(logprior_current,5)) + '.\n\n\n')
    f.close()
    
    return(samples,average,acceptance_stats)

""" MCMC for MSE and SSIM calculation """
def MCMC_mse(u_current,obs_data, m, t_max = 60):
    
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
    acceptance_stats = np.zeros(2) #number of acceptances, rejections.
        
    '''Print start time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\nVanilla MCMC algorithm was started: ' + str(time.ctime()))
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
          
#    np.save('ID/good_sample_MALA.npy',u_current)
    t_end = time.time()
    
    '''Print time needed in total and end time for the MCMC algorithm.'''
    f = open('ID/output.txt','a')
    f.write('\n\nMCMC algorithm terminated: ' + str(time.ctime()) + '\nTotal runtime was ' + str(t_end-t_start) + ' seconds.')
    f.write('\n\nFinal loglikelikehood: ' + str(round(loglikelihood_current,5)) + ', final logprior: ' + str(round(logprior_current,5)) + '.\n\n\n')
    f.close()
    
    np.save('ID/mean_MALA.npy',mean)
    np.save('ID/MSE_MALA.npy',MSE)
    np.save('ID/SSIM_MALA.npy',SSIM)
    
    return acceptance_stats
    
    
    
"""
Standard MALA MCMC step.
"""  
#@nb.njit()
def MCMC_step(u_current,loglikelihood_current,logprior_current,y,acceptance_stats,m):
    
    ''' RESAMPLING '''   
    delta = 0.0000055
    
    grad_ll_u = sc.grad_post(u_current,y,m)
    u_proposal = sc.propose(u_current,grad_ll_u,delta,m)
    
    '''CALCULATE PRIOR '''    
    logprior_proposal = sc.logprior(u_proposal,m)
    prior_diff = logprior_proposal-logprior_current
        
    ''' CALCULATE EXACT LIKELIHOOD '''       
    loglikelihood_proposal = sc.loglikelihood(u_proposal,y,m)
    likelihood_diff = loglikelihood_proposal-loglikelihood_current

    ''' CALCULATE TRANSITION PROBABILITIES '''
    q_1 = sc.log_q(u_current,u_proposal,sc.grad_post(u_proposal,y,m),delta,m)
    q_2 = sc.log_q(u_proposal,u_current,grad_ll_u,delta,m)
    q_diff = q_1-q_2
    
    MH_ratio = np.exp(prior_diff+likelihood_diff+q_diff)
    acceptance_probability = min(1,MH_ratio)
    
    rand = np.random.uniform()
    
    if rand < acceptance_probability:
        acceptance_stats[0]+=1
        return u_proposal,loglikelihood_proposal,logprior_proposal,acceptance_stats
    else:       
        acceptance_stats[1]+=1
        return u_current,loglikelihood_current,logprior_current,acceptance_stats
    
    


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