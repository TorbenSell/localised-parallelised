#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 2 12:17:21 2019

@author: torbensell
"""

t_max = 5*3600 # Runtime

"""

Main programme for the Imaging example. Three algorithms are compared.

"""

from IDax_alg_MALA import MCMC as MCMC_v
from IDax_alg_axMALA import MCMC as MCMC_ax
from IDax_alg_DAaxMALA import MCMC as MCMC_da
from IDax_alg_MALA import MCMC_mse as MCMC_mse_v
from IDax_alg_axMALA import MCMC_mse as MCMC_mse_ax
from IDax_alg_DAaxMALA import MCMC_mse as MCMC_mse_da
import IDax_scenario as sc
import imageio
import numpy as np
import copy



"""
Initialise the truth, a starting point, and create observations.
"""

m = sc.m
u_true = sc.u_true
post_mean = np.load('ID/posterior_mean_estimate.npy')
y = sc.y


f = open('ID/output.txt','a')
f.write('\n\n\n\n\n')
f.write('m = ' + str(sc.m) + ', t_max = ' + str(t_max) + ', delta_k = ' + str(sc.delta_inner) + ', delta_h = ' + str(sc.delta_outer) + '.\n')
f.close()




"""
Run the MCMC chains, save samples.
"""


'''
Run MALA.
'''
print('\n\nNow starting MALA.')
u_start = np.load('ID/good_sample_MALA.npy')
[samples,average,acceptance_stats]=MCMC_v(u_start,y,m,t_max)

''' Print stats. '''
f = open('ID/output.txt','a')
f.write('Acceptances: ' + str(acceptance_stats[0]) + '\n')
f.write('Rejections: ' + str(acceptance_stats[1]) + '\n')
f.write('MSE: ' + str(np.linalg.norm(average-post_mean)) + '\n\n\n\n\n')
f.close()




'''
Run the asynchronous MCMC method.
'''
print('\n\nNow starting local approximate parallel MALA.')
u_start = np.load('ID/good_sample_axMALA.npy')
[samples,average,acceptance_stats]=MCMC_ax(u_start,y,m,t_max)

''' Print stats. '''
f = open('ID/output.txt','a')
f.write('Acceptances: ' + str(acceptance_stats[0]) + '\n')
f.write('Rejections: ' + str(acceptance_stats[1]) + '\n')
f.write('MSE: ' + str(np.linalg.norm(average-post_mean)) + '\n\n\n\n\n')
f.close()




'''
Run the delayed acceptance MCMC method.
'''
print('\n\nNow starting delayed acceptance parallel MALA.')
u_start = np.load('ID/good_sample_DAaxMALA.npy')
[samples,average,acceptance_stats]=MCMC_da(u_start,y,m,t_max)

''' Print stats. '''
f = open('ID/output.txt','a')
f.write('Acceptances: ' + str(acceptance_stats[0]) + '\n')
f.write('Rejections at first stage: ' + str(acceptance_stats[1]) + '\n')
f.write('Rejections at second stage: ' + str(acceptance_stats[2]) + '\n')
f.write('MSE: ' + str(np.linalg.norm(average-post_mean)) + '\n\n\n\n\n')
f.close()







# =============================================================================
# """
# Run the MCMC chains, estimate MSE/SSIM.
# """
# 
# u_start = sc.u_true
# 
# '''
# Run MALA.
# '''
# print('\n\nNow starting MALA.')
# acceptance_stats=MCMC_mse_v(copy.deepcopy(u_start),y,m,t_max)
# 
# ''' Print stats. '''
# f = open('ID/output.txt','a')
# f.write('Acceptances: ' + str(acceptance_stats[0]) + '\n')
# f.write('Rejections: ' + str(acceptance_stats[1]) + '\n')
# f.write('MSE: ' + str(np.linalg.norm(np.load('ID/good_sample_MALA1.npy')-post_mean)) + '\n\n\n\n\n')
# f.close()
# 
# 
# 
# '''
# Run the asynchronous MCMC method.
# '''
# print('\n\nNow starting local approximate parallel MALA.')
# acceptance_stats=MCMC_mse_ax(copy.deepcopy(u_start),y,m,t_max)
# 
# ''' Print stats. '''
# f = open('ID/output.txt','a')
# f.write('Acceptances: ' + str(acceptance_stats[0]) + '\n')
# f.write('Rejections: ' + str(acceptance_stats[1]) + '\n')
# f.write('MSE: ' + str(np.linalg.norm(np.load('ID/good_sample_axMALA1.npy')-post_mean)) + '\n\n\n\n\n')
# f.close()
# 
# 
# 
# '''
# Run the delayed acceptance MCMC method.
# '''
# print('\n\nNow starting delayed acceptance parallel MALA.')
# acceptance_stats=MCMC_mse_da(copy.deepcopy(u_start),y,m,t_max)
# 
# ''' Print stats. '''
# f = open('ID/output.txt','a')
# f.write('Acceptances: ' + str(acceptance_stats[0]) + '\n')
# f.write('Rejections at first stage: ' + str(acceptance_stats[1]) + '\n')
# f.write('Rejections at second stage: ' + str(acceptance_stats[2]) + '\n')
# f.write('MSE: ' + str(np.linalg.norm(np.load('ID/good_sample_DAaxMALA1.npy')-post_mean)) + '\n\n\n\n\n')
# f.close()
# =============================================================================







# =============================================================================
# """ Plot MSE and SSIM """
# 
# import matplotlib.pyplot as plt
# 
# SSIMs_MALA = np.load('ID/SSIM_MALA.npy')
# SSIMs_axMALA = np.load('ID/SSIM_axMALA.npy')
# SSIMs_DAaxMALA = np.load('ID/SSIM_DAaxMALA.npy')
# MSEs_MALA = np.load('ID/MSE_MALA.npy')
# MSEs_axMALA = np.load('ID/MSE_axMALA.npy')
# MSEs_DAaxMALA = np.load('ID/MSE_DAaxMALA.npy')
# MSE_steps = len(MSEs_MALA)
# 
# 
# plt.plot(range(1,1+MSE_steps),SSIMs_MALA,label='MALA')
# plt.plot(range(1,1+MSE_steps),SSIMs_axMALA,label='para-MwG')
# plt.plot(range(1,1+MSE_steps),SSIMs_DAaxMALA,label='para-DA')
# plt.legend()
# plt.savefig("ID/IDax_SSIM.pdf", dpi=300, bbox_inches='tight')
# plt.close()
# 
# plt.plot(range(1,1+MSE_steps),MSEs_MALA,label='MALA')
# plt.plot(range(1,1+MSE_steps),MSEs_axMALA,label='para-MwG')
# plt.plot(range(1,1+MSE_steps),MSEs_DAaxMALA,label='para-DA')
# plt.legend()
# plt.savefig("ID/IDax_MSE.pdf", dpi=300, bbox_inches='tight')
# plt.close()
# =============================================================================



print('\n\n\n\nDone.')