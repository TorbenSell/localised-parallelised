#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 15:58:32 2018

@author: torbensell
"""


"""

This is an Image Denoising Example as found in [Morzfeld, Tong, Marzouk 2019].

"""


import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numba as nb



# =============================================================================
# 
# Global functions. They operate on the entire domain.
#     
# =============================================================================

"""
The observations are created by adding noise to the truth.
"""
def observe(u_true,m):
    y = np.dot(H,u_true)+np.random.normal(size=m**2,scale=sigma)
    return y

"""
The prior on u is a multivariate Gaussian with sparse cov. matrix. We don't 
evaluate the prior, but it's logarithm (ignoring a constant).
"""
@nb.njit()
def logprior(u,m):
    logpdf = -np.dot(u,np.dot(Omega,u))/2
    return logpdf

@nb.njit()
def grad_logprior(u,m):
    grad_logpdf = -np.dot(Omega,u)
    return grad_logpdf

def sample_prior(m):
    u = np.dot(U,np.random.multivariate_normal(np.zeros(m**2),np.diag(D)))   
    return u


"""
The Likelihood for u given the data is a multivariate normal. We again only
evaluate the loglikelihood up to a constant.
"""
@nb.njit()
def loglikelihood(u,y,m):
    diff = np.dot(H,u)-y
    LL = -1/sigma**2*np.dot(diff,diff)/2
    return LL

@nb.njit()
def grad_loglikelihood(u,y,m):
    diff = np.dot(H,u)-y
    grad_ll = -1/sigma**2*np.dot(H.T,diff)
    return grad_ll

@nb.njit()
def grad_post(u,y,m):
    return grad_loglikelihood(u,y,m)+grad_logprior(u,m)


"""
MALA proposal and density q(.,.)
"""
def propose(u,grad_ll_u,delta,m):
    u_proposal = u+delta/2*grad_ll_u+np.sqrt(delta)*np.random.normal(size=m**2)
    return u_proposal

@nb.njit()
def log_q(u_new,u,grad_ll_u,delta,m):
    mean = u_new-u-delta/2*grad_ll_u
    log_q = -np.dot(mean,mean)/2/delta
    return log_q


# =============================================================================
# 
# Local funtions. They only operate on a subset of the entire domain, which
# allows parallelisation.
# 
# =============================================================================
    
"""
The Likelihood for u given the data is a multivariate normal. We again only
evaluate the loglikelihood up to a constant, and here only locally. Also, all
gradients are gradients of the log!
"""
@nb.njit()
def logprior_local(u_local,u_cond,m):
    diff = u_local-np.dot(np.ascontiguousarray(C_AB),np.dot(np.ascontiguousarray(C_BB_inv),np.ascontiguousarray(u_cond)))
    logpdf = -np.dot(diff,np.dot(Omega_AA,diff))/2  
    return logpdf

@nb.njit()
def loglikelihood_local(u_local,u_cond,y_work,m):
    diff_local = np.dot(H_work_local,u_local)-(y_work-np.dot(H_work_cond,u_cond))
    LL = -1/sigma**2*np.dot(diff_local,diff_local)/2
    return LL


"""
Calculate local gradients.
"""
@nb.njit()
def grad_prior_loc(u_local,u_cond,m):
    diff = u_local-np.dot(np.ascontiguousarray(C_AB),np.dot(np.ascontiguousarray(C_BB_inv),np.ascontiguousarray(u_cond)))
    grad_prior = -np.dot(Omega_AA,diff)
    return grad_prior
    

@nb.njit()
def grad_ll_loc(u_local,u_cond,y_work,m):
    diff_local = np.dot(H_work_local,u_local)-(y_work-np.dot(H_work_cond,u_cond))
    grad_ll = -1/sigma**2*np.dot(H_work_local.T,diff_local)
    return grad_ll

"""
Globalise integrates a local sample into the global one.
"""
@nb.njit()
def globalise_u(u,u_local,i_min,j_min,m):   
    u_new = np.zeros(m**2)
    u_new += u
    for j in range(delta_inner):
         for i in range(delta_inner):
             u_new[i_min+i+m*(j_min+j)] = u_local[i+delta_inner*j]
    return u_new


"""
local MALA proposal and density q_local(.,.)
"""
def propose_loc(u_local,grad_log_post_local,delta,m):
    u_proposal_local = u_local+delta/2*grad_log_post_local+np.sqrt(delta)*np.random.normal(size=delta_inner**2)
    return u_proposal_local

@nb.njit()
def log_q_loc(u_1_local,u_2_local,grad_log_post_2_local,delta,m):
    diff = u_1_local-u_2_local-delta/2*grad_log_post_2_local
    log_q_local = -np.dot(diff,diff)/2/delta
    return log_q_local



"""
Creates some heatmaps.
"""
def plots(k,l,name_k,name_l,name_method,m):   
    
    k_matrix = np.zeros((m,m))
    k_matrix = np.reshape(k,(m,m))
    l_matrix = np.zeros((m,m))
    l_matrix = np.reshape(l,(m,m))
    
    a = plt.figure()
    plt.subplot(131,aspect='equal')
    sns.heatmap(k_matrix)
    plt.title(name_k)
    plt.axis('off')
    
    plt.subplot(132,aspect='equal')
    sns.heatmap(l_matrix)
    plt.title(name_l)
    plt.axis('off')
    
    plt.subplot(133,aspect='equal')
    sns.heatmap(np.abs(k_matrix-l_matrix))
    plt.title('|' + name_k + '-' + name_l + '|')
    plt.axis('off')
    
    a.savefig('ID/Animation_and_plots/' + name_k + '_' + name_l + '_' + name_method + '.png', bbox_inches='tight')
    plt.close(a)




"""
MAIN INITIALISATIONS.
"""


  
"""
Define problem size. The deltas are the extra amount to cover for the sides.
"""   
m = 128
dimY = m**2
delta_outer = 16
delta_inner = 8 # make sure this divides m for the systematic parallel implementation
delta_side = int((delta_outer-delta_inner)/2) # ideally delta_side*2 = delta_inner for the implementation of local_MCMC




"""
Define prior mean and covariance matrix, as well as precision matrx.
"""
sigma = 10**-(2.5)  # Noise in observations
u_bar = np.zeros(m**2)  # mean of GP


''' Define covariance matrix, and store it in a file. '''
try:
    Omega = np.load('ID/cov_inv.npy')
    C = np.load('ID/cov_C.npy')
    U = np.load('ID/cov_U.npy')
    D = np.load('ID/cov_D.npy')
    H = np.load('ID/H.npy')
    
except FileNotFoundError:
    ''' The precision matrix is calculated as in Morzfeld et al, Ex. 5.2 '''
    L = np.zeros((m**2,m**2))
    for i in range(m):
        for j in range(m):
            L[i+m*j,i+m*j] = 4
            L[i+m*j,(i+m-1)%m+m*j] = -1
            L[i+m*j,(i+1)%m+m*j] = -1
            L[i+m*j,i+m*((j+m-1)%m)] = -1
            L[i+m*j,i+m*((j+1)%m)] = -1
    
    '''Calculate the svd of the covariance, and the precision matrices.'''
    D,U = np.linalg.eigh(L)

    print('smallest eigenvalue D: ',np.min(D))
    if np.min(D)<0:
        D = D-10*np.min(D)*np.ones(m**2)
    D = D*10
        
    C = np.matmul(U,np.matmul(np.diag(D**-1),U.T))
    Omega = np.matmul(U,np.matmul(np.diag(D),U.T))
    
    '''Save the important matrices.'''
    np.save('ID/cov_C.npy',C)
    np.save('ID/cov_U.npy',U)
    np.save('ID/cov_D.npy',D)
    np.save('ID/cov_inv.npy',Omega)

    ''' Initialise Blurring kernel '''
    H = np.zeros((m**2,m**2))
    for j in range(m):
        for i in range(m):
            for k in range(5):
                for l in range(5): 
                    H[i+m*j,((i-2+k)%m)+m*((j-2+l)%m)] = 1/25
    np.save('ID/H.npy',H)
    
    
    
    
    
"""
Define local operators.
"""
    
indices_resample_region = []
for j in range(delta_inner):
    for i in range(delta_inner):
        indices_resample_region.append(delta_side+i+m*(delta_side+j))

indices_work_region = []
for j in range(delta_outer):
    for i in range(delta_outer):
        indices_work_region.append(i+m*j)
indices_cond_region = [x for x in indices_work_region if x not in indices_resample_region]

C_AA = C[:,indices_resample_region]
C_AA = C_AA[indices_resample_region,:]
C_AB = C[:,indices_cond_region]
C_AB = C_AB[indices_resample_region,:]
C_BB_inv = C[:,indices_cond_region]
C_BB_inv = C_BB_inv[indices_cond_region,:]
C_BB_inv = np.linalg.inv(C_BB_inv)

Omega_AA = Omega[:,indices_resample_region]
Omega_AA = Omega_AA[indices_resample_region,:]
C_loc = np.linalg.inv(Omega_AA)
D_loc,U_loc = np.linalg.eigh(C_loc)

H_work_local = H[indices_work_region,:]
H_work_cond = H_work_local[:,indices_cond_region] 
H_work_local = H_work_local[:,indices_resample_region]



"""
Define true u.
"""
try:
    u_true = np.load('ID/u_true.npy')
    y = np.load('ID/observations.npy')
    plots(u_true,y,'u_true','y','',m)
    print('u_true was loaded. Logprior(u_true) = ',logprior(u_true,m),'. Loglikelihood(u_true) = ',loglikelihood(u_true,observe(u_true,m),m))
except FileNotFoundError:
    img_full = cv2.imread('ID/image_true.png',0)
    
    img_true = img_full[:,0:m]
    img_true = img_true[0:m,:]
    
    img_arr = img_true.reshape((1,m**2))[0]
    img_arr = img_arr*1.00
    
    u_true = img_arr
    y = observe(u_true,m)
    plots(u_true,y,'u_true','y','',m)
    np.save('ID/u_true.npy',u_true)
    np.save('ID/observations.npy',y)
    print('u_true was saved. Logprior(u_true) = ',logprior(u_true,m),'. Loglikelihood(u_true) = ',loglikelihood(u_true,observe(u_true,m),m))
    
    
    
"""
Import image.
"""

#img_full = cv2.imread('ID/image_true.png',0)
#
#img_true = img_full[:,0:m]
#img_true = img_true[0:m,:]
#
#img_arr = img_true.reshape((1,m**2))[0]
#img_arr = img_arr*1.00
#
#plt.imshow(img_arr.reshape((m,m)),cmap ='gray')
#plt.savefig('ID/img_true.png', bbox_inches='tight')
#
#plt.figure()
#img_obs = observe(img_arr,m)
#plt.imshow(img_obs.reshape((m,m)),cmap ='gray')
#plt.savefig('ID/img_obs.png', bbox_inches='tight')