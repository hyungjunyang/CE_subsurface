### Plot result and compute the error 
### Please execute CE_serial.py or CE_parallel.py first 
############################################################## 

import numpy as np 
import dolfin as df 
import sys 
import os 
from time import time 
from util.misc import *
from util.plot import * 
from util.sampling import * 
from util.forward import def 

def getChannelTrue( channel_params, simul_params, input_file, input_dir):
    m_true,_ = getInput(input_dir, input_file, 1)
    m_true *= (channel_params["hperm"] - channel_params["lperm"]) 
    m_true += channel_params["lperm"] 
    d_true = forwardWellTrans( m_true, simul_params)
    return m_true, d_true

def dataMisfit( d, d_obs, obs_vec, simul_params): 
    nstep = simul_params["num_steps"]
    obj = 0
    for n in range(nstep):
        d_ = d[:, n+1:n+2]
        d_obs_ = d_obs[:, n+1:n+2]
        obj += 0.5 * np.dot((obs_vec*(d_-d_obs_)).T, obs_vec*(d_-d_obs_))
    obj /= nstep
    return obj   

def PSNR(m_true, m_update): 
    N = m_true.shape[0] 
    mse = (np.square(m_true-m_update)).mean(axis = 1)
    MAX_I = np.max(m_update)
    psnr = 20 * np.log10(MAX_I) - 10 * np.log10(mse) 
    return psnr, mse 

def threshold(m_true, m_update, channel_params): 
    threshold = 0.5 * (channel_params["hperm"] + channel_params["lperm"])
    m_thres = (m_update > threshold).astype(float) 
    m_thres *= channel_params["hperm"] 
    misclass = np.abs(m_true - m_thres)
    misclass_ = (misclass > 1e-1).astype(int) 
    N_mis = np.sum(misclass_) 
    return m_thres, N_mis 
       
def getPlots():
    input_file = "Input.txt" 
    Ref_file = "Ref.DAT"
    MPS_file = "Train.DAT" 
    # read inputfile 
    input_dir, simul_params, channel_params, obs_params, pnp_params = readInput( input_file )
    n_samples = channel_params["num_sample"]
    m_true, d_obs = getChannelTrue( channel_params, simul_params, Ref_file, data_dir)
    if ( channel_params["initial"] = "MPS" ): 
    	m_new_final = np.zeros((simul_params["ngx"] * simul_params["ngy"], n_samples))
    	m_ini_final = np.zeros((simul_params["ngx"] * simul_params["ngy"], n_samples))
   	for ind in range (n_samples):
        	filename = output_dir + "m_new_" + str( ind+1 ) + ".txt"
        	m_new_final[:,ind] = np.loadtxt( filename )
       		m_ini_final[:,ind:ind+1] = getMPSInitial( ind+1, channel_params, simul_params, MPS_file, data_dir ) 
        m_ini_avg = np.mean( m_ini_final, axis = 1, keepdims = True )
    	m_new_avg = np.mean( m_new_final, axis = 1, keepdims = True )      
        plotField( m_ini_avg, simul_params, "initial_field_avg", output_dir )
        plotField( m_new_avg, simul_params, "updated_field_avg", output_dir )
        plotField( m_true, simul_params, "true_field", output_dir ) 
    else: 
        filename = output_dir + "m_new_" + str( ind+1 ) + ".txt"
        m_new_final[:,ind] = np.loadtxt( filename )

if __name__ == '__main__':
    getPlots() 
