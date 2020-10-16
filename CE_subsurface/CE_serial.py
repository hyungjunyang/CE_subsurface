# Main file for CE framework  (Serial version) 
# This main file does not support parallel computing
# CE framework integrates three different agents for one consistent image reconstruction 
# Agent1: adjoint method, Agent2: CNN denoiser, Agent3: CNN based VAE model
# #####################################################################################  

import numpy as np 
import dolfin as df 
import sys 
import os 
from time import time 
from util.misc import *
from util.plot import * 
from util.sampling import * 
from util.forward import *
from model.posterior import *
from model.denoisePrior import * 
from model.initial import *
from model.dncnnPrior import * 
from model.vaePrior2 import *

# get Reference image and its simulation results
def getChannelTrue( channel_params, simul_params, input_file, input_dir):
    m_true,_ = getInput(input_dir, input_file, 1)
    m_true *= (channel_params["hperm"] - channel_params["lperm"]) 
    m_true += channel_params["lperm"] 
    d_true = forwardWellTrans( m_true, simul_params)
    print(d_true.shape) 
    return m_true, d_true

# calculate misfit between true data and forward simulation result of current data 
def dataMisfit( d_obs, m, obs_vec, simul_params): 
    d = forwardWellTrans( m, simul_params)
    nstep = simul_params["num_steps"]
    obj = 0
    for n in range(nstep):
        d_ = d[:, n+1:n+2]
        d_obs_ = d_obs[:, n+1:n+2]
        obj += 0.5 * np.dot((obs_vec*(d_-d_obs_)).T, obs_vec*(d_-d_obs_))
    obj /= nstep 
    return obj  
  
# residual to check convergence 
def residual(m_old, m_new):
    N = m_old.shape[0] 
    res = np.linalg.norm(m_old - m_new) / N
    return res

def CE_p_d_v( input_dir, simul_params, channel_params, obs_params, pnp_params, ind ): 
    start_time = time()
    Ref_file = "Ref.DAT"
    MPS_file = "Train.DAT" 
    m_true, d_obs = getChannelTrue( channel_params, simul_params, Ref_file, input_dir)
    log_file = "log_" + str(ind) + ".txt" 
    _, obs_vec = obsMatrix( obs_params["nxblock_dyn"], obs_params["nyblock_dyn"], simul_params["ngx"], simul_params["ngy"] ) 
    f = open( input_dir + log_file, "w" ) 
    # read initials 
    if (channel_params["initial"] == "Gauss"):
        m_ini = gaussInitial( m_true, channel_params, simul_params, obs_params ) 
    elif (channel_params["initial"] == "SVR"): 
        m_ini = svmInitial( m_ture, channel_params, simul_params, obs_params ) 
    else: 
        m_ini = getMPSInitial( ind, channel_params, simul_params, MPS_file, input_dir ) 
    m_1 = m_ini
    m_2 = m_ini
    m_3 = m_ini
    m_new = m_ini
    q = 1
    while True: 
        m_old = m_new
        f.write( "PnP-DR Iteration: %d \n" %q )
        # data loss proximal mapping 
        m_1_new = 2 * posteriorTransProx( d_obs, m_1, m_1, obs_vec, channel_params, simul_params, pnp_params ) - m_1 
        # image denosing  
        if ( pnp_params["denoiser"] == "bm3d"): 
            m_2_new = bm3dPrior(m_2, simul_params, pnp_params)
        elif (pnp_params["denoiser"] == "TV"): 
            m_2_new = tvPrior(m_2, simul_params)
        elif (pnp_params["denoiser"] == "dncnn"):
            m_2_new = dncnnPrior(m_2, channel_params, simul_params)
        m_2_new = 2 * m_2_new - m_2
        # VAE based proximal mapping 
        m_3_new = 2 * vaeGeologyPrior(m_3, channel_params, simul_params, pnp_params) - m_3
        # synchronize three different agents 
        m_1 = (1 - pnp_params["rho"]) * m_1 + pnp_params["rho"]/3 * ( -1 * m_1_new + 2 * m_2_new + 2 * m_3_new )  
        m_2 = (1 - pnp_params["rho"]) * m_2 + pnp_params["rho"]/3 * ( 2 * m_1_new + -1 * m_2_new + 2 * m_3_new ) 
        m_3 = (1 - pnp_params["rho"]) * m_3 + pnp_params["rho"]/3 * ( 2 * m_1_new + 2 * m_2_new - 1 * m_3_new ) 
        m_new = (m_1 + m_2 + m_3)/3
        res = pnpResidual( m_old, m_new ) 
        obj, d_new = printDataLoss( d_obs, m_new, obs_vec, simul_params ) 
        f.write("Data misfit: %6.2f, residual: %2.6f \n"%(obj,res))
        # convergence check 
        if ( res < pnp_params["tol"] or q > pnp_params["max_iter"] ): 
           break
        q += 1  
    f.write("Realization number %d, total time : %5.2f s \n" %(ind, time()-start_time))
    f.close()
    # save output 
    np.savetxt( input_dir + "m_new_" + str(ind) + ".txt", m_new)
             
def main():
    input_file = "Input.txt" 
    # read input  
    input_dir, simul_params, channel_params, obs_params, pnp_params = readInput( input_file )
    if ( channel_params["initial"] == MPS ):
        # Monte Carlo : randomly sample the initial and apply CE serially 
        rand_ind = np.random.choice(channel_params["num_ens"],channel_params["num_sample"] )
        for i in range ( channel_params["num_samples"] ) : 
            CE_p_d_v( input_dir, simul_params, channel_params, obs_params, pnp_params, rand_ind[i] )
    else: 
        # only one CE based inmage reconstruction (deterministic) 
        CE_p_d_v( input_dir, simul_params, channel_params, obs_params, pnp_params, 0 )
  
if __name__ == '__main__': 
    main() 
