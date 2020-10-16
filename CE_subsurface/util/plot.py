import matplotlib.pyplot as plt
import numpy as np
import sys 
import os 

def plotField(u, simul_params, Filename, Dir): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    u = u.reshape(ngy,ngx) 
    plt.figure()
    plt.imshow(u)
    plt.colorbar()
    plt.clim(0, 5)
    plt.savefig(Dir + Filename+'.png')
    
def plotUncertainty(i, j, u, u_true, channel_params,simul_params, Filename, Dir): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    output_tsteps = simul_params["output_steps"]
    nsteps = simul_params["num_steps"]
    n_samples = channel_params["num_ens"]
    x1 = np.linspace(0, 0.1, nsteps+1, endpoint = True)
    ind = (i-1) + (j-1) * ngx 
    qoi = u[ind, :, :] 
    qoi_true = u_true[ind, :]
    plt.figure()
    for i in range (n_samples): 
        plt.plot(x1, qoi[:, i], color = 'gray', linewidth=1)
    qoi_avg= np.mean(qoi, axis = 1 )
    plt.plot(x1, qoi_true, color = 'blue', linewidth = 3)
    plt.plot(x1, qoi_avg, color = 'r', linewidth = 3)
    plt.savefig(Dir + Filename + '.png')  

def plotFieldTstep(u, simul_params, Filename, Dir): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    output_tsteps = simul_params["output_steps"]
    for tstep in output_tsteps:
        u_plot = u[:, tstep]
        u_plot = u_plot.reshape(ngy,ngx) 
        plt.figure()
        plt.imshow(u_plot)
        plt.colorbar()
        Filename_ = Filename + "_T_%s" % tstep
        plt.savefig(Dir + Filename_ +  '.png')
   
def plotPoints(d_obs, obs_x, obs_y, Filename, Dir): 
    plt.figure()
    plt.scatter(obs_x, obs_y, c=d_obs)
    plt.gca().invert_yaxis() 
    plt.colorbar()
    plt.savefig(Dir + Filename + '.png')  
