import matplotlib.pyplot as plt
import numpy as np
from dolfin import *
from dolfin_adjoint import * 
import sys
import os 
from time import time 
from util.misc import *
 
tol = 1e-14 
def boundary_r(x, on_boundary): 
    return on_boundary and (near(x[0], 1, tol))

def boundary_l(x, on_boundary): 
    return on_boundary and (near(x[0], 0, tol))

def boundary_t(x, on_boundary): 
    return on_boundary and (near(x[1], 1, tol))

def boundary_b(x, on_boundary): 
    return on_boundary and (near(x[1], 0, tol))

def boundary_well_1(x, on_boundary):
    tol = 0.01 
    return near(x[0], 0.5,  tol) and near(x[1], 0.3,  tol)

def boundary_well_2(x, on_boundary):
    tol = 0.01 
    return near(x[0], 0.5,  tol) and near(x[1], 0.7,  tol)


def boundary_well(x, on_boundary): 
    tol = 0.01 
    return near(x[0], 0.5,  tol) and near(x[1], 0.5,  tol)


def K(Y): 
    return exp(Y)

def S(Y): 
    return 0.1+0.05*Y

def forwardWellFunc( Y, V, simul_params): 
    set_log_level(LogLevel.ERROR)
    # get parameters 
    ngy = simul_params["ngy"] 
    ngx = simul_params["ngx"]
    lbc = simul_params["lbc"]
    rbc = simul_params["rbc"]
    wCond  = simul_params["wCond"] 
    # boundary conditions 
    u_l = Constant(lbc)
    u_r = Constant(rbc)
    u_w1 = Constant(0.0)
    u_w2 = Constant(0.3)
    bc_l = DirichletBC(V, u_l, boundary_l ) 
    bc_r = DirichletBC(V, u_r, boundary_r ) 
  #  bc_w1 = DirichletBC(V, u_w1, boundary_well_1, "pointwise")
  #  bc_w2 = DirichletBC(V, u_w2, boundary_well_2, "pointwise")
    bcs = [bc_l, bc_r]
  #  bcs = [bc_l, bc_r, bc_w1, bc_w2]
    # variational form 
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot( K(Y) * grad(u), grad(v)) * dx
    f = Constant(0.0) 
    L = f * v * dx 
    # solve PDE 
    u = Function(V)
    solve(a == L, u, bcs)
    return u 

def projectFunction(V, vec):
    Y = Function(V) 
    ordering = dof_to_vertex_map(V) 
    Y.vector()[:] = vec.flatten(order = 'C')[ordering]
    return Y    

def projectFunction_parallel(V, vec):
    dofmap = V.dofmap() 
    my_first, my_last = dofmap.ownership_range() 
    unowned = dofmap.local_to_global_unowned() 
    dofs = filter(lambda dof: dofmap.local_to_global_index(dof) not in unowned,xrange(my_last-myfirst)) 
    Y = Function(V) 
    ordering = dof_to_vertex_map(V) 
    values = vec.flatten(order = 'C')[ordering]
    values = values[dofs]
    Y.vector().set_local( values )
    return Y 

def forwardWellTransFunc(u_old, Y, V, simul_params): 
    set_log_level(LogLevel.ERROR)
    # get parameters 
    lbc = simul_params["lbc"]
    rbc = simul_params["rbc"]
    wCond  = simul_params["wCond"] 
    dt = simul_params["dt"]
    # boundary conditions 
    u_l = Constant(lbc)
    u_r = Constant(rbc)
    u_w = Constant(wCond)
    del_t = Constant(dt)
    bc_l = DirichletBC(V, u_l, boundary_l) 
    bc_r = DirichletBC(V, u_r, boundary_r)
  #  bc_w = DirichletBC(V, u_w, boundary_well, "pointwise") 
  #  bcs = [bc_l, bc_r, bc_w]
    bcs = [bc_l, bc_r]
    # variational form 
    u = TrialFunction(V)
    v = TestFunction(V)
    F = S(Y) * u * v * dx - S(Y) * u_old * v * dx + del_t * dot( K(Y) * grad(u), grad(v)) * dx
    a, L = lhs(F), rhs(F)
    # solve PDE 
    u = Function(V)
    solve(a == L, u, bcs)
    return u

def forwardWellTransFunc2(u_old, Y, V, simul_params): 
    set_log_level(LogLevel.ERROR)
    # get parameters 
    lbc = simul_params["lbc"]
    rbc = simul_params["rbc"]
    wCond  = simul_params["wCond"] 
    dt = simul_params["dt"]
    # boundary conditions 
    u_l = Constant(lbc)
    u_r = Constant(rbc)
    u_w = Constant(wCond)
    del_t = Constant(dt)
    bc_l = DirichletBC(V, u_l, boundary_l) 
    bc_r = DirichletBC(V, u_r, boundary_r)
    bc_w = DirichletBC(V, u_w, boundary_well, "pointwise") 
    bcs = [bc_l, bc_r, bc_w]
    # variational form 
    u = TrialFunction(V)
    v = TestFunction(V)
    F = S(Y) * u * v * dx - S(Y) * u_old * v * dx + del_t * dot( K(Y) * grad(u), grad(v)) * dx
    a, L = lhs(F), rhs(F)
    # solve PDE 
    u = Function(V)
    solve(a == L, u, bcs)
    return u


def writeHDF( m, simul_params, filename, input_dir): 
    ngx = simul_params["ngx"] 
    ngy = simul_params["ngy"]
    mesh = UnitSquareMesh(ngy - 1, ngx - 1) 
    V = FunctionSpace(mesh, "Lagrange", 1)
    Y = projectFunction(V, m)
    name = input_dir + filename + '.h5'
    hdf = HDF5File(mesh.mpi_comm(), name, "w") 
    hdf.write(mesh,"mesh")
    hdf.write(Y, "Y")
    order_v_dof = vertex_to_dof_map(V)
    order_dof_v = dof_to_vertex_map(V)
    hdf.close()
    return order_v_dof, order_dof_v
 
def forwardWellTrans(perm, simul_params): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    dt = simul_params["dt"] 
    num_steps = simul_params["num_steps"]
    u_ini = simul_params["ini_cond"] * np.ones((ngx*ngy, 1))
    u_sol = np.zeros((ngx*ngy, num_steps + 1))
    mesh = UnitSquareMesh(ngy - 1, ngx - 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    
    Y = projectFunction(V, perm)
    u_old = projectFunction(V, u_ini)
    u_sol[:,0:1] = u_ini
    
    for n in range(num_steps): 
        u = forwardWellTransFunc( u_old, Y, V, simul_params) 
        u_old = u
        u_sol[:,n+1:n+2] = u_old.compute_vertex_values(mesh).reshape(ngy*ngx, 1) 
    return u_sol



def forwardWellTrans2(perm, simul_params): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    dt = simul_params["dt"] 
    num_steps = simul_params["num_steps"]
    u_ini = simul_params["ini_cond"] * np.ones((ngx*ngy, 1))
    u_sol = np.zeros((ngx*ngy, num_steps + 1))
    mesh = UnitSquareMesh(ngy - 1, ngx - 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    
    Y = projectFunction(V, perm)
    u_old = projectFunction(V, u_ini)
    u_sol[:,0:1] = u_ini
    
    for n in range(num_steps): 
        u = forwardWellTransFunc2( u_old, Y, V, simul_params) 
        u_old = u
        u_sol[:,n+1:n+2] = u_old.compute_vertex_values(mesh).reshape(ngy*ngx, 1) 
    return u_sol

def forwardParallel( simul_params, filename, Dir): 
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"]
    dt = simul_params["dt"] 
    num_steps = simul_params["num_steps"]
    u_ini = simul_params["ini_cond"]
    mesh = UnitSquareMesh(ngy - 1, ngx - 1)
    mpi_comm = mesh.mpi_comm() 

    V = FunctionSpace(mesh, "Lagrange", 1)
    ordering = vertex_to_dof_map(V)
    Y = projectHDF( mpi_comm, V, filename, Dir) 

    u_old = Constant(u_ini)
    name = Dir + filename + '_out.h5'
    hdf = HDF5File( mpi_comm, name, "w" ) 
    hdf.write(mesh, "mesh")
    x = V.tabulate_dof_coordinates()[:, 0:1] 
    y = V.tabulate_dof_coordinates()[:, 1:2]
    output = np.concatenate((x, y), axis = 1) 
    for n in range(num_steps): 
        u = forwardWellTransFunc( u_old, Y, V, simul_params) 
        u_old = u
        hdf.write( u, "u/Vector/vector%d"%n )
        u_out = u.vector().get_local()[:, np.newaxis]
        output = np.concatenate((output, u_out), axis = 1)
    hdf.close()
    mpi_rank = MPI.rank(mpi_comm)
    name = Dir + filename + '_out' + str(mpi_rank) + '.txt'
    np.savetxt(name, output)
    return filename+'_out'
 
def projectHDF( mpi_comm, V, filename, input_dir):
    Y = Function(V)
    name = input_dir + filename + '.h5'
    hdf = HDF5File(mpi_comm, name, "r")
    hdf.read(Y, "Y")
    hdf.close() 
    return Y
 
def projectTstepHDF( mpi_comm, ind, V, filename, input_dir):
    u = Function(V)
    name = input_dir + filename + '.h5'
    hdf = HDF5File(mpi_comm, name, "r")
    hdf.read(u, "u/Vector/vector%d"%ind )
    hdf.close() 
    return u 

def forwardWell(perm, simul_params): 
    
    ngx = simul_params["ngx"]
    ngy = simul_params["ngy"] 
    
    mesh = UnitSquareMesh(ngy - 1, ngx - 1)
    V = FunctionSpace(mesh, "Lagrange", 1)
    Y = projectFunction(V, perm)
 
    u = forwardWellFunc( Y, V, simul_params) 
    u_out = u.compute_vertex_values(mesh).reshape(ngy*ngx, 1)
    
    return u_out 
'''
def main():
    hperm = 5
    lperm = 0 
    obs_params = { "nxblock_stat": 5, "nyblock_stat": 5, "nxblock_dyn": 5, "nyblock_dyn": 5 } 
    simul_params = {"ngx": 45, "ngy":45, "lbc": 1, "rbc" : 1, "wCond" : 0, "num_steps": 6, "dt" : 0.0006 } 
    input_dir = '/data/cees/hjyang3/PnP/data/'
    m_true,_ = getInput(input_dir, "channel_field.txt", 1) 
    m_true *= (hperm - lperm)
    m_true += lperm 
    u_ini = np.zeros((simul_params["ngy"] * simul_params["ngx"], 1))
    d_true = forwardWellTrans( m_true, u_ini, simul_params ) 
    plotFieldTstep(d_true, simul_params["ngx"], simul_params["ngy"], 1, "press_true", os.getcwd()) 
    plotFieldTstep(d_true, simul_params["ngx"], simul_params["ngy"], 3, "press_true", os.getcwd())
    plotFieldTstep(d_true, simul_params["ngx"], simul_params["ngy"], 6, "press_true", os.getcwd()) 
    print( " finish" ) 
if __name__ == '__main__':
    main()  
'''    
