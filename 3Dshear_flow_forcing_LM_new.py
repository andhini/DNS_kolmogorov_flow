"""
Dedalus script simulating a 3D periodic incompressible shear flow beased on shear a flow example in Dedalus package
DNS is based on results published by by Lalescu et. al. in  https://doi.org/10.1103/PhysRevLett.110.084102


    nu = 1 / Reynolds
    D = nu / Schmidt

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shear_flow.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""
## command : mpiexec -n 8 python3 3Dshear_flow_forcing_LM_new.py
## if continues : mpiexec -n 8 python3 3Dshear_flow_forcing_LM_new.py --restart (check restart_dir)
import numpy as np
import dedalus.public as d3
import logging, sys
from dedalus.tools.parallel import Sync
from sys import exit
logger = logging.getLogger(__name__)

def main():
    # Allow restarting via command line
    restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')
    restart_dir = 'checkpoints/checkpoints_s3.h5'

    # Parameters
    Lx, Ly, Lz = 6.*np.pi,2.*np.pi,2.*np.pi
    Nx, Ny, Nz = 144,48,48   
    Reynolds = 30
    dealias = 3/2
    stop_sim_time = 83
    timestepper = d3.RK443
    max_timestep = 1.e-1
    dtype = np.float64
    A = 1.  # amplitude of forcing
    kx = 1. # frequency of X forcing along Y 

    # Bases
    coords = d3.CartesianCoordinates('x', 'y', 'z')
    dist = d3.Distributor(coords,dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)
    zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

    # Fields
    p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    u_ave = dist.VectorField(coords, name='u_ave', bases=(xbasis,ybasis,zbasis))
     # Es_ave = dist.TensorField(coords, name='Es_ave', bases=(xbasis,ybasis,zbasis))
    
    
    # forcing
    f = dist.VectorField(coords, name='f', bases=(xbasis,ybasis,zbasis))

    # Substitutions
    nu = 1. / Reynolds 
    x, y, z = dist.local_grids(xbasis, ybasis, zbasis)
    ex, ey, ez = coords.unit_vector_fields(dist)

    # Problem
    problem = d3.IVP([u, p, tau_p], namespace=locals()) 
    problem.add_equation("dt(u) + grad(p) - nu*lap(u) = - u@grad(u) + f")
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time

    # Background shear
    f['g'][0] = A * np.sin(kx*y)
    f['g'][1] = 0.
    f['g'][2] = 0.

    # Initial conditions
    u_ave['g'] = 0. 
    # Es_ave['g'] =0.
    if not restart:
        u.fill_random('g', seed=42, distribution='normal', scale=A*0.01) # Random noise
        u.low_pass_filter(scales=0.5)
        file_handler_mode = 'overwrite'
        initial_timestep = max_timestep
        
    else:
        write, initial_timestep = solver.load_state(restart_dir)
        initial_timestep = max_timestep*0.1
        file_handler_mode = 'append'
        ## read inputs from checkpoints
        drank = dist.comm.rank
        dsize = dist.mesh.tolist()
        u_ave_global=None;
        # Es_ave_global=None;
        dist.comm.Barrier()
        with Sync() as sync:
            if (drank == 0): 
                # u_ave_global,Es_ave_global=read_inputs(restart_dir)
                u_ave_global=read_inputs(restart_dir)
                u_split=np.split(u_ave_global,dsize[0],axis=2) #chunk on second axis
                # Es_split= np.split(Es_ave_global,dsize[0],axis=3) #chunk on second axis
                if(np.shape(dsize)[0]==2):
                    for j in range(len(u_split)): u_split[j]=np.split(u_split[j],dsize[1],axis=3); #chunk on third axis
                    u_split=np.concatenate(u_split,axis=0); #merge all chunks
                    # for j in range(len(Es_split)): Es_split[j]=np.split(Es_split[j],dsize[1],axis=4); #chunk on third axis
                    # Es_split=np.concatenate(Es_split,axis=0); #merge all chunks 
                ## sending data from root to ranks
                for i in range(np.prod(dsize)):
                    if (i == 0):
                        u_ave['g']=u_split[i]
                        # Es_ave['g']=Es_split[i]
                    else:
                        dist.comm.send(u_split[i],dest=i,tag=1)
                        # dist.comm.send(Es_split[i],dest=i,tag=1)
            else:
                u_ave['g']=dist.comm.recv(source=0,tag=1)
                # Es_ave['g']=dist.comm.recv(source=0,tag=1)

    ## Turbulence properties
    # Es = (d3.grad(u) + d3.transpose(d3.grad(u))) / 2 # strain rate  tensor
    # Ss = (d3.grad(u) - d3.transpose(d3.grad(u))) / 2 # rotation rate tensor
    # Ts = Es**2+Ss**2; # find the eigen values of Ts, the second biggest gives lambda2, then add to snapshots
     
    uprime = u - u_ave # vectors
    Es_prime = (d3.grad(u - u_ave) + d3.transpose(d3.grad(u - u_ave))) / 2 # strain rate  tensor of u_prime
    TKE = 0.5*d3.DotProduct(uprime,uprime) #fields
    # epsilon = d3.integ(2*nu*(Es_prime)**2.,('x','y','z'))/(Lx*Ly*Lz) 
    # Re_lambda = (5/3)**0.5 *2.*d3.Average(TKE,('x','y','z'))/(nu*epsilon)**0.5 #value
    
     # Analysis
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=2, max_writes=10, mode=file_handler_mode)
    snapshots.add_task(p, name='pressure')
    snapshots.add_task(u,name='velocity')
    snapshots.add_task(uprime,name='u_prime')
    snapshots.add_task(Es_prime,name='Es_prime')
    # snapshots.add_task(((nu**3.)/epsilon.evaluate()['g'])**(1./4.),name='eta')
    
    # #averaging checking, no need to write
    snapshots.add_task(u_ave,name='u_average')
    # snapshots.add_task(Es,name='Es')
    # snapshots.add_task(Es_ave,name='Es_average')

    #checkpoints
    checkpoints = solver.evaluator.add_file_handler('checkpoints', sim_dt=50, max_writes=1, mode=file_handler_mode)
    checkpoints.add_tasks(solver.state)
    checkpoints.add_task(u_ave,name='u_ave')
    # checkpoints.add_task(Es_ave,name='Es_ave')

    # CFL
    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.07,
                max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property((u@u)**2, name='u2')
    flow.add_property((u_ave-u), name='u_ave_check')

    # Main loop
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)

            ## moving averaging variables, compute per iteration
            it = solver.iteration
            if(it>0):
                u.change_scales(1)
                u_ave.change_scales(1)
                u_ave['g'] = (u_ave['g']*(it-1)+u['g'])/it
                # Es_ave['g'] = (Es_ave['g']*(it-1)+Es['g'])/it
            else:
                u_ave['g'] = u['g']
                # Es_ave['g'] = Es['g']

            if (solver.iteration-1) % 10 == 0:
                max_u = np.sqrt(flow.max('u2'))
                check_uave = flow.grid_average('u_ave_check')
                logger.info('Iteration=%i, Time=%e, dt=%e, max(u)=%.3e,u-u_ave=%.3e, ' \
                            %(solver.iteration, solver.sim_time, timestep, max_u, check_uave))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

## serial reading file and mount the variables : u_ave and Es_ave
def read_inputs(dir_in):
    import h5py, os, re
    mypath=os.getcwd()  
    # tasks = ['u_ave','Es_ave']
    tasks = ['u_ave']
    with h5py.File(mypath+'/'+dir_in, mode='r') as fi:
        # for n,task in enumerate(tasks):
        #     # print(n,task,np.shape(fi['tasks'][task][-1,:]))
        #     if(n==0):
        #         u_ave_out=fi['tasks'][task][-1,:]
        #     else: 
        #         Es_ave_out=fi['tasks'][task][-1,:]
        u_ave_out=fi['tasks'][tasks[0]][-1,:]
    # print(np.shape(u_ave_out), np.shape(Es_ave_out))  
    return u_ave_out#,Es_ave_out 

if __name__=='__main__':
    main()