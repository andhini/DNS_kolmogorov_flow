"""
Plot outputs

Usage:
    plot_outputs.py [--ndims=<ndims>] <path>

options:
    --ndims=<ndims>     number of dimensions [default: 3]

"""
# command example: python3 plot_outputs.py --ndims=3 ./folder/snapshots/

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import h5py, os, re
import pathlib
from docopt import docopt
from sys import exit
from dedalus.tools.parallel import Sync
import time
import sys

def main(ndims, args):
    # Parameters
    tasks = ['velocity','Es_prime','u_prime']#don't change the order! u_prime is always last, add new before u_prime 
    figsize = (6,10)
    ylims=[[None,None],[None,None],[1E-3,None]]
    xlims=[[None,None],[None,None],[1E-2,2]]
    # ylims=[[-0.001,100],[None,0.01],[None,None],[None,None],[1e-3,None]]
  
    figs = 5
    fig,axs = plt.subplots(figs,1  ,constrained_layout=True,figsize=figsize)

    ##------variables---------##
    nu = 1/30   #-->change here for Re
    kc = 0.15 #----> cut-off frequency, change ">" or "<" in line:331,339,222
    #-------------------------##

    ## get all *.h5 files, store fullname is 'sd'
    sd,myKeys = get_h5files(args['<path>'])
       
    ## store data into variable 'stores' for 0:time, 1:tasks1, 2:tasks2, 3:tasks3..., 
    ## and grid arrays into 'grids'. Need parallel process! 
    stores,grids = store_data(tasks,sd,myKeys)
        
    for key in stores: 
        print('key:',key, ' shape',np.shape(stores[key]))
     
    ## calculate dissipation rate (eps) and Kolmogorov lengthscale (eta), their average and arrays
    eta,eta_arrays,eps,eps_arrays = compute_dissipation_kolmogorov_len(nu,stores['Es_prime'])
   
    ## calculate TKE and Taylor-microscale Reynolds number
    compute_TKE_Re_Taylor(nu,eps,stores['u_prime'])
 
    ## plot several scalars
    scalars=[eps_arrays,eta_arrays]; 
    scalar_names=[r'$\epsilon$',r'$\eta$']
    plot_scalars(figs,axs,stores['time'],scalars,scalar_names)
    
    ## calculate energy spectrum, from u_prime (last tasks in stores)
    Lx=3.*2.*np.pi; Ly = 2.*np.pi; Lz= 2.*np.pi # domain lengths
    if(ndims==2):
        [nx,ny]=list(np.shape(stores['u_prime'][0,0,:,:])); 
        nz=1
        Lz=0
        twoD=True
    else:
        [nx,ny,nz]=list(np.shape(stores['u_prime'][0,0,:,:,:]))
        twoD=False
    k,Ek,Ekcut,u_recon = compute_average_energy_spectrum(stores['u_prime'],nx,ny,nz,Lx,Ly,Lz,grids,dk=(1./3.),twoD=twoD,eta=eta,kc=kc)
    # Re,eta,eps,vel_x = meanvalues
    
    ## plot spectra
    plot_spectra(figs,axs,nu,eta,eps,k,Ek,Ekcut,kc,nx,ny,nz,xlims,ylims)
      
    ## plot contour of instantaneous mid-z slice of EK plots
    plot_inst_Ek(fig,axs,stores['u_prime'],u_recon,grids,nz)
      
    # plt.tight_layout()
    plt.show()  
    # fig.savefig('scalar_outputs.pdf',dpi=100)

def get_h5files(pathfile):
      #get all file names ended with *.h5
    files={}
    for file in os.listdir(pathfile):
        if file.endswith(".h5"):
            sf = os.path.join(pathfile, file)
            m = re.search('snapshots_s(.*).h5', sf)
            num= m.group(1)
            # if(int(num)>2):continue
            files[int(num)]=sf

    myKeys = list(files.keys())
    myKeys.sort()

    # Sorted Dictionary
    sd_out = {i: files[i] for i in myKeys}
    # print('opening',len(sd_out),sd_out)
    return sd_out,myKeys

def check_h5file_integrity(filename):
    """Check if an HDF5 file is corrupted or not."""
    try:
        with h5py.File(filename, 'r') as f:
            # Try to access basic metadata which should work if file is not corrupted
            f.keys()
            return True
    except (RuntimeError, OSError, IOError) as e:
        print(f"Error checking file {filename}: {str(e)}")
        return False

def store_data(tasks, sd, myKeys):
    """Store data from HDF5 files with robust error handling."""
    stores = {}  # 0:time, 1:tasks1, 2:tasks2, 3:tasks3...
    grids = {}   # 0:x-grid, 1:y, 2:z
    valid_keys = []
    corrupted_files = []
    
    # First check all files for integrity
    print(f"Checking integrity of {len(sd)} HDF5 files...")
    for ki in sd:
        filename = sd[ki]
        if check_h5file_integrity(filename):
            valid_keys.append(ki)
        else:
            corrupted_files.append(filename)
    
    if corrupted_files:
        print(f"WARNING: Found {len(corrupted_files)} corrupted files that will be skipped:")
        for f in corrupted_files:
            print(f"  - {f}")
        
        if not valid_keys:
            print("ERROR: No valid files found. Cannot proceed.")
            sys.exit(1)
        
        print(f"Continuing with {len(valid_keys)} valid files.")
    
    # Process only the valid files
    total_files = len(valid_keys)
    
    for idx, ki in enumerate(valid_keys):
        filename = sd[ki]
        print(f"Processing file {idx+1}/{total_files}: {os.path.basename(filename)}", end="\r")
        
        try:
            with h5py.File(filename, mode='r') as fi:
                # Handle time data
                try:
                    time_data = np.array(fi['scales/sim_time'])
                    
                    if 'time' not in stores:
                        stores['time'] = time_data
                        
                        # Only initialize grids on the first valid file
                        try:
                            scale_list = list(fi['scales'].keys())
                            scale_names = ['scales/'+scale_list[d-3] for d in range(ndims)]
                            for d in range(ndims):
                                grids[d] = np.array(fi[scale_names[d]][:]).tolist()  # get_1d_vertices gives vertex, np.array gives the center points.
                        except Exception as e:
                            print(f"\nError reading grid data: {str(e)}")
                            if not grids:
                                print("ERROR: Failed to initialize grids. Cannot proceed.")
                                sys.exit(1)
                    else:
                        stores['time'] = np.append(stores['time'], time_data)
                except Exception as e:
                    print(f"\nWarning: Error reading time data from {filename}: {str(e)}")
                    continue  # Skip to next file if time data can't be read

                # Process each task
                for i in range(len(tasks)):
                    task = tasks[i]
                    try:
                        # First check if task exists in the file
                        if 'tasks' not in fi or task not in fi['tasks']:
                            print(f"\nWarning: Task '{task}' not found in file {filename}")
                            continue
                            
                        # For 3D data, read in chunks to avoid memory issues
                        if (ndims == 3):
                            # Get dataset shape without loading data
                            task_shape = fi['tasks'][task].shape
                            # Create empty array of the right shape
                            task_data = np.empty(task_shape, dtype=fi['tasks'][task].dtype)
                            
                            # Safely read the data
                            task_data = fi['tasks'][task][...]
                        else:
                            # For 2D data
                            task_data = fi['tasks'][task][...]

                        # Store the data
                        if tasks[i] not in stores:
                            stores[tasks[i]] = task_data
                        else:
                            stores[tasks[i]] = np.append(stores[tasks[i]], task_data, axis=0)
                    except Exception as e:
                        print(f"\nWarning: Error reading task '{task}' from {filename}: {str(e)}")
                        # Continue with next task rather than skipping the whole file
                        continue
        except Exception as e:
            print(f"\nError processing file {filename}: {str(e)}")
            continue
    
    print("\nData loading complete.                        ")
    return stores, grids

def compute_dissipation_kolmogorov_len(nu,s_ij):
    ## calculate dissipation rate epsilon and eta manually from Es_prime
    
    epsilon_0 = 2.*nu*(s_ij*s_ij); 
    epsilon_0 = np.apply_over_axes(np.sum, epsilon_0, [1,2]);
    epsilon_r = np.empty((np.shape(epsilon_0)[0])); 
    for k in range(np.shape(epsilon_0)[0]):
                # ensemble average over volume space func(z,y,x)= 1/domain_volume*(triple_integ of epsilon_0*grid_volume) 
                epsilon_r[k]=np.mean(epsilon_0[k,:])
    print('shape eps_r', np.shape(epsilon_r))#,epsilon_r)
    epsilon_manual=np.ravel(epsilon_r) 
    eta_manual = ((nu**3.)/epsilon_manual)**(1./4.); 
    eta= np.mean(eta_manual[int(len(eta_manual)/2):]);
    eps=np.mean(epsilon_manual[int(len(epsilon_manual)/2):])
    print('manual epsilon:',eps,'eta:',eta,'resolution:',2*np.pi*(1/np.shape(epsilon_0)[4])/eta)#,'eta list',eta_manual)

    return eta,eta_manual,eps,epsilon_manual

def compute_TKE_Re_Taylor(nu,eps,uprimes):
    ax_domain=(1,2)
    if(ndims==3): ax_domain=(1,2,3)
    uprime_std = np.nanstd(uprimes[:,0,:],axis=ax_domain)
    vprime_std = np.nanstd(uprimes[:,1,:],axis=ax_domain)
    if(ndims==3): wprime_std = np.nanstd(uprimes[:,2,:],axis=ax_domain)
    if(ndims==3):
        TKE_manual = np.multiply((uprime_std**2+vprime_std**2+wprime_std**2),0.5)
    else:
        TKE_manual = np.multiply((uprime_std**2+vprime_std**2),0.5)
    TKE_manual_average = np.mean(TKE_manual[int(np.shape(TKE_manual)[0]/2.):])
    print('manual TKE=',TKE_manual_average, 'Re_lambda =', np.sqrt(5/3)*2*TKE_manual_average/np.sqrt(nu*eps) )

def compute_average_energy_spectrum(vel_batch,Nx,Ny,Nz,Lx,Ly,Lz,grids,dk=1.,k_max= None,twoD=False,eta=0.1,kc=0.15):
    # if len(vel_batch.shape) == 3: 
    #     vel_batch = vel_batch[jnp.newaxis, ...]

    # setup Fourier grid
    all_kx = Lx * np.fft.fftfreq(Nx, 3*Lx/ Nx)
    all_ky = Ly * np.fft.fftfreq(Ny, Ly/ Ny)
    
    # NNx=grids[0];NNy=grids[1]
    # all_kx = Lx*np.fft.fftfreq(len(NNx), (max(NNx)-min(NNx))/len(NNx)) # if using original grids
    # all_ky = Ly*np.fft.fftfreq(len(NNy), (max(NNy)-min(NNy))/len(NNy)) # if using original grids

    if(twoD==False):
        all_kz = Lz * np.fft.fftfreq(Nz, Lz/ Nz)
        # NNz=grids[2]
        # all_kz = Lz*np.fft.fftfreq(len(NNz), (max(NNz)-min(NNz))/len(NNz)) # if using original grids
        kx_mesh,ky_mesh,kz_mesh =np.meshgrid(all_kz,all_ky,all_kx)
    else:
        kx_mesh,ky_mesh =np.meshgrid(all_kx,all_ky)
    
    # print(np.shape(kx_mesh),'all_kx:',all_kx[:])
    # print(np.shape(ky_mesh),'all_ky:',all_ky[:])
    # if(twoD==False): print(np.shape(kz_mesh),'all_kz:',all_kz[:4])
    
    if k_max == None: k_max = int(1.5* np.max(np.array([np.max(all_kx),np.max(all_ky)])))#,np.max(all_kz)

    midbatch=int(len(vel_batch)/4); # only sampling from a quarter of batch len (converging parts)
    if(twoD==False):
        abs_wavenumbers = np.sqrt(kx_mesh**2 + ky_mesh**2 +kz_mesh**2).transpose(2,0,1) #for non-cubic array, transpose == non-transpose if cubic.
        # print('3D wavenumbers',np.shape(abs_wavenumbers))#,abs_wavenumbers[:4,:4,:4])
        # construct spectral energy
        vel_batch_ft  = np.fft.fftn(vel_batch[midbatch:,:,:,:,:], axes=(2,3,4)) # Dedalus3D velocity:[t,n,x,y,z]    
        # print('vel batch_ft size', np.shape(vel_batch_ft), 'before ftt', np.shape(vel_batch))

        ke_in_kxky = 0.5 * np.sum(np.abs(vel_batch_ft * vel_batch_ft.conj()),
                    axis=1) / (np.array(Nx*Ny*Nz, dtype=np.float64)**2) # 0.5<uu>, then average over all axis x,y,z
        # print('ke shape',np.shape(ke_in_kxky),ke_in_kxky[0,:1,:1,:1],np.shape(vel_batch)) # gives (sample,Nx,Ny,Nz)
    else:
        abs_wavenumbers = np.sqrt(kx_mesh**2 + ky_mesh**2).T #for non-cubic array, transpose == non-transpose if cubic.
        # print('3D wavenumbers',np.shape(abs_wavenumbers),abs_wavenumbers[:4,:4])
        # construct spectral energy
        vel_batch_ft  = np.fft.fftn(vel_batch[midbatch:,:,:,:], axes=(2,3)) # Dedalus2D velocity:[t,n,x,y]

        ke_in_kxky = 0.5 * np.sum(np.abs(vel_batch_ft * vel_batch_ft.conj()),
                    axis=1) / (np.array(Nx*Ny, dtype=np.float64)**2)
    k_grid = []
    E_k    = []
    E_k_cut = [] 
    # print('shape vel_batch',np.shape(vel_batch),'shape ke_in_kxky',np.shape(ke_in_kxky.real),' shape abs_wavenumbers',np.shape(abs_wavenumbers))

    for k in np.arange(0,k_max, dk):
        k_grid.append(k)
        kx_ky_indices = (abs_wavenumbers >= k) & (abs_wavenumbers < k + dk)
        # if(k==0 or k==dk):print('kx_ky_indices',np.shape(kx_ky_indices),kx_ky_indices)#,kx_ky_indices.sum(),'location',np.asarray(kx_ky_indices==True).nonzero())
        E_k.append(np.sum(np.mean(ke_in_kxky[:,kx_ky_indices],axis=0))) # average over time sampling
                
        ##locate cut-off limit
        kx_ky_indices_cut = (abs_wavenumbers >= k) & (abs_wavenumbers < k + dk) & (abs_wavenumbers >= kc/eta)
        E_k_cut.append(np.sum(np.mean(ke_in_kxky[:,kx_ky_indices_cut],axis=0))) # average over time sampling
        #E_k_batch_cut.append(ke_in_kxky[:,kx_ky_indices_cut])
    
    u_batch_uncut=np.empty_like(vel_batch,dtype=np.complex128)
    # time loop for filtering based on k
    for ti in range(np.shape(vel_batch_ft)[0]):
        for k in np.arange(0,k_max, dk):
            kx_ky_indices = (abs_wavenumbers >= k) & (abs_wavenumbers < k + dk) & (abs_wavenumbers >= kc/eta)
            # for dim in range(ndims):
            #     # u_batch_uncut[midbatch+ti,dim,:,]+=np.fft.ifftn(vel_batch_ft[ti,dim,:,]*kx_ky_indices) #if using iFFTN, for x axis
            #     u_batch_uncut[midbatch+ti,dim,:,]+=np.fft.ifftn(apply_mask(vel_batch_ft[ti,dim,:,],kx_ky_indices)) #if using iFFTN, for x axis
            u_batch_uncut[midbatch+ti,0,:,:,:]+=np.fft.ifftn(apply_mask(vel_batch_ft[ti,0,:,:,:],kx_ky_indices)) #if using iFFTN, for x axis
            u_batch_uncut[midbatch+ti,1,:,:,:]+=np.fft.ifftn(apply_mask(vel_batch_ft[ti,1,:,:,:],kx_ky_indices)) # y
            u_batch_uncut[midbatch+ti,2,:,:,:]+=np.fft.ifftn(apply_mask(vel_batch_ft[ti,2,:,:,:],kx_ky_indices)) # z axis


    # print('vel batch shape',np.shape(vel_batch),'u_batch recons shape',np.shape(u_batch_uncut))
   
    return k_grid, E_k, E_k_cut,u_batch_uncut.real
    
def apply_mask(array,mask):
    return np.where(mask, array, 0)

def plot_scalars(figs,axs,times,scalars,scalar_names):
    meanvalues = []
    for p in range(len(scalars)):
        #print(np.shape(stores['time']),np.shape(stores[p+1]))
        axs[p]=plt.subplot(figs,1,p+1)
        plt.plot(times, scalars[p], label=scalar_names[p])
        axs[p].legend(loc="best")
        # axs[p].set_ylim(ylims[p])
        
        ## hline properties & mean
        mid = int(len(scalars[p])*0.5)
        avrg=np.nanmean(scalars[p][mid:]);
        meanvalues.append(avrg)
        axs[p].hlines(y=avrg,xmin =min(times),xmax=max(times),linestyles= ':',colors='k')
        axs[p].annotate('mean of last half ={:.2e}'.format(avrg),xy=(np.mean(times), avrg*1.)) 
    plt.xlabel('t')

def plot_spectra(figs,axs,nu,eta,eps,k,Ek,Ekcut,kc,nx,ny,nz,xlims,ylims):
    k_norm = np.float64(k)*eta 
    Ek_norm = np.float64(Ek)*(eps**(-2/3)) * (eta**(-5/3))
    Ekcut_norm = np.float64(Ekcut)*(eps**(-2/3)) * (eta**(-5/3))
    
    x_fivethird = np.linspace(min(k_norm),max(k_norm),100)#,2,100)
    y_fivethird = 2*x_fivethird**(-5/3)

    axs[2]=plt.subplot(figs,1,3)
    if(ndims==2):
        simtype='2D';
    else:
        simtype='3D';

    plt.plot(k_norm,Ek_norm,label=simtype+' '+str(nx)+'x'+str(ny)+'x'+str(nz))
    plt.plot(k_norm,Ekcut_norm,label=simtype+' '+str(nx)+'x'+str(ny)+'x'+str(nz)+r' $k>{:.2f}/\eta$'.format(kc), linestyle="-")
    plt.plot(x_fivethird,y_fivethird, color='k',linestyle=':',label=r'$2 (\kappa \eta)^{-5/3}$')
    plt.xlabel(r'$\kappa \eta$');plt.ylabel(r'$E(\kappa) \epsilon^{-2/3} \eta^{-5/3}$')
    # plt.xlabel(r'$\kappa $');plt.ylabel(r'$E(\kappa)$')

    if(nu==1/30):plt.scatter(LME().res48[:,0],LME().res48[:,1],color='g',marker='o',label='LME 144x48x48')
    if(nu==1/50):plt.scatter(LME().res64[:,0],LME().res64[:,1],color='orange',marker='o',label='LME 192x64x64')
    if(nu==1/70):plt.scatter(LME().res96[:,0],LME().res96[:,1],color='b',marker='o',label='LME 288x96x96')
    if(nu==1/90):plt.scatter(LME().res128[:,0],LME().res128[:,1],color='pink',marker='o',label='LME 384x128x128')
    if(nu==1/110):plt.scatter(LME().res256[:,0],LME().res256[:,1],color='k',marker='o',label='LME 768x256x256')
    # axs[len(tasks)-1].annotate('tke ={:.2f}'.format(TKE_manual_average),xy=(np.mean(k_norm), np.mean(Ek)))#np.trapezoid(Ek,k)
    axs[2].set_yscale('log'); axs[2].set_xscale('log');
    axs[2].set_ylim(ylims[-1]); axs[2].set_xlim(xlims[-1])
    axs[2].grid(True)

def plot_inst_Ek(fig,axs,u_prime_arrays,u_recon_arrays,grids,nz):
    Xi, Yi = np.meshgrid(grids[0], grids[1])
    viewtime= -1 #last timestep
    sliceZ = int(nz/2)
    u_primes = u_prime_arrays[viewtime,0,:] #uprimes[t=-1,axis=0,x,y,z]
    v_primes = u_prime_arrays[viewtime,1,:]
    u_primes_recon = u_recon_arrays[viewtime,0,:]
    v_primes_recon = u_recon_arrays[viewtime,1,:]
    
    if(ndims==3): 
        w_primes = u_prime_arrays[viewtime,2,:]
        Zi = np.multiply(np.power(u_primes,2)+np.power(v_primes,2)+np.power(w_primes,2),0.5)
        Zi = Zi[:,:,sliceZ].T
        # Zi = u_prime_arrays[viewtime,0,:,:,sliceZ].T  

        w_primes_recon= u_recon_arrays[viewtime,2,:]
        Zi_cut = np.multiply(np.power(u_primes_recon,2)+np.power(v_primes_recon,2)+np.power(w_primes_recon,2),0.5)
        Zi_cut = Zi_cut[:,:,sliceZ].T
        # Zi_cut = u_recon_arrays[viewtime,0,:,:,sliceZ].T
    else: 
        Zi = np.multiply(np.power(u_primes,2)+np.power(v_primes,2),0.5)
        Zi = Zi[:,:].T
        Zi_cut = np.multiply(np.power(u_primes_recon,2)+np.power(v_primes_recon,2),0.5)
        Zi_cut = Zi_cut[:,:].T
    
    print('norm2 errors=',LA.norm(Zi-Zi_cut)/LA.norm(Zi)) ##for checking error of total k

    levels=np.linspace(0,np.max(Zi),100)
    CP3=axs[3].contourf(Xi,Yi,Zi,levels=levels,cmap='Greys',vmin=0, vmax=levels[-1])
    axs[3].set(xlabel="x", ylabel="y",title='Instantaneous mid-Z Ek, unfiltered')
    CP4=axs[4].contourf(Xi,Yi,Zi_cut,levels=levels,cmap='Greys',vmin=0, vmax=levels[-1])
    axs[4].set(xlabel="x", ylabel="y",title='Instantaneous mid-Z Ek, filtered')

    plt.legend()
    fig.colorbar(CP4,label=r'$E_k$'),fig.colorbar(CP3,label=r'$E_k$')

class LME():
    def __init__(self):
        self.res48=[[0.032733942469289,198.881889187874],\
            [0.0365286323213472,153.405448531404],\
            [0.0407632224722108,118.956299093967],\
            [0.0454887084657668,92.2431454006449],\
            [0.0507619975161228,71.150777740079],\
            [0.0566465938193442,55.02698975151],\
            [0.0632133633101138,48.9723909251431],\
            [0.0705413870729836,69.2905638406934],\
            [0.0787189136855237,101.743028769707],\
            [0.0878444219620707,148.999452591456],\
            [0.0980278068962911,147.038781369779],\
            [0.109391703084411,99.6093754006386],\
            [0.122072961566588,74.2315521857179],\
            [0.136224297871466,55.4660734339082],\
            [0.152016131110668,48.842822414117],\
            [0.169638636270748,46.5683393033756],\
            [0.189304034417566,37.0802767029383],\
            [0.211249148393128,27.2695840161912],\
            [0.235738254782175,22.0614504702692],\
            [0.263066266493662,16.1386981187482],\
            [0.293562284283726,11.6506499897072],\
            [0.327593559989781,8.19079876871823],\
            [0.365569919203437,5.78899892073437],\
            [0.407948696642804,3.87007171274926],\
            [0.455240243658971,2.60097194564697],\
            [0.508014074201442,1.6360193363573],\
            [0.566905723256047,1.00748073836956],\
            [0.632624400348846,0.599415333044944],\
            [0.705961530284249,0.341829904847563],\
            [0.787800283970168,0.18933728631758],\
            [0.874754123761931,0.101456598330545],\
            [0.966475017314295,0.0545387415456115],\
            [1.06250268062161,0.0289310613346796],\
            [1.16226244425849,0.0153370690379668],\
            [1.26506588291763,0.0080977168274602],\
            [1.37011446244185,0.0043912102831062],\
            [1.4765063734146,0.00256388279348598],\
            [0.0930368948713856,180.019333289153],\
            [0.0617586329858686,44.9243983471437],\
            [0.0617586329858686,46.1639115739473],\
            [0.0930368948713856,170.481982476717],\
            [0.0930368948713856,175.185766611537],\
            [0.0311478234758186,223.811190163224],\
            [1.52765141595144,0.00200724540691555],\
            [0.176077040515489,46.1639115739473]]
        self.res48=np.array(self.res48)
        self.res64=[[0.0243054155729019,279.274139685818],\
            [0.027045864815444,236.434083420268],\
            [0.0305705821335701,194.416594387524],\
            [0.0340174324209363,163.909707618362],\
            [0.0370175237034035,138.959008892092],\
            [0.0435857453909208,110.367995840127],\
            [0.0485000626377879,111.290780526495],\
            [0.0541224436170565,143.465727138339],\
            [0.0603643466932923,183.504665067857],\
            [0.0676987786408987,234.229745726435],\
            [0.0751710642846312,201.734173438839],\
            [0.0838504427857865,149.460691678646],\
            [0.0944300128798294,108.99814536097],\
            [0.105114476141515,84.1240815040414],\
            [0.114983811519152,76.8326415927854],\
            [0.129783923403742,73.0885882101687],\
            [0.144829154743125,57.8896171636187],\
            [0.171582848537892,42.1882452569075],\
            [0.192019902984554,32.590758681563],\
            [0.220472331922038,23.5542469019235],\
            [0.246732561833527,17.8210033418908],\
            [0.268365131749445,14.2776029393493],\
            [0.317939118235463,8.47032279206754],\
            [0.389400693673565,4.37550405037049],\
            [0.44848193277004,2.65316851624242],\
            [0.50422908926823,1.59324614390854],\
            [0.595885967528108,0.746833800893794],\
            [0.668287729773207,0.429296985038964],\
            [0.745759139420997,0.239682099084307],\
            [1.17390963786954,0.015486778129028],\
            [1.29699798693851,0.00730185808962569],\
            [1.42282257648843,0.00377610146195344],\
            [1.50624720041728,0.00269522086254241],\
            [0.0460570649873585,98.9172120959349],\
            [0.0409958732345472,119.67796184265],\
            [0.818546730706904,0.147983319823752],\
            [0.890110415107857,0.0882320718841774],\
            [0.981546821945467,0.0484818514143269],\
            [1.06736123467887,0.0281301842545708]]
        self.res64=np.array(self.res64)
        self.res96=[[0.0147421642095223,615.449124165583],\
            [0.0168944572382294,463.453324354096],\
            [0.0193931781654123,344.616073817529],\
            [0.0222984910267542,255.629154028844],\
            [0.0253593695359891,195.227659291503],\
            [0.0293334549226068,191.471411702614],\
            [0.0329796591121559,285.148310400786],\
            [0.0375628650012487,444.062095680105],\
            [0.0428470401090765,546.668398798294],\
            [0.04817300861731,362.64669692099],\
            [0.0555281077258228,250.832559672214],\
            [0.063740815767571,185.972113792051],\
            [0.0729739022420726,162.957162246849],\
            [0.0795077668684482,154.629535648386],\
            [0.0932603346883221,112.221280597843],\
            [0.106592139402992,94.2190126035658],\
            [0.113842975230142,83.852130357107],\
            [0.13786408227773,60.1498328230221],\
            [0.180997586762538,35.4943149334197],\
            [0.197991686451503,29.3693321381668],\
            [0.242896610327383,18.4245241255621],\
            [0.291617298773424,10.9040105880416],\
            [0.337541165561451,6.84050986245203],\
            [0.367397066593226,5.26239527100056],\
            [0.437440302888638,2.85367897151262],\
            [0.508014074201442,1.50304039331078],\
            [1.02606154801213,0.0405147133689255],\
            [1.17390963786954,0.0159447306944409],\
            [1.28412954614912,0.00905917432143746],\
            [0.577264866662673,0.868029088317069],\
            [0.642520153286506,0.531824476797772],\
            [0.745759139420997,0.248198443926444],\
            [0.837827674817883,0.132718493119005],\
            [0.936889937080075,0.0709682064767474],\
            [1.34091551432597,0.00761722335475657],\
            [0.0139830405729589,683.148889076289],\
            [0.0281150151285317,161.449916618167],\
            [0.0409958732345472,596.229845291084],\
            [0.0348306498502621,345.945027197457],\
            [0.0154194455008812,549.481048999609],\
            [0.01615441379883,506.397701480889],\
            [0.0177311170625212,418.551987943596],\
            [0.0186629713203075,375.377368192559],\
            [0.0204845148413133,310.259985830445],\
            [0.0213612117305872,285.933325584341],\
            [0.0234461089401941,229.98638539612],\
            [0.026835882064767,175.185766611537],\
            [0.0305730471432081,217.801800554426],\
            [0.0394968248436983,520.369767264791],\
            [0.0445800494627816,466.69240464617],\
            [0.0462720247462532,407.313756444733],\
            [0.0243359741893009,211.953764644895],\
            [0.0318815133448103,256.438632065129],\
            [0.0361525999011049,385.734439851527],\
            [0.0505523271235102,310.259985830445],\
            [0.0529619051995306,278.255940220711]]
        self.res96=np.array(self.res96)
        self.res128=[[0.0110870447181335,1470.1876828433],\
            [0.012497183450999,1233.8323802684],\
            [0.0156762980382978,837.848125408925],\
            [0.018759907844825,671.604166289586],\
            [0.0216329577937828,594.43393604369],\
            [0.0244505427078293,706.091504643137],\
            [0.027556450652855,841.351750600692],\
            [0.0305747256442368,949.586023948767],\
            [0.0347942464468452,694.081418697246],\
            [0.0385756397014329,500.085375267124],\
            [0.0432671997268339,386.352369283175],\
            [0.0476389199259274,298.900819573986],\
            [0.0538588291802301,261.577639050202],\
            [0.0607068196460697,240.149683558714],\
            [0.0673689394493918,177.507249500411],\
            [0.0767950024498798,144.681624180577],\
            [0.0858464833065777,120.099021916796],\
            [0.094224094704356,106.077563028656],\
            [0.10727375577218,86.6718347100959],\
            [0.118728718394303,70.6439982133697],\
            [0.136160374411137,61.8820850681165],\
            [0.167237218952483,40.4559257649743],\
            [0.185156434837829,33.1193908182408],\
            [0.210808537350979,24.4425702219571],\
            [0.237630954263452,18.4654665615577],\
            [0.271367305348985,13.1711702572449],\
            [0.301359358343158,9.5052355402639],\
            [0.345415024910811,6.25358048945741],\
            [0.387772718052256,4.08435120517366],\
            [0.446329593389147,2.43189464047827],\
            [0.497744960519114,1.56910936563398],\
            [0.594307389782284,0.713063884655367],\
            [0.666666981491521,0.409347826590895],\
            [0.811661700118194,0.168436018026483],\
            [0.892206825050333,0.0978303156831689],\
            [0.989283671862478,0.0531465239195143],\
            [1.0864857811632,0.0292136589143187],\
            [1.19742101937837,0.0157000325829627],\
            [1.31536568637499,0.00923161699021027],\
            [0.0105628944313819,1590.09034642146],\
            [0.0119231238078282,1349.99373659237],\
            [0.0134586642786633,1115.30427001651],\
            [0.0143658905663409,999.999999999999],\
            [0.0169893393503471,782.284866716217],\
            [0.0205653660472415,595.499658257023],\
            [0.0231035217075393,646.288666805378],\
            [0.0260749210893821,782.284866716217],\
            [0.0294288039965234,921.414359933919],\
            [0.0321510574616984,921.414359933919],\
            [0.0335292791160786,782.284866716217],\
            [0.036634881050617,595.499658257023],\
            [0.0407815678522943,429.241221225663],\
            [0.0456094983402436,309.400724995506],\
            [0.0500643914553423,262.682583905431],\
            [0.0565076574189307,262.682583905431],\
            [0.0649864127840108,199.962182065963],\
            [0.0710049036396208,156.427388945756],\
            [1.2609008868002,0.0117785016576087],\
            [1.13784040031578,0.0220598165103525],\
            [1.04128029930043,0.0370442135515572],\
            [0.944029365802005,0.0712985288275355],\
            [0.55979532835031,1.00547127583834],\
            [0.603180761936435,0.705246975333953],\
            [0.626115119587635,0.59875780114899]]
        self.res128=np.array(self.res128)
        self.res256=[[0.00551468178290866,3716.69173344044],\
            [0.00603260499437688,3505.77739379005],\
            [0.00683370465405649,3119.176321302],\
            [0.00721911591912234,2942.16970863587],\
            [0.0075882650058893,2840.82444633246],\
            [0.00842610218031647,2617.72071001568],\
            [0.00890132241081944,2469.17063523425],\
            [0.00940323329617095,2398.08736408451],\
            [0.0104414898178972,2196.88180781798],\
            [0.011651921940136,2196.88180781798],\
            [0.0130026736861163,2196.88180781798],\
            [0.0145097545047349,2295.28092043554],\
            [0.015251403669074,2329.05046079824],\
            [0.016111307444485,2284.13356721413],\
            [0.0179502913984031,1954.61974504801],\
            [0.0188061915710842,1739.07323285918],\
            [0.019611525539912,1529.32229059642],\
            [0.0203702271381132,1298.54451852913],\
            [0.0215351380657338,1126.77870339495],\
            [0.0227343361723592,969.607164748181],\
            [0.0238983233065355,804.275350265534],\
            [0.0251204814375635,767.550562165589],\
            [0.0258832822725956,767.550562165589],\
            [0.026404641498801,767.550562165589],\
            [0.0269365023082247,767.550562165589],\
            [0.0278931668032069,767.550562165589],\
            [0.0294656056718988,767.550562165589],\
            [0.0309726204960117,723.993702577027],\
            [0.0325041871214208,607.600569488073],\
            [0.0338854839194682,509.919424337747],\
            [0.0355608087858385,436.357395863547],\
            [0.0374420288038488,403.657228452143],\
            [0.0395533946741578,388.237901018336],\
            [0.0439205035469344,359.143758283279],\
            [0.0482889497374545,284.301729185536],\
            [0.0505050546065029,284.301729185536],\
            [0.0520399052195819,268.168211591619],\
            [0.0541600272206032,238.595849583647],\
            [0.0593969137958495,203.183912927837],\
            [0.0651394002003076,178.156576023692],\
            [0.0681304256540367,168.046569791987],\
            [0.0730591927793788,149.515163826337],\
            [0.0752785751781358,145.210873640958],\
            [0.0807235080002712,133.027316425964],\
            [0.0909905672733142,111.641127443766],\
            [0.102564684715145,90.9958308146596],\
            [0.109983233975355,83.361051935789],\
            [0.121523957662844,69.9594795493703],\
            [0.124595815171172,64.0897034062388],\
            [0.135621263835794,58.7124162323279],\
            [0.139742799423999,55.3806116633606],\
            [0.145432528337289,52.2378799072352],\
            [0.152874269498497,46.4773257909323],\
            [0.169761880134974,37.8824808672749],\
            [0.184788427429925,32.7346648564926],\
            [0.192316765709467,29.1248359599193],\
            [0.198161279848781,27.4720635525004],\
            [0.225614044627167,20.5130507817274],\
            [0.232472295383199,18.9758241848144],\
            [0.241941383759724,17.2152620838429],\
            [0.255595594063582,14.8758957129995],\
            [0.276842642777628,12.1249625519811],\
            [0.310523795501659,8.53978964542886],\
            [0.336340867466397,6.76018142339856],\
            [0.343123788561996,6.37655552620502],\
            [0.375384475361092,4.76129530901739],\
            [0.481839145948038,1.86969097203247],\
            [0.624710549534355,0.65323603279542],\
            [0.659997919444946,0.502221392651987],\
            [0.722817733294936,0.320279561995972],\
            [0.755304082960739,0.256516228012742],\
            [0.78610017284852,0.211125566021013],\
            [0.818158358776795,0.170415623112173],\
            [0.851520565783224,0.138901420685679],\
            [0.886243190176337,0.113214999400614],\
            [0.922392603316927,0.0896221063486951],\
            [0.960016532803476,0.0709457403073858],\
            [0.994194257354832,0.0578260604299559],\
            [1.07695374639663,0.0362364765091417],\
            [1.13214141349849,0.0270573297620098],\
            [1.17244698747724,0.022053738238719],\
            [1.21418748758305,0.017975438617924],\
            [1.31851924691917,0.0117687681394902],\
            [1.37563675273267,0.0109399603989049],\
            [0.411476917728869,3.43198217617876],\
            [0.445416977102834,2.68478771926444],\
            [0.517116997843407,1.43350134335177],\
            [0.546894644110582,1.09122596690424],\
            [0.583798555943705,0.830675266799452],\
            [0.690554568342834,0.3869703810439],\
            [0.0169820299020518,2266.9933019671],\
            [0.00629773516070565,3321.44460910282],\
            [0.00802361108159466,2819.92116285624],\
            [0.00994103176956586,2266.9933019671]]
        self.res256=np.array(self.res256)

if __name__ == "__main__":
    args = docopt(__doc__ )
    ndims = int(args['--ndims'])

    # print ('rank no', Sync().comm.rank)
    main(ndims,args)