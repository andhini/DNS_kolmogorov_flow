"""hdf2vtk: convert Dedalus fields stored in HDF5 files to vtk format for 3D visualization

Usage:
    hdf2vtk [--fields=<fields>] <path> [<output_file>]

Options:
    --fields=<fields>        comma separated list of fields to extract from the hdf5 file [default: None]
   

"""
# command example: python3 hdf2vtk.py --fields=velocity,pressure ./snapshots/ ./frames3D
# currently only for 3D
from dedalus.extras import plot_tools
from pathlib import Path
from docopt import docopt
from pyevtk.hl import gridToVTK,VtkFile,VtkRectilinearGrid
import h5py,math,os,re
import numpy as np
from scipy.interpolate import interpn
from sys import exit
H5_FIELD_PATH = 'tasks/'
H5_SCALE_PATH = 'scales/'
H5_DIM_LABEL = 'DIMENSION_LABELS'
H5_STR_DECODE = 'UTF-8'

def main(fields,args):
    if fields is None:
        raise ValueError("Must specify fields to copy.")
    fields = fields.split(',')
    print("fields = {}".format(fields))

    #get all file names ended with *.h5
    files={}
    for file in os.listdir(args['<path>']):
        if file.endswith(".h5"):
            sf = os.path.join(args['<path>'], file)
            m = re.search('snapshots_s(.*).h5', sf)
            num= m.group(1)
            # if(int(num)>2):continue
            files[int(num)]=sf

    myKeys = list(files.keys())
    myKeys.sort()

    # Sorted Dictionary
    infiles = {i: files[i] for i in myKeys}
    print('opening ',len(infiles),' h5 files in ', Path(args['<path>']))
    
    if args['<output_file>']:
        outfile = os.path.join(args['<path>'],Path(args['<output_file>']))
    else:
        outfile = os.path.join(args['<path>'],Path(args['<path>'])) #infile.stem
    print("outfile = {}".format(outfile))
      
    field_names = [H5_FIELD_PATH+f for f in fields]
    # print(field_names)
    
    # ## how many timesteps (to know digits for vtk names)
    # times = []
    # for f in range(len(infiles)):
    #     # find sime time numbers
    #     datafile=h5py.File(infiles[f+1],"r")
    #     times.append(list(datafile[H5_SCALE_PATH+'sim_time']))
    # times=flatten(times)
    # digits=math.floor(math.log(len(times), 10)) #789 returns 2
    
    nt=0
    for fs in range(len(infiles)):
        # if(fs!=len(infiles)-1):continue
        print('processing',infiles[fs+1])
        datafile=h5py.File(infiles[fs+1],"r")
        dim_labels = datafile[field_names[0]].attrs[H5_DIM_LABEL][1:]
    
        scale_list = list(datafile['scales'].keys())
        # currently cartesian only
        scale_names = [H5_SCALE_PATH+scale_list[d-3] for d in range(3)] #decode(H5_STR_DECODE)
        # print(scale_names)
        
        # filename of x, y, z from scale_names list
        x = plot_tools.get_1d_vertices(datafile[scale_names[0]][:])
        y = plot_tools.get_1d_vertices(datafile[scale_names[1]][:])
        z = plot_tools.get_1d_vertices(datafile[scale_names[2]][:])
        nx, ny, nz = len(x)-1, len(y)-1, len(z)-1
        ncells = nx * ny * nz
        npoints = (nx+1) * (ny+1) * (nz+1)
        start, end = (0,0,0), (nx, ny, nz)
        
        xc = np.array(datafile[scale_names[0]][:])
        yc = np.array(datafile[scale_names[1]][:])
        zc = np.array(datafile[scale_names[2]][:])
       
        XX,YY,ZZ= np.meshgrid(*[x,y,z],indexing='ij')
        
        sim_times=list(datafile[H5_SCALE_PATH+'sim_time'])
        #
        for ts in range(len(sim_times)): 
            # if(ts!=len(sim_times)-1):continue 
            # cellData = {}
            # for i, f in enumerate(fields):
            #     if(i==0):
            #         cellData[f] = datafile[field_names[i]][ts,-1,:,:,:] #only for vorticity-Z if the second digit -1
            #     else: 
            #         cellData[f] = datafile[field_names[i]][ts,:,:,:]
                                    
            w = VtkFile(outfile+'_'+str(nt).zfill(5), VtkRectilinearGrid)
            w.openGrid(start = start, end = end)
            w.openPiece( start = start, end = end)

            # # Point data, if velocities (cell data) are filled to vertex at the last grids, 
            # vx = np.zeros([nx + 1, ny + 1, nz + 1], dtype="float64", order = 'F') 
            # vx[:-1,:-1,:-1] = datafile[H5_FIELD_PATH+'velocity'][ts,0,:,:,:]; 
            # vy = np.zeros([nx + 1, ny + 1, nz + 1], dtype="float64", order = 'F') 
            # vy[:-1,:-1,:-1] = datafile[H5_FIELD_PATH+'velocity'][ts,1,:,:,:]
            # vz = np.zeros([nx + 1, ny + 1, nz + 1], dtype="float64", order = 'F') 
            # vz[:-1,:-1,:-1] = datafile[H5_FIELD_PATH+'velocity'][ts,-1,:,:,:]
            # # last point data are same with previous ones (no data available)
            # vx[-1,:,:]=vx[-2,:,:]; vx[:,-1,:]=vx[:,-2,:]; vx[:,:,-1]=vx[:,:,-2];
            # vy[-1,:,:]=vy[-2,:,:]; vy[:,-1,:]=vy[:,-2,:]; vy[:,:,-1]=vy[:,:,-2];
            # vz[-1,:,:]=vz[-2,:,:]; vz[:,-1,:]=vz[:,-2,:]; vz[:,:,-1]=vz[:,:,-2];

            ## if interpolated &extrapolated for point-data velocities
            vx_in = np.zeros([nx, ny, nz], dtype="float64", order = 'F') 
            vx_in[:,:,:] = datafile[H5_FIELD_PATH+'velocity'][ts,0,:,:,:]        
            vx = interpn((xc,yc,zc), vx_in, (XX,YY,ZZ),method='linear',bounds_error=False,fill_value=None)
            vy_in = np.zeros([nx, ny, nz], dtype="float64", order = 'F') 
            vy_in[:,:,:] = datafile[H5_FIELD_PATH+'velocity'][ts,1,:,:,:]
            vy = interpn((xc,yc,zc), vy_in, (XX,YY,ZZ),method='linear',bounds_error=False,fill_value=None)
            vz_in = np.zeros([nx, ny, nz], dtype="float64", order = 'F') 
            vz_in[:,:,:] = datafile[H5_FIELD_PATH+'velocity'][ts,-1,:,:,:]
            vz = interpn((xc,yc,zc), vz_in, (XX,YY,ZZ),method='linear',bounds_error=False,fill_value=None)
            
            w.openData("Point", vectors = "velocity")
            w.addData("velocity", (vx,vy,vz))
            w.closeData("Point")
         
            # print('velo_X', np.shape(vx[:-1,:-1,:-1]),vx[:2,:2,:2],'/n should be:',np.shape(datafile[H5_FIELD_PATH+'velocity'][ts,0,:,:,:]),datafile[H5_FIELD_PATH+'velocity'][ts,0,:2,:2,:2])
            # print('velo_Y',vy[:2,:2,:2],'/n should be:',datafile[H5_FIELD_PATH+'velocity'][ts,1,:2,:2,:2])
            # print('velo_Z',vz[:2,:2,:2],'/n should be:',datafile[H5_FIELD_PATH+'velocity'][ts,-1,:2,:2,:2])

            # Cell data
            pressure = datafile[H5_FIELD_PATH+'pressure'][ts,:,:,:]
            w.openData("Cell", scalars = "pressure")
            w.addData("pressure", pressure)
            w.closeData("Cell")

            # Coordinates of cell vertices
            w.openElement("Coordinates")
            w.addData("x_coordinates", x);
            w.addData("y_coordinates", y);
            w.addData("z_coordinates", z);
            w.closeElement("Coordinates");

            w.closePiece()
            w.closeGrid()

            w.appendData(data = (vx,vy,vz))
            w.appendData(data = pressure)
            w.appendData(x).appendData(y).appendData(z)
            w.save()
            nt+=1
        # gridToVTK(outfile+'_'+str(ts+nt).zfill(5), x, y, z, cellData = cellData)
   
    # for key, value in cellData.items():
    #     if(key=='vorticity'):print('andhini check',key,value[-1,-1,-1])# np.asarray(value).shape)
def flatten(xss):
    arrays = [x for xs in xss for x in xs]
    str_array = map(float,arrays)
    return list(str_array)

if __name__ == "__main__":
    args = docopt(__doc__ )
    
    fields = args['--fields']

    main(fields,args)
 