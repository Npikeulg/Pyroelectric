#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Nicholas Pike
Email : Nicholas.pike@smn.uio.no

Purpose: Calculate the pyroelectric coefficient from first-principles data
         This is done by feeding this program with a set of files and individual
         parameters after the DFT, DFPT, and TDEP calculations are completed.
         
Notes: - Oct 4th start of program 
       - Jan 15th changed read in and updated user interface
       - Feb 28th added in calculation of the anharmonic coupling term 
       - Apr 5th parallelization of anharmonic coupling term
       
"""
#import useful modules
import os
import sys
import h5py
import linecache
import platform
import subprocess
import numpy as np
from multiprocessing import cpu_count
import scipy.constants as sc
from mendeleev import element
from functools import partial
from multiprocessing import Pool

#User defined and known parameters parameters 
hbar         = sc.hbar  # hbar in Js 
ang_to_m     = sc.physical_constants['Angstrom star'][0] #angstrom to meter
Thz_to_hz    = sc.tera    #THz to Hz
echarge      = sc.physical_constants['elementary charge'][0] # electron charge
kb           = sc.k     # boltzmanns constant
amu_kg       = sc.atomic_mass
J_to_cal     = 1.0/sc.calorie
h            = sc.h
Na           = sc.N_A
ev_to_J      = echarge
cm_to_Thz    = 0.02998 
amu_to_kg    = sc.physical_constants['atomic mass constant'][0]
kgm3_to_cm3  = 1000.0
kbar_to_GPa  = 0.1
m_to_cm      = 100.0
difftol      = 1E-5


"""
##############################################################################
The following information should be modified by the user
##############################################################################
"""
batch_header_rlx =  '# Specify jobname:\n'\
                    '#SBATCH --job-name=rlx\n'\
                    '# Specify the number of nodes and the number of CPU (tasks) per node:\n'\
                    '#SBATCH --nodes=1  --ntasks-per-node=16\n'\
                    '#SBATCH --account=nn2615k\n'\
                    '# The maximum time allowed for the job, in hh:mm:ss\n'\
                    '#SBATCH --time=3:00:00\n'\
                    '#SBATCH --mem-per-cpu=1800M\n'\
                    '#SBATCH --mail-user=Nicholas.pike@smn.uio.no\n'\
                    '#SBATCH --mail-type=ALL'
                    
batch_header_ela =  '# Specify jobname:\n'\
                    '#SBATCH --job-name=elas\n'\
                    '# Specify the number of nodes and the number of CPU (tasks) per node:\n'\
                    '#SBATCH --nodes=2  --ntasks-per-node=16\n'\
                    '#SBATCH --account=nn2615k\n'\
                    '# The maximum time allowed for the job, in hh:mm:ss\n'\
                    '#SBATCH --time=15:00:00\n'\
                    '#SBATCH --mem-per-cpu=1800M\n'\
                    '#SBATCH --mail-user=Nicholas.pike@smn.uio.no\n'\
                    '#SBATCH --mail-type=ALL'                    
                    
batch_header_tdep = '# Specify jobname:\n'\
                    '#SBATCH --job-name=tdep\n'\
                    '# Specify the number of nodes and the number of CPU (tasks) per node:\n'\
                    '#SBATCH --nodes=1  --ntasks-per-node=16\n'\
                    '#SBATCH --account=nn2615k\n'\
                    '# The maximum time allowed for the job, in hh:mm:ss\n'\
                    '#SBATCH --time=00:05:00\n'\
                    '#SBATCH --mem-per-cpu=2000M\n'\
                    '#SBATCH --mail-user=Nicholas.pike@smn.uio.no\n'\
                    '#SBATCH --mail-type=ALL'  
                    
batch_header_gs  =  '# Specify jobname:\n'\
                    '#SBATCH --job-name=tdepconfig\n'\
                    '# Specify the number of nodes and the number of CPU (tasks) per node:\n'\
                    '#SBATCH --nodes=4  --ntasks-per-node=16\n'\
                    '#SBATCH --account=nn2615k\n'\
                    '# The maximum time allowed for the job, in hh:mm:ss\n'\
                    '#SBATCH --time=12:00:00\n'\
                    '#SBATCH --mem-per-cpu=1800M\n'\
                    '#SBATCH --mail-user=Nicholas.pike@smn.uio.no\n'\
                    '#SBATCH --mail-type=ALL'  
                    
batch_header_pyro=  '# Specify jobname:\n'\
                    '#SBATCH --job-name=multi_python\n'\
                    '# Specify the number of nodes and the number of CPU (tasks) per node:\n'\
                    '#SBATCH --nodes=4  --ntasks-per-node=16\n'\
                    '#SBATCH --account=nn2615k\n'\
                    '# The maximum time allowed for the job, in hh:mm:ss\n'\
                    '#SBATCH --time=24:00:00\n'\
                    '#SBATCH --mem-per-cpu=1800M\n'\
                    '#SBATCH --mail-user=Nicholas.pike@smn.uio.no\n'\
                    '#SBATCH --mail-type=ALL'                      

"""
##############################################################################
VASP parameters. Make sure you change convergence parameters before doing your calculation
##############################################################################
"""
ecut     = '600'  #value in eV
ediff    = '1E-6' #value in Hartree    
kdensity = 7        

"""
##############################################################################
TDEP parameters. These parameters may need to be converged.
##############################################################################
"""
natom_ss  = '200'       #number of atoms in the supercell (metals ~100, semiconductor ~200])
rc_cut    = '100'       #second order cut-off radius (100 defaults to the maximum radius)
rc3_cut   = '4'         #third order force constant cut off radius
tmin      = '0'         #minimum temperature
tmax      = '3000'      #maximum temperature
tsteps    = '1500'      #number of temperature steps
qgrid     = '30 30 30'  #q point grid density for DOS
iter_type = '3'         #method for numerical integration
n_configs = '12'        #number of configurations
t_configs = '300'       #temperature of configurations

"""
##############################################################################
Paths to executables and main file name
##############################################################################
"""

##############################################
# Stallo
##############################################
VASPSR   = 'source /home/espenfl/vasp/bin/.jobfile_local'  #source path to VASP executable
TDEPSR   = '~nicholasp/bin/TDEP/bin/'  #source path to TDEP bin of executables
PYTHMOD  = 'module load StdEnv\nmodule load intel/2016b\nmodule load HDF5/1.8.17-intel-2016b\nmodule load Python/2.7.12-intel-2016b\n' #module for python
__root__ = os.getcwd()                    


"""
Build definitions and other useful functions
"""
def CHECK_FILE(filename): 
    """
    Author: Nicholas Pike
    Email : Nicholas.pike@ulg.ac.be
        
    Purpose: Checks if the file "filename" exists in the current directory and outputs
    a boolean value indicating whether or not the file is found.
    
    Return: logic value (True or false)
    """
    if os.path.isfile(filename):
        logic  = True
    else:
        logic  = False
        
    return logic


def READ_INPUT_DFT(DFT_INPUT):
    """
    Author: Nicholas Pike
    Email : Nicholas.pike@sintef.no
    
    Purpose: To read an input file which is formatted and contains the results of 
             our DFT, DFPT, and TDEP calculations
    
    Input: DFT_INPUT  - name of formatted input file
    
    OUTPUT: cell_data  - array of data about the unit cell
            
    """
    #store data in arrays
    cell_data = [0] *30
    #open file and look for data 
    print('   2 - Reading input data from DFT and DFPT calculations in %s' %DFT_INPUT)
    with open(DFT_INPUT,'r') as f:
        for num,line in enumerate(f,1):
            if line.startswith( 'volume'):
                l = line.strip('\n').split(' ')
                cell_data[0] = float(l[1])*ang_to_m**3.0 #cell volume converted to m^3
            elif line.startswith('alat'):
                l = line.strip('\n').split(' ')
                cell_data[1] = float(l[1])*ang_to_m
            elif line.startswith('blat'):
                l = line.strip('\n').split(' ')
                cell_data[2] = float(l[1])*ang_to_m
            elif line.startswith('clat'):
                l = line.strip('\n').split(' ')
                cell_data[3] = float(l[1])*ang_to_m
            elif line.startswith('alpha'):
                l = line.strip('\n').split(' ')
                cell_data[27] = float(l[1])
            elif line.startswith('beta'):
                l = line.strip('\n').split()
                cell_data[28] = float(l[1])
            elif line.startswith('gamma'):
                l = line.strip('\n').split(' ')
                cell_data[29] = float(l[1])
            elif line.startswith('atpos'):
                l = line.strip('\n').split(' ')
                cell_data[5] = []
                for i in range(1,len(l)):
                    if l[i] == '':
                        pass
                    else:
                        elname = element(l[i])
                        cell_data[5] = np.append(cell_data[5],[l[i],float(elname.mass*amu_kg)])
            elif line.startswith('natom'):
                l = line.strip('\n').split(' ')
                cell_data[4] = int(l[1])    
            
            elif line.startswith('bectensor'):
                if cell_data[4] != 0:
                    cell_data[6] = np.zeros(shape=(cell_data[4],9))
                    for i in range(cell_data[4]):
                        if i == 0:
                            l= linecache.getline(DFT_INPUT,num+i).strip('\n').split(' ')
                            cell_data[6][i] = [float(l[2]),float(l[3]),float(l[4]),float(l[5]),float(l[6]),float(l[7]),float(l[8]),float(l[9]),float(l[10])]
                        else:
                            l= linecache.getline(DFT_INPUT,num+i).strip('\n').split(' ')
                            cell_data[6][i] = [float(l[11]),float(l[12]),float(l[13]),float(l[14]),float(l[15]),float(l[16]),float(l[17]),float(l[18]),float(l[19])]
                else:
                    print('Data read-in order incorrect for BEC tensor')
                    sys.exit()
                    
            elif line.startswith('dielectric'):
                l = line.strip('\n').split(' ')
                cell_data[8] = np.array([[float(l[1]),float(l[2]),float(l[3])],
                                         [float(l[4]),float(l[5]),float(l[6])],
                                         [float(l[7]),float(l[8]),float(l[9])]])


            elif line.startswith('piezo'):
                l = line.strip('\n').split(' ')
                cell_data[9] = np.array([[float(l[1]),float(l[2]),float(l[3]),float(l[4]),float(l[5]),float(l[6])],
                                         [float(l[7]),float(l[8]),float(l[9]),float(l[10]),float(l[11]),float(l[12])],
                                         [float(l[13]),float(l[14]),float(l[15]),float(l[16]),float(l[17]),float(l[18])]])        
                
            elif line.startswith('TMIN'):
                l = line.strip('\n').split(' ')
                cell_data[23] = l[1]
            elif line.startswith('TMAX'):
                l = line.strip('\n').split(' ')
                cell_data[24] = l[1]  
            elif line.startswith('TSTEP'):
                l = line.strip('\n').split(' ')
                cell_data[25] = l[1]  

    #set file names
    cell_data[11] = 'outfile.grid_dispersions.hdf5'     # tdep file for phonon dispersion data on the full grid
    cell_data[12] = 'outfile.dispersion_relations.hdf5' # tdep file for phonon dispersion data on a path
    cell_data[20] = 'outfile.forceconstant_thirdorder'  # tdep file giving third order force constants
                
    print('      a - Reading TDEP generated data files.')
    if CHECK_FILE(cell_data[11]):
        #read hdf5 file with phonon information inside
        f = h5py.File(cell_data[11], 'r')
        #extract data and save to specific file names
        omega      = f['frequencies'][:]  # given in hdf5 file in hertz
        eigenvecre = f['eigenvectors_re'][:]
        eigenvecim = f['eigenvectors_im'][:]
        qpoints    = f['qpoints'][:]
        #group_vecs = f['group_velocities'][:]
        #gruneisen = f['gruneisen_parameters'][:]
        f.close()
    else: 
        print(' ERROR: %s not found in the correct directory.\n Calculation aborted.' %cell_data[11])
        sys.exit()


    if CHECK_FILE(cell_data[12]):
        print('      b - Extracting q = 0 phonon frequencies in hz.')
        g = h5py.File(cell_data[12],'r')
        #open dispersion relation file and extract q= 0
        omegazero = g['frequencies'][:][0]  # bandstructure starts at q = 0 (gamma point) in cm-1!!!
        omegazero = list(map(lambda x: x*cm_to_Thz*Thz_to_hz, omegazero)) # converts omegazero to Hz
        eigenvecrezero = g['eigenvectors_re'][:][0]
        eigenvecimzero = g['eigenvectors_im'][:][0]
        g.close()
        
        print('          Zero momentum frequencies')
        for i in range(len(omegazero)):
            omega_convert = omegazero[i]
            print('          %1.3e' %omega_convert)
        print('')
        
        #check that imaginary frequencies are zero.
        if np.all(eigenvecimzero != np.zeros(shape=(eigenvecimzero.shape))):
            print(' ERROR: Imaginary frequencies found for q = 0 phonons.\n Calculation aborted.')
            sys.exit()
            
    else:
        print(' ERROR: %s not found in the correct directory.\n Calculation aborted.' %cell_data[12])
        sys.exit()

        
    if CHECK_FILE(cell_data[20]):
        print('      c - Extracting third order coupling constants.')
        
        #sanity check first
        numcoupling = np.empty(shape=(cell_data[4]))
        i = 0
        with open(cell_data[20],'r') as f:
            for num,line in enumerate(f,1):
                if 'How many triplets are atom' in line:
                    l = line.strip('\n').split()
                    numcoupling[i] = int(l[0])
                    i += 1
            for i in range(cell_data[4]):
                if numcoupling[0] != numcoupling[i]:
                    print(' ERROR:There are an unequivalent number of coupling constants. \n Something is wrong caculation aborted.')
                    sys.exit()
                    
        coupling_terms = np.zeros(shape=(int(cell_data[4]),int(numcoupling[0]),39))
        with open(cell_data[20],'r') as f:
            for num,line in enumerate(f,1):
                if 'How many triplets are atom' in line:
                    l = line.strip('\n').split()
                    numcoupling = int(l[0])
                    k = int(l[6])-1
                    for i in range(numcoupling):
                        for j in range(15):
                            l = linecache.getline(cell_data[20],num+j+1+i*15).strip('\n').split()
                            if j == 0:
                                coupling_terms[k][i][0] = int(l[0]) #first atom type
                            elif j == 1:
                                coupling_terms[k][i][1] = int(l[0]) #second atom type
                            elif j == 2:
                                coupling_terms[k][i][2] = int(l[0]) #third atom type  
                            elif j == 3:
                                coupling_terms[k][i][3] = float(l[0]) #position of first atom
                                coupling_terms[k][i][4] = float(l[1])
                                coupling_terms[k][i][5] = float(l[2])
                            elif j == 4:
                                coupling_terms[k][i][6] = float(l[0]) #position of second atom
                                coupling_terms[k][i][7] = float(l[1])
                                coupling_terms[k][i][8] = float(l[2])
                            elif j == 5:
                                coupling_terms[k][i][9]  = float(l[0]) #position of third atom
                                coupling_terms[k][i][10] = float(l[1])
                                coupling_terms[k][i][11] = float(l[2])                            
                            elif j == 6:
                                coupling_terms[k][i][12] = float(l[0])*ev_to_J/ang_to_m  #first three coupling constants
                                coupling_terms[k][i][13] = float(l[1])*ev_to_J/ang_to_m
                                coupling_terms[k][i][14] = float(l[2])*ev_to_J/ang_to_m                 
                            elif j == 7:
                                coupling_terms[k][i][15] = float(l[0])*ev_to_J/ang_to_m #next three coupling constants
                                coupling_terms[k][i][16] = float(l[1])*ev_to_J/ang_to_m
                                coupling_terms[k][i][17] = float(l[2])*ev_to_J/ang_to_m      
                            elif j == 8:
                                coupling_terms[k][i][18] = float(l[0])*ev_to_J/ang_to_m #next three coupling constants
                                coupling_terms[k][i][19] = float(l[1])*ev_to_J/ang_to_m
                                coupling_terms[k][i][20] = float(l[2])*ev_to_J/ang_to_m                                  
                            elif j == 9:
                                coupling_terms[k][i][21] = float(l[0])*ev_to_J/ang_to_m #next three coupling constants
                                coupling_terms[k][i][22] = float(l[1])*ev_to_J/ang_to_m
                                coupling_terms[k][i][23] = float(l[2])*ev_to_J/ang_to_m
                            elif j == 10:
                                coupling_terms[k][i][24] = float(l[0])*ev_to_J/ang_to_m #next three coupling constants
                                coupling_terms[k][i][25] = float(l[1])*ev_to_J/ang_to_m
                                coupling_terms[k][i][26] = float(l[2])*ev_to_J/ang_to_m  
                            elif j == 11:
                                coupling_terms[k][i][27] = float(l[0])*ev_to_J/ang_to_m #next three coupling constants
                                coupling_terms[k][i][28] = float(l[1])*ev_to_J/ang_to_m
                                coupling_terms[k][i][29] = float(l[2])*ev_to_J/ang_to_m  
                            elif j == 12:
                                coupling_terms[k][i][30] = float(l[0])*ev_to_J/ang_to_m #next three coupling constants
                                coupling_terms[k][i][31] = float(l[1])*ev_to_J/ang_to_m
                                coupling_terms[k][i][32] = float(l[2])*ev_to_J/ang_to_m  
                            elif j == 13:
                                coupling_terms[k][i][33] = float(l[0])*ev_to_J/ang_to_m #next three coupling constants
                                coupling_terms[k][i][34] = float(l[1])*ev_to_J/ang_to_m
                                coupling_terms[k][i][35] = float(l[2])*ev_to_J/ang_to_m  
                            elif j == 14:
                                coupling_terms[k][i][36] = float(l[0])*ev_to_J/ang_to_m #last three coupling constants
                                coupling_terms[k][i][37] = float(l[1])*ev_to_J/ang_to_m
                                coupling_terms[k][i][38] = float(l[2])*ev_to_J/ang_to_m 
        
        cell_data[21] = coupling_terms
        cell_data[26] = numcoupling
        print('          # of coupling constants per atom: %i' %numcoupling)
        print('          Extraction of third order coupling constants completed.\n')                               
        
    else:
        print(' ERROR: %s not found in the correct directory.\n Calculation aborted.' %cell_data[20])
        sys.exit()        

    print('      d - Extracted data:')
    #set extracted information to cell_data 
    cell_data[13] = omega
    cell_data[14] = omegazero
    cell_data[15] = eigenvecre
    cell_data[16] = eigenvecim
    cell_data[17] = qpoints
    cell_data[18] = eigenvecrezero
    
    #print data that is read in so far to the terminal
    print('          alat: \t %1.10e meters' %cell_data[1])
    print('          blat: \t %1.10e meters' %cell_data[2])
    print('          clat: \t %1.10e meters' %cell_data[3])
    print('          alpha: \t %f' %cell_data[27])
    print('          beta:  \t %f' %cell_data[28])
    print('          gamma: \t %f' %cell_data[29])
    print('          Volume: \t %1.10e meters^3' %cell_data[0])
    print('          natom: \t %s'%cell_data[4])
    print('          Tmin: \t %s' %cell_data[23])
    print('          Tmax: \t %s' %cell_data[24])
    print('          Tstep: \t %s\n' %cell_data[25])    
    attype = ''
    for i in range(len(cell_data[5])):
        if i %2 == 0:
            attype += cell_data[5][i]+' '
    print('          atom type: \t %s\n'%attype)
    print('          Dielectric: \t %6.3f %6.3f %6.3f' %(cell_data[8][0][0],cell_data[8][0][1],cell_data[8][0][2]))
    print('          \t\t %6.3f %6.3f %6.3f' %(cell_data[8][1][0],cell_data[8][1][1],cell_data[8][1][2]))
    print('          \t\t %6.3f %6.3f %6.3f' %(cell_data[8][2][0],cell_data[8][2][1],cell_data[8][2][2]))
    print('          Piezo: \t %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f' %(cell_data[9][0][0],cell_data[9][0][1],cell_data[9][0][2],cell_data[9][0][3],cell_data[9][0][4],cell_data[9][0][5]))
    print('          \t\t %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f' %(cell_data[9][1][0],cell_data[9][1][1],cell_data[9][1][2],cell_data[9][1][3],cell_data[9][1][4],cell_data[9][1][5]))
    print('          \t\t %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f' %(cell_data[9][2][0],cell_data[9][2][1],cell_data[9][2][2],cell_data[9][2][3],cell_data[9][2][4],cell_data[9][2][5]))
    for i in range(cell_data[4]):
        print('          BEC atom %i: \t %6.3f %6.3f %6.3f' %(i+1,cell_data[6][i][0],cell_data[6][i][1],cell_data[6][i][2]))
        print('          \t\t %6.3f %6.3f %6.3f' %(cell_data[6][i][3],cell_data[6][i][4],cell_data[6][i][5]))
        print('          \t\t %6.3f %6.3f %6.3f' %(cell_data[6][i][6],cell_data[6][i][7],cell_data[6][i][8]))    
    print('')
           
    return cell_data

def check_coupling(filename,lam,lamp,qp):
    """
    Author: Nicholas Pike
    Email : Nicholas.pike@smn.uio.no
    
    Purpose: Determine if the coupling constant calculation was completed and, 
             if so, read the file back into memory.
    
    Return: Check (True or False) and calana the array of coupling constants
    """
    #First, check if the file is in the directory
    check2 = 0
    if CHECK_FILE(filename):
        print('          Starting read in of coupling constants file.')
        check = True
        #now open file and look for the last line of the file
        calana = np.zeros(shape=(lam,lamp,qp))
        with open(filename,'r') as f:
            for num,line in enumerate(f,1):
                if 'COUPLING_CONSTANTS_CALCULATED' in line:
                    check2 = 0                    
                    #read in file
                    with open(filename,'r') as g:
                        for num,line in enumerate(g,1):
                            if num % 500 == 0: 
                                print('          . ')
                            if not line.startswith('#'):
                                l = line.strip('\n').split()
                                calana[0,0,int(l[0])]  = l[1]
                                calana[0,1,int(l[0])]  = l[2]
                                calana[0,2,int(l[0])]  = l[3]
                                calana[0,3,int(l[0])]  = l[4]
                                calana[0,4,int(l[0])]  = l[5]
                                calana[0,5,int(l[0])]  = l[6]
                                calana[0,6,int(l[0])]  = l[7]
                                calana[0,7,int(l[0])]  = l[8]
                                calana[0,8,int(l[0])]  = l[9]
                                calana[0,9,int(l[0])]  = l[10]
                                calana[0,10,int(l[0])] = l[11]
                                calana[0,11,int(l[0])] = l[12]
                                calana[1,0,int(l[0])]  = l[13]
                                calana[1,1,int(l[0])]  = l[14]
                                calana[1,2,int(l[0])]  = l[15]
                                calana[1,3,int(l[0])]  = l[16]
                                calana[1,4,int(l[0])]  = l[17]
                                calana[1,5,int(l[0])]  = l[18]
                                calana[1,6,int(l[0])]  = l[19]
                                calana[1,7,int(l[0])]  = l[20]
                                calana[1,8,int(l[0])]  = l[21]
                                calana[1,9,int(l[0])]  = l[22]
                                calana[1,10,int(l[0])] = l[23]
                                calana[1,11,int(l[0])] = l[24]
                                calana[2,0,int(l[0])]  = l[25]
                                calana[2,1,int(l[0])]  = l[26]
                                calana[2,2,int(l[0])]  = l[27]
                                calana[2,3,int(l[0])]  = l[28]
                                calana[2,4,int(l[0])]  = l[29]
                                calana[2,5,int(l[0])]  = l[30]
                                calana[2,6,int(l[0])]  = l[31]
                                calana[2,7,int(l[0])]  = l[32]
                                calana[2,8,int(l[0])]  = l[33]
                                calana[2,9,int(l[0])]  = l[34]
                                calana[2,10,int(l[0])] = l[35]
                                calana[2,11,int(l[0])] = l[36]
                                calana[3,0,int(l[0])]  = l[37]
                                calana[3,1,int(l[0])]  = l[38]
                                calana[3,2,int(l[0])]  = l[39]
                                calana[3,3,int(l[0])]  = l[40]
                                calana[3,4,int(l[0])]  = l[41]
                                calana[3,5,int(l[0])]  = l[42]
                                calana[3,6,int(l[0])]  = l[43]
                                calana[3,7,int(l[0])]  = l[44]
                                calana[3,8,int(l[0])]  = l[45]
                                calana[3,9,int(l[0])]  = l[46]
                                calana[3,10,int(l[0])] = l[47]
                                calana[3,11,int(l[0])] = l[48] 
                                calana[4,0,int(l[0])]  = l[49]
                                calana[4,1,int(l[0])]  = l[50]
                                calana[4,2,int(l[0])]  = l[51]
                                calana[4,3,int(l[0])]  = l[52]
                                calana[4,4,int(l[0])]  = l[53]
                                calana[4,5,int(l[0])]  = l[54]
                                calana[4,6,int(l[0])]  = l[55]
                                calana[4,7,int(l[0])]  = l[56]
                                calana[4,8,int(l[0])]  = l[57]
                                calana[4,9,int(l[0])]  = l[58]
                                calana[4,10,int(l[0])] = l[59]
                                calana[4,11,int(l[0])] = l[60]
                                calana[5,0,int(l[0])]  = l[61]
                                calana[5,1,int(l[0])]  = l[62]
                                calana[5,2,int(l[0])]  = l[63]
                                calana[5,3,int(l[0])]  = l[64]
                                calana[5,4,int(l[0])]  = l[65]
                                calana[5,5,int(l[0])]  = l[66]
                                calana[5,6,int(l[0])]  = l[67]
                                calana[5,7,int(l[0])]  = l[68]
                                calana[5,8,int(l[0])]  = l[69]
                                calana[5,9,int(l[0])]  = l[70]
                                calana[5,10,int(l[0])] = l[71]
                                calana[5,11,int(l[0])] = l[72]
                                calana[6,0,int(l[0])]  = l[73]
                                calana[6,1,int(l[0])]  = l[74]
                                calana[6,2,int(l[0])]  = l[75]
                                calana[6,3,int(l[0])]  = l[76]
                                calana[6,4,int(l[0])]  = l[77]
                                calana[6,5,int(l[0])]  = l[78]
                                calana[6,6,int(l[0])]  = l[79]
                                calana[6,7,int(l[0])]  = l[80]
                                calana[6,8,int(l[0])]  = l[81]
                                calana[6,9,int(l[0])]  = l[82]
                                calana[6,10,int(l[0])] = l[83]
                                calana[6,11,int(l[0])] = l[84]
                                calana[7,0,int(l[0])]  = l[85]
                                calana[7,1,int(l[0])]  = l[86]
                                calana[7,2,int(l[0])]  = l[87]
                                calana[7,3,int(l[0])]  = l[88]
                                calana[7,4,int(l[0])]  = l[89]
                                calana[7,5,int(l[0])]  = l[90]
                                calana[7,6,int(l[0])]  = l[91]
                                calana[7,7,int(l[0])]  = l[92]
                                calana[7,8,int(l[0])]  = l[93]
                                calana[7,9,int(l[0])]  = l[94]
                                calana[7,10,int(l[0])] = l[95]
                                calana[7,11,int(l[0])] = l[96]                                    
                                calana[8,0,int(l[0])]  = l[97]
                                calana[8,1,int(l[0])]  = l[98]
                                calana[8,2,int(l[0])]  = l[99]
                                calana[8,3,int(l[0])]  = l[100]
                                calana[8,4,int(l[0])]  = l[101]
                                calana[8,5,int(l[0])]  = l[102]
                                calana[8,6,int(l[0])]  = l[103]
                                calana[8,7,int(l[0])]  = l[104]
                                calana[8,8,int(l[0])]  = l[105]
                                calana[8,9,int(l[0])]  = l[106]
                                calana[8,10,int(l[0])] = l[107]
                                calana[8,11,int(l[0])] = l[108]
                                calana[9,0,int(l[0])]  = l[109]
                                calana[9,1,int(l[0])]  = l[110]
                                calana[9,2,int(l[0])]  = l[111]
                                calana[9,3,int(l[0])]  = l[112]
                                calana[9,4,int(l[0])]  = l[113]
                                calana[9,5,int(l[0])]  = l[114]
                                calana[9,6,int(l[0])]  = l[115]
                                calana[9,7,int(l[0])]  = l[116]
                                calana[9,8,int(l[0])]  = l[117]
                                calana[9,9,int(l[0])]  = l[118]
                                calana[9,10,int(l[0])] = l[119]
                                calana[9,11,int(l[0])] = l[120]
                                calana[10,0,int(l[0])]  = l[121]
                                calana[10,1,int(l[0])]  = l[122]
                                calana[10,2,int(l[0])]  = l[123]
                                calana[10,3,int(l[0])]  = l[124]
                                calana[10,4,int(l[0])]  = l[125]
                                calana[10,5,int(l[0])]  = l[126]
                                calana[10,6,int(l[0])]  = l[127]
                                calana[10,7,int(l[0])]  = l[128]
                                calana[10,8,int(l[0])]  = l[129]
                                calana[10,9,int(l[0])]  = l[130]
                                calana[10,10,int(l[0])] = l[131]
                                calana[10,11,int(l[0])] = l[132]
                                calana[11,0,int(l[0])]  = l[133]
                                calana[11,1,int(l[0])]  = l[134]
                                calana[11,2,int(l[0])]  = l[135]
                                calana[11,3,int(l[0])]  = l[136]
                                calana[11,4,int(l[0])]  = l[137]
                                calana[11,5,int(l[0])]  = l[138]
                                calana[11,6,int(l[0])]  = l[139]
                                calana[11,7,int(l[0])]  = l[140]
                                calana[11,8,int(l[0])]  = l[141]
                                calana[11,9,int(l[0])]  = l[142]
                                calana[11,10,int(l[0])] = l[143]
                                calana[11,11,int(l[0])] = l[144]                                    
                else:
                    check2 = 1
                    
        print('          Finished read in of coupling constants file.')
    else:
        check = False
        calana = []
        
    if check2 == 1:
        print('          File is incomplete. Recalculating coupling constants.')
        check = False
        calana = []
        
    
    return check,calana

def calc_anharmonic(cell_data):
    """
    Author: Nicholas Pike
    Email:  Nicholas.pike@smn.uio.no
    
    Purpose: precalculation of the anharmonic coefficient in parallel or serial 
             depending on the number of coupling constants
             
    Return: array of coupling constants
    """
    lam             = len(cell_data[14])  # length of omegazero i.e. number of phonon modes
    lamp            = len(cell_data[14])  # number of phonon modes 
    qpoint          = cell_data[17]       # list of qpoints
    max_core_number = cell_data[22]
    steps           = len(qpoint)
    numcoupling     = cell_data[26]
    
    #check if the coupling constants have already been calculated
    print('      a - Checking if coupling constants have been calculated...')
    checkcoupling, calcana = check_coupling('coupling_constants',lam,lamp,len(qpoint))
    if checkcoupling:
        print('          Coupling constants have already been calculated. Moving on.\n')
        
    else:
        print('          Coupling constants not calculated. Doing so now...\n')
        print('           --- This will take a while---')
        #build array for storage of results    
        calcana = np.zeros(shape=(lam,lamp,len(qpoint)))
        
        #determine how the calculation will be parallized
        max_cores, max_loops, chunk = deter_max_cores(steps,max_core_number)
        #max_cores is the number of cores used for this calculation
        #max_loops is the max_number of loops those cores will be looped over
        #chunk is the length of the chunk of the interable array send to each core
        
        #name of function which will be parallized
        func_name = anharmonic_coupling 
    
        #check first if parallelization is even necessary.  There is a considerable 
        #delay time for short calculations using this parallization scheme.
        if numcoupling <= 10:
            print('      Choosing to run on a single core due to low number of couplings (%i)' %numcoupling)
            for i in range(lam):
                for j in range(lamp):
                    for l in range(len(qpoint)):
                        calcana[i,j,l] = anharmonic_coupling(i,j,cell_data,l)
        else:
            print('      Calculation of anaharmonic coupling constants is \n       done in parallel over q points.')
            print('      Number of cores used:            %s' %max_cores)
            print('      Number of loops over cores:      %s' %max_loops)
            print('      Size of array chunk:             %s' %chunk)
            print('      lambda times lambda prime:       %i' %(int(lam*lamp)))
            print('      Thus, the total number of loops: %i\n' %(int(lam*lamp*max_loops)))
            c = 0    
            iterable = []
            for i in range(lam):
                for j in range(lamp):
                    outputs  = []
                    for y in range(int(max_loops)):
                        if (steps-y*max_cores*chunk)>= max_cores*chunk:
                            iterable = range(len(qpoint))
                            func     = partial(func_name,i,j,cell_data)
                            pool     = Pool(processes=max_cores)
                            output   = pool.map(func, iterable,chunk)
                            outputs.extend(output)
                                           
                        else:
                            iterable = range(len(qpoint))
                            func     = partial(func_name,i,j,cell_data)
                            pool     = Pool(processes=max_cores)
                            output   = pool.map(func, iterable,chunk)
                            outputs.extend(output)

                        # Using spyder to interactively use this program results in a error when 
                        # the pool is closed.  This prevents problems on spyder but allows the code
                        # to work from the command line                
                        if not any('SPYDER' in name for name in os.environ):
                            pool.close()
                            pool.join()
                        for n in range(len(outputs)):
                            calcana[i,j,n] = outputs[n]
                        
                        #calculate the correct number of steps
                        c += max_cores*chunk 
                        
        print('      c - Printing coupling constants to a file.')                
        #print the coupling constant as a function of mode and q point
        f1= open('coupling_constants','w')
        f1.write('#Coupling constants \n')
        f1.write('# qpoint\t 00 \t 01\t .. lambdalambdap\n' )
        for l in range(calcana.shape[2]):
            f1.write('%i ' %l)
            for j in range(calcana.shape[1]):
                for i in range(calcana.shape[0]):
                    f1.write('%s '%(calcana[i,j,l]))         
            f1.write('\n')
        f1.write('#THIS LINE ENDS THE FILE\n')
        f1.write('#COUPLING_CONSTANTS_CALCULATED')
        f1.close()
    
    return calcana

def anharmonic_coupling(zeromode,qmode,cell_data,qp):
    """
    Author: Nicholas Pike
    Email:  Nicholas.pike@smn.uio.no
    
    Purpose: Calculates the anharmonic coupling constant V3 for the given 
             zero mode frequency, qpoint, negative q point, and qmode
             
             cell_data is an array of information
             zeromode is an index
             qmode is an index
             qp is an index 
    
    Return: array of coupling constants    
    """
    #initialize variables from read in file. 
    alat        = cell_data[1]
    blat        = cell_data[2]
    clat        = cell_data[3]
    natom       = cell_data[4]
    mass        = cell_data[5]
    omega       = cell_data[13]
    omegazero   = cell_data[14]
    eigre       = cell_data[15]
    eigim       = cell_data[16]
    qpoint      = cell_data[17]
    eigzero     = cell_data[18] # structured as # of phonon modes by 3*natom by
    third_order = cell_data[21] # structured as natom, # of coupling constants, 
                                # 0-2 atom type, 3-11 positions of 3 atoms, 
                                # 12-38 third order force constants
    
    #we need the reversed (i.e. imaginary part changes sign.                                
    #imeig       = -cell_data[16]
    
    #allocating memory
    V3    = 0
    const = 0
    ep    = 0
    cpex  = 0
    cup   = [[],[],[]]
    
    #check if qp is an index or an array
    if omegazero[zeromode] == 0 or omega[qp,qmode] == 0:
        V3 = 0
    else:
        for i in range(natom):       
            for l in range(third_order.shape[1]):
                #what atoms are we summing over in the triplet
                tyat1 = third_order[i][l][0]-1
                tyat2 = third_order[i][l][1]-1
                tyat3 = third_order[i][l][2]-1  
                
                #get atom masses
                m1 = float(mass[2*int(tyat1)+1])
                m2 = float(mass[2*int(tyat2)+1])
                m3 = float(mass[2*int(tyat3)+1]) 
                                
                #first term in the summation
                const = np.sqrt(hbar**3.0/(8.0*m1*m2*m3*omegazero[zeromode]*omega[qp,qmode]*omega[qp,qmode]))
                
                #determine eigenvalues for the second and third atom in each direction
                eigx2  = eigre[qp][qmode][int(2*tyat2+0)]+1j*eigim[qp][qmode][int(2*tyat2+0)]
                eigy2  = eigre[qp][qmode][int(2*tyat2+1)]+1j*eigim[qp][qmode][int(2*tyat2+1)]
                eigz2  = eigre[qp][qmode][int(2*tyat2+2)]+1j*eigim[qp][qmode][int(2*tyat2+2)]
                
                eigmx3 = eigre[qp][qmode][int(2*tyat3+0)]-1j*eigim[qp][qmode][int(2*tyat3+0)]
                eigmy3 = eigre[qp][qmode][int(2*tyat3+1)]-1j*eigim[qp][qmode][int(2*tyat3+1)]
                eigmz3 = eigre[qp][qmode][int(2*tyat3+2)]-1j*eigim[qp][qmode][int(2*tyat3+2)]   
                                
                #make these into an array
                veceig2  = np.array([eigx2,eigy2,eigz2])
                veceigm3 = np.array([eigmx3,eigmy3,eigmz3])
                veceigz  = np.array([eigzero[zeromode][int(2*tyat1+0)],eigzero[zeromode][int(2*tyat1+1)],eigzero[zeromode][int(2*tyat1+2)]])
                
                #what are the positions of those atoms
                pos2     = np.array([third_order[i][l][6],third_order[i][l][ 7],third_order[i][l][ 8]])
                pos3     = np.array([third_order[i][l][9],third_order[i][l][10],third_order[i][l][11]])
                posdiff  = pos2-pos3
                
                #pos is unitless, so change units
                pos      = np.array([posdiff[0]*alat/ang_to_m,posdiff[1]*blat/ang_to_m,posdiff[2]*clat/ang_to_m])
                ep       = np.exp(1j*np.dot(qpoint[qp],pos))
                                              
                #now calculate via matrix multiplication
                cup[0] = np.array([[third_order[i][l][12],third_order[i][l][13],third_order[i][l][14]],
                                   [third_order[i][l][15],third_order[i][l][16],third_order[i][l][17]],
                                   [third_order[i][l][18],third_order[i][l][19],third_order[i][l][20]]])
                
                cup[1] = np.array([[third_order[i][l][21],third_order[i][l][22],third_order[i][l][23]],
                                   [third_order[i][l][24],third_order[i][l][25],third_order[i][l][26]],
                                   [third_order[i][l][27],third_order[i][l][28],third_order[i][l][29]]])

                cup[2] = np.array([[third_order[i][l][30],third_order[i][l][31],third_order[i][l][32]],
                                   [third_order[i][l][33],third_order[i][l][34],third_order[i][l][35]],
                                   [third_order[i][l][36],third_order[i][l][37],third_order[i][l][38]]])
                
                cpex = (veceigz[0]*np.sum(np.dot(cup[0],np.outer(veceig2,veceigm3)))+
                        veceigz[1]*np.sum(np.dot(cup[1],np.outer(veceig2,veceigm3)))+
                        veceigz[2]*np.sum(np.dot(cup[2],np.outer(veceig2,veceigm3))))
                
                V3 += const*cpex*ep

    return np.real(V3)

def deter_max_cores(num_steps,max_core_number):
    """
    Author: Nicholas Pike 
    Email:  Nicholas.pike@ulg.ac.be
    
    Args: num_steps       - variable we wish to parallize over
          max_core_number -
    
    Returns:  max_cores  - maximum number of cores to use in this 
                           calculation
              max_loop   - maximum number of loops 
              chunk      - chunk size of interable parameter 

    """  
    # Declare variables for later use:
    max_loops = 0
    max_num_loops = max_core_number
    
    # First, determine the number of cores that computer or machine has. This 
    # will determine the largest number of cores that can be used.
    num_cores = max_core_number 
    
    # Determine number of loops based on the number of max cores by dividing
    # the num_steps by the num_loops 
    num_loops = float(num_steps)/float(num_cores)
    
    #determine number of size of the chunk to keep the number of loops at a max value
    chunk = np.ceil(num_steps/max_num_loops) #will give one extra element for the chunk
    
    num_loops = float(num_steps)/float(num_cores*chunk)
    
    #set variables
    max_cores = num_cores
    max_loops = np.ceil(num_loops)
    chunk     = chunk
    
                       
    return int(max_cores), int(max_loops), int(chunk)

def bash_commands(bashcommand):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Execute bash command and output any error message
    
    Return: string with a message (if any) about an error in the script
    """
    output = subprocess.Popen([bashcommand],stdout=subprocess.PIPE,shell=True).communicate()[0].decode('utf-8').strip()
    
    return str(output)

def bash_commands_cwd(bashcommand,wd):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Execute bash command and output any error message
    
    Return: string with a message (if any) about an error in the script
    """
    output = subprocess.Popen([bashcommand],stdout=subprocess.PIPE,shell=True,cwd=wd).communicate()[0].decode('utf-8').strip('\n')
    
    return str(output)

def make_folder(folder_name):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: execute bash command to generate a new folder called "folder_name"
        
    """
    bashcommand = 'mkdir '+folder_name
    output = bash_commands(bashcommand)
    
    return output

def copy_file(originalfile,newfile):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: To move a file to a new folder, keeping a copy of that file in the orginal folder if original == True 
    """

    bashcp = 'cp '+originalfile+' '+newfile
    output = bash_commands(bashcp)
        
    return output

def move_file(filename,newlocation,original):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: To move a file to a new folder, keeping a copy of that file in the orginal folder if original == True 
    """
    
    if original == True:
        bashcp = 'cp '+filename+' MOVEDCOPY'
        output = bash_commands(bashcp)
        
        bashmv = 'mv MOVEDCOPY '+newlocation+'/'+filename
        output = bash_commands(bashmv)
        
    else:
        bashmv = 'mv '+filename+' '+newlocation+'/'+filename
        output = bash_commands(bashmv)
        
    return output

def remove_file(filename):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Remove a file 
    """
    
    bashrm = 'rm '+filename
    output = bash_commands(bashrm)
    
    return output

def launch_calc(location,nameofbatch):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Launch calculation using the bash script nameofbatch
    """
    bashsub = 'sbatch '+nameofbatch
    output = bash_commands_cwd(bashsub,wd = __root__+'/'+location+'/')
    
    print(output)
    
    return None

def launch_calc_root(nameofbatch):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Launch calculation using the bash script nameofbatch
    """
    bashsub = 'sbatch '+nameofbatch
    output = bash_commands(bashsub)
    
    print(output)
    
    return None

def launch_script(location,nameofbatch):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Launch calculation using the bash script nameofbatch
    """
    bashsub = './'+nameofbatch
    output = bash_commands_cwd(bashsub,wd = __root__+'/'+location+'/')
    
    print(output)
    
    return None

def vasp_converge(calctype):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Simple function to determine if the VASP calculation completed 
    """
    check = "False"
    
    #stopping phrases in OUTCAR file
    Relaxation_phrase = ' reached required accuracy - stopping structural energy minimisation'
    Elastic_phrase    = ' Eigenvectors and eigenvalues of the dynamical matrix' 
    TDEP_phrase       = ' aborting loop because EDIFF is reached' 
    
    #now execute loop statements
    if calctype == 'Relaxation':
        with open('OUTCAR','r') as out:
              for line in out:
                    if Relaxation_phrase in line:
                        check = "True"
                        break
                    
    elif calctype == 'Elastic':
        with open('OUTCAR','r') as out:
              for line in out:
                    if Elastic_phrase in line:
                        check = "True"
                        break
                    
    elif calctype == 'TDEP':
        with open('OUTCAR','r') as out:
          for line in out:
                if TDEP_phrase in line:
                    check = "True"
                    break    
    print(check)
    
    return None

def generate_LOTO():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: This file extracts the dielectric tensor and born effective charge tensors from data_extract
             This code is very similiar to a code written by Olle Hellman that does the same thing
    
    Return: None
    """
    #get data from data_extraction
    dietensor = np.zeros(shape=(3,3))
    bectensor = [] 
    with open('../data_extraction','r') as datafile:
        for i,line in enumerate(datafile):
            if 'dielectric' in line:
                l = line.split()
                dietensor[0][0]= float(l[1])
                dietensor[0][1]= float(l[2])
                dietensor[0][2]= float(l[3])
                dietensor[1][0]= float(l[4])
                dietensor[1][1]= float(l[5])
                dietensor[1][2]= float(l[6])
                dietensor[2][0]= float(l[7])
                dietensor[2][1]= float(l[8])
                dietensor[2][2]= float(l[9])
                
            elif 'natom' in line:
                l = line.split()
                natom = l[1]
            elif 'bectensor' in line:
                bectensor = np.zeros(shape=(int(natom),3,3))
                for j in range(int(natom)):
                    if j == 0:
                        data1 = linecache.getline('../data_extraction',i+j+1).split()
                        bectensor[j][0][0] = float(data1[2])
                        bectensor[j][0][1] = float(data1[3])
                        bectensor[j][0][2] = float(data1[4])
                        bectensor[j][1][0] = float(data1[5])
                        bectensor[j][1][1] = float(data1[6])
                        bectensor[j][1][2] = float(data1[7])
                        bectensor[j][2][0] = float(data1[8])
                        bectensor[j][2][1] = float(data1[9])
                        bectensor[j][2][2] = float(data1[10]) 
                    else:
                        data1 = linecache.getline('../data_extraction',i+j+1).split()
                        bectensor[j][0][0] = float(data1[1])
                        bectensor[j][0][1] = float(data1[2])
                        bectensor[j][0][2] = float(data1[3])
                        bectensor[j][1][0] = float(data1[4])
                        bectensor[j][1][1] = float(data1[5])
                        bectensor[j][1][2] = float(data1[6])
                        bectensor[j][2][0] = float(data1[7])
                        bectensor[j][2][1] = float(data1[8])
                        bectensor[j][2][2] = float(data1[9]) 
    #now print data to file
    f = open('infile.lotosplitting','w')
    f.write('%f %f %f\n' %(dietensor[0][0],dietensor[0][1],dietensor[0][2]))
    f.write('%f %f %f\n' %(dietensor[1][0],dietensor[1][1],dietensor[1][2]))
    f.write('%f %f %f\n' %(dietensor[2][0],dietensor[2][1],dietensor[2][2]))
    for i in range(int(natom)):
        f.write('%f %f %f\n' %(bectensor[i][0][0],bectensor[i][0][1],bectensor[i][0][2]))
        f.write('%f %f %f\n' %(bectensor[i][1][0],bectensor[i][1][1],bectensor[i][1][2]))
        f.write('%f %f %f\n' %(bectensor[i][2][0],bectensor[i][2][1],bectensor[i][2][2]))
    f.close()
    
    return None

def find_gcd(x, y):
    """
    Determines the gcd for an array of values.
    """
    while(y):
        x, y = y, x % y
    return x

def find_norm(vec,scale):
    """
    Determines the norm of a three componennt vector 
    """
    norm = np.sqrt((scale*float(vec[0]))**2+(scale*float(vec[1]))**2+(scale*float(vec[2]))**2)
    return norm

def find_debye():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Determine the debye temperature from the calculated elastic tensor
    
    Return: Debye Temperature
    """
    debye_temp = 0
    density    = 0
    masstot    = 0
    molarmass  = 0
    data       = ['',0,0,0,0]
    molar      = []
    eltensor   = np.zeros(shape=(6,6))
    
    #read information from data_extraction file
    with open('data_extraction','r') as datafile:
        for line in datafile:
            if line.startswith('elastic'):
                l = line.split()
                eltensor[0][0]= float(l[1])
                eltensor[0][1]= float(l[2])
                eltensor[0][2]= float(l[3])
                eltensor[0][3]= float(l[4])
                eltensor[0][4]= float(l[5])
                eltensor[0][5]= float(l[6])
                eltensor[1][0]= float(l[7])
                eltensor[1][1]= float(l[8])
                eltensor[1][2]= float(l[9])
                eltensor[1][3]= float(l[10])
                eltensor[1][4]= float(l[11])
                eltensor[1][5]= float(l[12])
                eltensor[2][0]= float(l[13])
                eltensor[2][1]= float(l[14])
                eltensor[2][2]= float(l[15])
                eltensor[2][3]= float(l[16])
                eltensor[2][4]= float(l[17])
                eltensor[2][5]= float(l[18])
                eltensor[3][0]= float(l[19])
                eltensor[3][1]= float(l[20])
                eltensor[3][2]= float(l[21])
                eltensor[3][3]= float(l[22])
                eltensor[3][4]= float(l[23])
                eltensor[3][5]= float(l[24])
                eltensor[4][0]= float(l[25])
                eltensor[4][1]= float(l[26])
                eltensor[4][2]= float(l[27])
                eltensor[4][3]= float(l[28])
                eltensor[4][4]= float(l[29])
                eltensor[4][5]= float(l[30])
                eltensor[5][0]= float(l[32])
                eltensor[5][1]= float(l[32])
                eltensor[5][2]= float(l[33])
                eltensor[5][3]= float(l[34])
                eltensor[5][4]= float(l[35])
                eltensor[5][5]= float(l[36])
                data[1] = eltensor
                
            elif line.startswith('volume'):
                l = line.split()
                data[3] = float(l[1])*ang_to_m**3
                
            elif line.startswith('atpos'):
                l = line.split()
                u,data[4] = np.unique(l[1:],return_counts = True)
                for i in range(len(u)):
                    m  = element(u[i]).mass
                    molar = np.append(molar,m)
                data[2] = molar
                    
            elif line.startswith('natom'):
                l = line.split()
                natom = float(l[1])
                   
    #calculate molar mass
    for i in range(len(data[2])):
        molarmass += data[2][i]
    
    #calculate density
    for i in range(len(data[2])):
        masstot +=data[2][i]*data[4][i] #data[2] is the mass data[4] is the multiplicity
    density = masstot*amu_kg/data[3]/kgm3_to_cm3 #data[3] is the volume in m**3
    
    #number of unit cells 
    uc_count = 0
    for i in range(len(data[4])):
        uc_count = find_gcd(uc_count,data[4][i])
    
    
    #calculate number of atoms per molecule
    atom_molecule = natom/uc_count
    
    #calculate compliance tensor
    if np.all(data[1]==0):
        s = np.zeros(shape=(6,6))
    else:
        s = np.linalg.inv(data[1])
    
    #calculate bulk and shear modulus
    B  = 1.0/9.0*((data[1][0][0]+data[1][1][1]+data[1][2][2])+2.0*(data[1][0][1]+data[1][0][2]+data[1][1][2]))
    BR = 1.0/((s[0,0]+s[1,1]+s[2,2])+2.0*(s[0,1]+s[1,2]+s[0,2]))
    G  = 1.0/15.0*((data[1][0][0]+data[1][1][1]+data[1][2][2])-(data[1][0][1]+data[1][0][2]+data[1][1][2])+3.0*(data[1][3][3]+data[1][4][4]+data[1][5][5]))
    GR = 15.0/(4.0*(s[0,0]+s[1,1]+s[3,3])-4.0*(s[0,1]+s[1,2]-s[0,2])+3.0*(s[3,3]+s[4,4]+s[5,5])) 

    #calculate universal elastic anisotropy constant
    Au = 5.0*(G/GR)+B/BR - 6.0
    
    #calculate shear velocity
    vs = np.sqrt(G*kbar_to_GPa*1E9/(density*kgm3_to_cm3))
    
    #calculate longitudional velocity
    vl  = np.sqrt((B+4.0/3.0*G)*kbar_to_GPa*1E9/(density*kgm3_to_cm3))
    
    #acoustic velocity
    va = (1.0/3.0*((1.0/vl**3)+(2.0/vs**3)))**(-1.0/3.0)
    
    #Calculate Debye temperature
    debye_temp  = h/kb*((3.0*atom_molecule*Na*density)/(4.0*np.pi*molarmass))**(1.0/3.0)*va*m_to_cm
    
    #print information to file
    printdebye = True
    with open('data_extraction','r') as f:
        for line in f:
            if 'Debye' in line:
                printdebye = False
                
    if printdebye:
        f1 =  open('data_extraction','a')
        f1.write('Debye %f\n'%debye_temp)
        f1.write('Bulk Mod %f\n' %B)
        f1.write('# Additional elastic information\n')
        f1.write('#Bulk Mod (Voigt Avg): %.2f kbar\n' %B)
        f1.write('#Bulk Mod (Reuss Avg): %.2f kbar\n' %BR)
        f1.write('#Shear Mod (Voigt Avg): %.2f kbar\n' %G)
        f1.write('#Shear Mod (Reuss Avg): %.2f kbar\n' %GR)
        f1.write('#Universal Elastic Aniostropy: %.2f\n' %Au)
        f1.write('#Avg. Vel.  %.2f m/s\n'%va)
        f1.write('#Shear Vel. %.2f m/s\n'%vs)
        f1.write('#Long. Vel. %.2f m/s\n'%vl)
        f1.close()
            
    return debye_temp

def generate_KPOINT(kdensity):
    """
    Author:Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Generate K-point file using the specified kpoint density
    
    """
    kdist=1.0/float(kdensity)

    #read  POSCAR 
    infile = open('POSCAR', 'r')  # open file for reading 

    infile.readline() #comment line
    scaleline = infile.readline()
    scale = float(scaleline)

    vec1line = infile.readline()
    vec2line = infile.readline()
    vec3line = infile.readline()
    
    #determine quantities from POSCAR file
    vec1 = []; vec2 = []; vec3 = []
    vec1 = np.array(vec1line.split(), dtype=np.float32)
    vec2 = np.array(vec2line.split(), dtype=np.float32)
    vec3 = np.array(vec3line.split(), dtype=np.float32)

    alength = scale*np.sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1] + vec1[2]*vec1[2])
    blength = scale*np.sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1] + vec2[2]*vec2[2])
    clength = scale*np.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
    
	# Calculate required density of k-points
    nkx = int(np.ceil(2.0*np.pi/(alength*kdist)))
    nky = int(np.ceil(2.0*np.pi/(blength*kdist)))
    nkz = int(np.ceil(2.0*np.pi/(clength*kdist)))
    nkxt = str(nkx)
    nkyt = str(nky)
    nkzt = str(nkz)
	
    # Print file(s)
    kpointfile = open('KPOINTS','w')
    kpointfile.write('k-density: %.1f\n'\
                     '0\n'\
                     'Gamma\n'\
                     '%s %s %s\n'
                     '0  0  0'%(kdensity,nkxt,nkyt,nkzt))
            
    return None

def generate_POTCAR():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: generate potcar file using the poscar file to read in the atom types
    """
    #open POSCAR for reading
    
    infile = open('POSCAR','r') #open file for reading
    
    infile.readline() #comment line
    infile.readline() #scale 
    infile.readline() #first vector
    infile.readline() #second vector
    infile.readline() #third vector
    atomnames = infile.readline()
    
    atom_name = atomnames.split()
    
    string = ''
    for i in range(len(atom_name)):
        string += '../pseudos/'+atom_name[i]+'/POTCAR '
    #create POTCAR file
    bashpotcar = 'cat '+string+' > POTCAR'
    output = bash_commands(bashpotcar) 
    
    output = output  #removes warning
    
    return None

def generate_INCAR(incartype):
    """
    Author: Nicholas Pike
    Email : Nicholas.pike@smn.uio.no
    
    Purpose: Generate INCAR file for vasp calculations using some predefined 
             quantities and convergence parameters. The generated incar file is 
             placed in the correct folder at the time of its creation.
             
    Return: None
    """
    name  = 'INCAR'
    ECUT  = 'ENCUT = '+ecut+'\n'
    EDIFF = 'EDIFF = '+ediff+'\n'
        
    if incartype == 'Relaxation':
        #print file
        f = open(name,'w') #printed to an internal directory
        f.write('INCAR for ionic relaxation   (AUTOMATICALLY GENERATED)\n')
        f.write('\n')
        f.write('! Electronic relaxation\n')
        f.write('IALGO   = 48      ! Algorithm for electronic relaxation\n')
        f.write('NELMIN = 4         ! Minimum # of electronic steps\n')
        f.write(EDIFF)
        f.write(ECUT)
        f.write('PREC   = Accurate  ! Normal/Accurate\n')
        f.write('LREAL  = Auto      ! Projection in reciprocal space?\n')
        f.write('ISMEAR = 1         ! Smearing of partial occupancies. Metals: 1; else < 1.\n')
        f.write('SIGMA  = 0.2       ! Smearing width\n')
        f.write('ISPIN  = 1         ! Spin polarization? 1-no 2- yes\n')
        f.write('\n')
        f.write('! Ionic relaxation\n')
        f.write('EDIFFG = -0.0005     ! Tolerance for ions\n')
        f.write('NSW    = 800        ! Max # of ionic steps\n')
        f.write('MAXMIX = 80        ! Keep dielectric function between ionic movements\n')
        f.write('IBRION = 2         ! Algorithm for ions. 0: MD 1: QN/DIIS 2: CG\n')
        f.write('ISIF   = 3         ! Relaxation. 2: ions 3: ions+cell\n')
        f.write('ADDGRID= .TRUE.    ! More accurate forces with PAW\n')
        f.write('\n')
        f.write('! Output options\n')
        f.write('NWRITE = 1         ! Write electronic convergence at first step only\n')
        f.write('\n')
        f.write('! Memory handling\n')
        f.write('LPLANE  = .TRUE.\n')
        f.write('LSCALU  = .FALSE\n')
        f.write('NSIM    = 4\n')
        f.write('NCORE  = 4\n')
        f.write('LREAL=.FALSE.\n')
        f.write('\n')
        f.close()
               
    elif incartype == 'Elastic':
        #print file
        f1 = open(name,'w') #printed to an internal directory
        f1.write('INCAR for elastic tensor (AUTOMATICALLY GENERATED)\n')
        f1.write(ECUT)
        f1.write(EDIFF)
        f1.write('PREC   = Accurate  ! Normal/Accurate\n')
        f1.write('LREAL  = .FALSE.      ! Projection in reciprocal space?\n')
        f1.write('ISMEAR = -5         ! Smearing of partial occupancies. k-points >2: -5; else 0\n')
        f1.write('SIGMA  = 0.2       ! Smearing width\n')
        f1.write('ISPIN  = 1         ! Spin polarization?\n')
        f1.write('\n')
        f1.write('## calculation of the q=0 phonon modes and eigenvectors.  Elastic tensor also calculated\n')
        f1.write('IBRION = 6\n')
        f1.write('ISIF = 3\n')
        f1.write('LEPSILON = .TRUE.\n')
        f1.write('\n')
        f1.close()
        
    elif incartype == 'TDEP':
        #print file
        f1 = open(name,'w') #printed to an internal directory
        f1.write('INCAR for molecular dynamics\n')
        f1.write('\n')
        f1.write('! Electronic relaxation\n')
        f1.write('ALGO   = Fast      ! Algorithm for electronic relaxation\n')
        f1.write('NELMIN = 4         ! Minimum # of electronic steps\n')
        f1.write('NELM   = 500       ! Maximum # of electronic steps\n')
        f1.write(EDIFF)
        f1.write(ECUT)
        f1.write('PREC   = High      ! Normal/Accurate\n')
        f1.write('LREAL  = .False.   ! Projection in reciprocal space? False gives higher accuracy\n')
        f1.write('ISMEAR = -5         ! Smearing of partial occupancies. k-points >2: -5; else 0\n')
        f1.write('SIGMA  = 0.2       ! Smearing width\n')
        f1.write('ISPIN  = 1         ! Spin polarization?\n')
        f1.write('\n')
        f1.write('! Ionic relaxation\n')
        f1.write('NSW    = 0         ! Number of MD steps\n')
        f1.write('ISIF    = 2\n')
        f1.write('\n')
        f1.write('! Output options\n')
        f1.write('!LWAVE  = .FALSE.   ! Write WAVECAR?\n')
        f1.write('!LCHARG = .FALSE.   ! Write CHGCAR?\n')
        f1.write('NWRITE = 1         ! Write electronic convergence etc. at first and last step only\n')
        f1.write('\n')
        f1.write('! Memory handling\n')
        f1.write('NCORE   = 8\n')
        f1.close()
        
    elif incartype == 'TDEP_rlx':
        #print file
        f = open(name,'w') #printed to an internal directory
        f.write('INCAR for ionic relaxation   (AUTOMATICALLY GENERATED)\n')
        f.write('\n')
        f.write('! Electronic relaxation\n')
        f.write('IALGO   = 48      ! Algorithm for electronic relaxation\n')
        f.write('NELMIN = 4         ! Minimum # of electronic steps\n')
        f.write(EDIFF)
        f.write(ECUT)
        f.write('PREC   = Accurate  ! Normal/Accurate\n')
        f.write('LREAL  = Auto      ! Projection in reciprocal space?\n')
        f.write('ISMEAR = 1         ! Smearing of partial occupancies. Metals: 1; else < 1.\n')
        f.write('SIGMA  = 0.2       ! Smearing width\n')
        f.write('ISPIN  = 1         ! Spin polarization? 1-no 2- yes\n')
        f.write('\n')
        f.write('! Ionic relaxation\n')
        f.write('EDIFFG = -0.0005     ! Tolerance for ions\n')
        f.write('NSW    = 800        ! Max # of ionic steps\n')
        f.write('MAXMIX = 80        ! Keep dielectric function between ionic movements\n')
        f.write('IBRION = 1         ! Algorithm for ions. 0: MD 1: QN/DIIS 2: CG\n')
        f.write('ISIF   = 2         ! Relaxation. 2: ions 3: ions+cell\n')
        f.write('ADDGRID= .TRUE.    ! More accurate forces with PAW\n')
        f.write('\n')
        f.write('! Output options\n')
        f.write('NWRITE = 1         ! Write electronic convergence at first step only\n')
        f.write('\n')
        f.write('! Memory handling\n')
        f.write('LPLANE  = .TRUE.\n')
        f.write('LSCALU  = .FALSE\n')
        f.write('NSIM    = 4\n')
        f.write('NCORE  = 4\n')
        f.write('LREAL=.FALSE.\n')
        f.write('\n')
        f.close()
                   
    else:
        print('INCAR type not programmed. Aborting.')
        sys.exit()
        
    return None

def generate_batch(batchtype,batchname,tags):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Automatic generation of the submission script for each HPC system
             and for each calculation type. The batch script is printed to the correct 
             folder for each calculation.

             batchtype - relax, elastic, etchttps://www.researchgate.net/
             batchname - name of batchfile
             
    Return: None
    """
    #unpack tags
    if tags != '':
        withsolver  = tags[0]
        withntvol   = tags[1]
    else:
        withsolver  = False
        withntvol   = False
    
    #generate scripts
    if batchtype == 'Relaxation' :  
        #print file
        f = open(batchname,'w')
        f.write('#!/bin/bash\n')
        f.write(batch_header_rlx)
        f.write('\n')
        f.write(VASPSR+'\n')
        f.write(PYTHMOD+'\n')
        f.write('\n')
        f.write('## submit vasp job\n')
        f.write('if [ -f "OUTCAR" ] && [ `python ../../pyroelectric_coeff.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('echo "Already converged in previous run, moving on"\n') 
        f.write('   python ../../pyroelectric_coeff.py --outcar\n')
        f.write('   echo "Launching calculations for elastic and dielectric tensors"\n')
        f.write('   python ../../pyroelectric_coeff.py --copy_file CONTCAR CONTCAR2\n')
        f.write('   python ../../pyroelectric_coeff.py --move_file CONTCAR2 ../Elastic True\n')
        f.write('   python ../../pyroelectric_coeff.py --generate_batch Elastic batch.sh\n')
        f.write('   python ../../pyroelectric_coeff.py --move_file batch.sh ../Elastic False\n')
        f.write('   python ../../pyroelectric_coeff.py --launch_calc ../Elastic batch.sh\n')
        f.write('else\n')
        f.write('echo "Starting first relaxation."\n')
        f.write('$MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write('rm -f WAVECAR CHG\n')
        f.write('if [ `python ../../pyroelectric_coeff.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write(' echo "Converged in first attempt, moving on"\n')
        f.write('   python ../../pyroelectric_coeff.py --outcar\n')
        f.write('   echo "Launching calculations for elastic and dielectric tensors"\n')
        f.write('   python ../../pyroelectric_coeff.py --copy_file CONTCAR CONTCAR2\n')        
        f.write('   python ../../pyroelectric_coeff.py --move_file CONTCAR2 ../Elastic True\n')
        f.write('   python ../../pyroelectric_coeff.py --generate_batch Elastic batch.sh\n')
        f.write('   python ../../pyroelectric_coeff.py --move_file batch.sh ../Elastic False\n')
        f.write('   python ../../pyroelectric_coeff.py --launch_calc ../Elastic batch.sh\n')
        f.write(' else\n')
        f.write(' mv CONTCAR POSCAR\n')
        f.write('echo "Starting second relaxation."\n')
        f.write(' $MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write('  rm -f WAVECAR CHG\n')
        f.write('if [ `python ../../pyroelectric_coeff.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('  echo "Converged on second run, moving on"\n')
        f.write('   python ../../pyroelectric_coeff.py --outcar\n')
        f.write('   echo "Launching calculations for elastic and dielectric tensors"\n')
        f.write('   python ../../pyroelectric_coeff.py --copy_file CONTCAR CONTCAR2\n')
        f.write('   python ../../pyroelectric_coeff.py --move_file CONTCAR2 ../Elastic True\n')
        f.write('   python ../../pyroelectric_coeff.py --generate_batch Elastic batch.sh\n')
        f.write('   python ../../pyroelectric_coeff.py --move_file batch.sh ../Elastic False\n')
        f.write('   python ../../pyroelectric_coeff.py --launch_calc ../Elastic batch.sh\n')
        f.write('  else\n')
        f.write(' mv CONTCAR POSCAR\n')
        f.write('echo "Starting third relaxation."\n')        
        f.write('  $MPIEXEC_LOCAL $VASPLOC $option\n')
        f.write('  rm -f WAVECAR CHG\n')
        f.write('if [ `python ../../pyroelectric_coeff.py --vasp_converge Relaxation` == "True" ] ; then\n')
        f.write('   echo "Converged on third run, moving on"\n')
        f.write('   python ../../pyroelectric_coeff.py --outcar\n')
        f.write('   echo "Launching calculations for elastic and dielectric tensors"\n')
        f.write('   python ../../pyroelectric_coeff.py --copy_file CONTCAR CONTCAR2\n')
        f.write('   python ../../pyroelectric_coeff.py --move_file CONTCAR2 ../Elastic True\n')
        f.write('   python ../../pyroelectric_coeff.py --generate_batch Elastic batch.sh\n')
        f.write('   python ../../pyroelectric_coeff.py --move_file batch.sh ../Elastic False\n')
        f.write('   python ../../pyroelectric_coeff.py --launch_calc ../Elastic batch.sh\n')
        f.write('   else\n') 
        f.write('   echo "Run failed to converge after 3 attempts. Aborting"\n')
        f.write('   exit 1 \n')
        f.write('   fi\n')
        f.write('  fi\n')         
        f.write(' fi\n')
        f.write('fi\n')
        f.close()
        
    elif batchtype == 'Elastic':
        #print file
        f = open(batchname,'w')
        f.write('#!/bin/bash\n')
        f.write(batch_header_ela)
        f.write('\n')
        f.write(VASPSR)
        f.write('\n')
        f.write(PYTHMOD+'\n')
        f.write('## submit vasp job\n')
        f.write(' mv CONTCAR2 POSCAR\n')
        f.write('$MPIEXEC_LOCAL $VASPLOC $option\n')     
        f.write('\n')
        f.write('if [ `python ../../pyroelectric_coeff.py --vasp_converge Elastic` == "True" ] ; then\n')
        f.write('   echo "Elastic calculation converged. Moving on..."\n')
        f.write('   python ../../pyroelectric_coeff.py --outcar\n')  
        f.write('   python ../../pyroelectric_coeff.py --copy_file CONTCAR CONTCAR2\n')
        f.write('   python ../../pyroelectric_coeff.py --move_file CONTCAR2 ../TDEP/ True\n')
        f.write('   python ../../pyroelectric_coeff.py --generate_loto 1\n')
        f.write('   python ../../pyroelectric_coeff.py --move_file infile.lotosplitting ../TDEP True\n')
        f.write('   python ../../pyroelectric_coeff.py --calc_debye 1\n')
        f.write('   python ../../pyroelectric_coeff.py --generate_batch TDEP batch.sh\n')
        f.write('   python ../../pyroelectric_coeff.py --move_file batch.sh ../TDEP True\n')
        f.write('   python ../../pyroelectric_coeff.py --launch_calc ../TDEP batch.sh\n')
        f.write('   else\n')
        f.write('   echo "Calculation not converged. Aborting"\n')
        f.write('fi\n')
        f.close()
    
    elif batchtype == 'TDEP':
        #gather information from data_extract file
        debye = 0
        with open('../data_extraction','r') as datafile:
            for line in datafile:
                if 'Debye' in line:
                    l = line.split()
                    debye = l[1]
        
        #print file
        f = open(batchname,'w')
        f.write('#!/bin/bash\n')
        f.write(batch_header_tdep)
        f.write('\n')
        f.write(VASPSR)
        f.write('\n')
        f.write(PYTHMOD+'\n')
        f.write('#set up directories and files\n')
        f.write('initial_MD_dir=moldyn\n')
        f.write('config_dir=configs\n')
        f.write('# Default parameters:\n')
        f.write('natom='+natom_ss+'\n')
        f.write('n_configs='+n_configs+'\n')
        f.write('t_configs='+t_configs+'\n')
        f.write('debye=%s\n'%debye)
        f.write('\n')
        f.write('echo "Starting configuration creation "\n')
        f.write('\n')
        f.write('mkdir -p $initial_MD_dir\n')
        f.write('cp -up INCAR $initial_MD_dir/INCAR\n')
        f.write('cp -up CONTCAR2 $initial_MD_dir/infile.ucposcar\n')
        f.write('cp -up POTCAR $initial_MD_dir/\n')
        f.write('cp -up infile.lotosplitting $initial_MD_dir/\n')
        f.write('\n')
        f.write('cd  $initial_MD_dir/\n')
        f.write('# Create start structure for high accuracy calcs at finite temperature \n')
        f.write('\n')
        f.write('if [ ! -f "contcar_conf0001" ]; then\n')
        f.write('    echo "Run generate_structure" \n')
        f.write('    srun -n 1  '+TDEPSR+'generate_structure -na $natom\n')
        f.write('     ln -s outfile.ssposcar infile.ssposcar\n')
        f.write('     cp outfile.ssposcar POSCAR\n')
        f.write('    echo "Run canonical_configuration"\n')
        f.write('      srun -n 1 '+TDEPSR+'canonical_configuration -n $n_configs -t $t_configs -td $debye --quantum\n')
        f.write('  fi\n')
        f.write(' cp outfile.fakeforceconstant infile.forceconstant\n')
        f.write('\n')
        f.write('if [ ! -f "contcar_conf0001" ]; then\n')
        f.write('    echo "canonical_configuration failed; cannot find contcar_conf0001. Exiting."\n')
        f.write('    exit 1\n')
        f.write('fi\n')
        f.write('echo "Run file operations for configs first iteration"\n')
        f.write('mkdir -p ../$config_dir\n')
        f.write('mv -u contcar_* ../$config_dir\n')
        f.write('cp -up infile.ucposcar outfile.ssposcar infile.lotosplitting POTCAR ../$config_dir\n')
        f.write('cp -up INCAR ../$config_dir/INCAR\n')
        f.write('cd ../$config_dir\n')
        f.write('\n')
        f.write('echo "Current dir: " $PWD\n')
        f.write('if [ -f "outfile.free_energy" ] ; then\n')
        f.write('    rm -f contcar_*\n')
        f.write('    echo "Calculation already converged, continuing"\n')
        f.write('else\n')
        f.write('    g="contcar_"\n')
        f.write('    for f in contcar_conf*; do\n')
        f.write('        if [ -e "$f" ]; then\n')
        f.write('            echo $f\n')
        f.write('        else\n')
        f.write('            echo "configurations do not exist, exiting"; exit 1\n')
        f.write('        fi\n')
        f.write('        dir=${f#$g}\n')
        f.write('        mkdir -p $dir\n')
        f.write('        if [ -e "$dir/POSCAR" ]; then\n')
        f.write('            rm -f $f\n')
        f.write('            echo "$dir/POSCAR already exists, skipping"\n')
        f.write('        else\n')
        f.write('            mv -n $f $dir/POSCAR\n')
        f.write('        fi\n')
        f.write('    done\n')
        f.write('    ln -s outfile.ssposcar infile.ssposcar\n')
        f.write('\n')
        f.write('# Build VASP files\n')
        f.write('    echo "Build VASP configs "\n')
        f.write('    for d in conf*; do\n')
        f.write('    cd $d\n')
        f.write('    if [ -f "OUTCAR" ] ; then\n')
        f.write('       pythonresult=$(python ../../../../pyroelectric_coeff.py --vasp_converge TDEP)\n')
        f.write('    fi\n')
        f.write('    cd ..\n')
        f.write('    if [ -f "$d/OUTCAR" ] && [ "$pythonresult" == "True" ] ; then\n')
        f.write('            echo "Already converged in $d, moving on"\n')
        f.write('        else\n')
        f.write('            cp -up INCAR POTCAR $d\n')
        f.write('            cd $d\n')
        f.write('            #make KPOINT File\n')
        f.write('            python ../../../../pyroelectric_coeff.py --make_KPOINTS '+str(int(kdensity))+'\n')
        f.write('            python ../../../../pyroelectric_coeff.py --generate_batch Ground_state batch.sh \n')
        f.write('            #$MPIEXEC_LOCAL $VASPLOC $option\n') 
        f.write('            #rm -f CHG WAVECAR DOSCAR CHGCAR vasprun.xml\n')
        f.write('            cd ..\n')
        f.write('            python ../../../pyroelectric_coeff.py --launch_calc $d batch.sh \n')
        f.write('        fi\n')
        f.write('    done\n')
        f.write('\n')
        f.write('fi\n')
        f.write('\n')
        f.write('echo "Generatations of configurations complete. "\n')
        f.write('\n')
        f.close()
                
    elif batchtype == 'Ground_state':
        #print file
        f = open(batchname,'w')
        f.write('#!/bin/bash\n')
        f.write(batch_header_gs)
        f.write('\n')
        f.write(VASPSR)
        f.write('\n')
        f.write(PYTHMOD+'\n')
        f.write('## submit vasp job\n')
        f.write('$MPIEXEC_LOCAL $VASPLOC $option\n')     
        f.write('rm -f CHG WAVECAR DOSCAR CHGCAR vasprun.xml\n')
        f.close()
        
    elif batchtype == 'script':
        #print file
        f = open(batchname,'w')
        f.write(PYTHMOD+'\n')
        f.write('#loop through configuration files and run TDEP \n')
        f.write('echo "Analysing OUTCAR files"\n')
        f.write('python '+TDEPSR+'process_outcar_5.3.py */OUTCAR\n')
        f.write('echo "Extracting force constants (2nd and 3rd order)"\n')
        if withsolver == True:
            f.write(TDEPSR+'extract_forceconstants -rc2 '+rc_cut+' -rc3 '+rc3_cut+' --solver 2 --polar\n')
        else:
            f.write(TDEPSR+'extract_forceconstants -rc2 '+rc_cut+' -rc3 '+rc3_cut+' --polar\n')
        f.write('ln -s outfile.forceconstant infile.forceconstant\n')
        f.write('ln -s outfile.forceconstant_thirdorder infile.forceconstant_thirdorder\n')
        f.write('echo "Calculating phonon density of states"\n')
        f.write(TDEPSR+'phonon_dispersion_relations --dos --temperature_range '+tmin+' '+tmax+' '+tsteps+' -qg '+qgrid+' -it '+iter_type+' --unit icm --dumpgrid\n')    
        f.write('\n# Calculations complete. Now we should move files.\n')
        f.write(' python ../../../pyroelectric_coeff.py --move_file outfile.forceconstant ../../ False\n')
        f.write(' python ../../../pyroelectric_coeff.py --move_file outfile.forceconstant_thirdorder ../../ False\n')
        f.write(' python ../../../pyroelectric_coeff.py --move_file outfile.dispersion_relations.hdf5 ../../ False\n')
        f.write(' python ../../../pyroelectric_coeff.py --move_file outfile.grid_dispersions.hdf5 ../../ False\n')
        f.close()
        
    elif batchtype == 'outcar':
        #print file
        f = open(batchname,'w')
        f.write(PYTHMOD+'\n')
        f.write('   python ../../pyroelectric_coeff.py --outcar\n')
        f.close()
        
    elif batchtype == 'pyroelectric':
        f = open(batchname,'w')
        f.write('#!/bin/bash\n')
        f.write(batch_header_pyro)
        f.write('\n')
        if withntvol:
            f.write('   python ../pyroelectric_coeff.py --pyroelectric\n')
        else:
            f.write('   python ../pyroelectric_coeff.py --pyroelectric --ntvol\n')
        f.close()
        
        
    #make batch script executable
    bashsub = 'chmod +x '+batchname
    outp = bash_commands(bashsub)
    
    outp = outp #removes warning message
        
    return None

def calc_dudT(numatoms,massatom,coupling,cell_data):
    """
    Author: Nicholas Pike
    Email : Npike@ues.com
    
    Purpose: Calculates the derivative du/dT and stores it in a file
    
    Return: Array of dudT for each temperature, direction, atom, and phonon branch
    
    """
    #Define output variables
    Tstep     = int(cell_data[25])
    Tmin      = int(cell_data[23])
    Tmax      = int(cell_data[24])   
    omegazero = cell_data[14]
    omega     = cell_data[13]
    eigensz   = cell_data[18]
    eigvecre  = cell_data[15] 
    eigvecim  = cell_data[16] 
    dudT      = np.zeros(shape=(Tstep,3,numatoms,int(3.0*numatoms)))
    thermeig  = np.zeros(shape=(Tstep,3,numatoms,int(3.0*numatoms)))
               # indexed as temperature, directions, atoms, number of phonon modes
         
    # some basic printing so you know what is happening.
    print('   temp    natom    ')
    print('   -----------------')
    for t in range(Tstep):
        temp = Tmin+(Tmax-Tmin)/Tstep*t
        print('   %i' %temp)
        for n in range(numatoms):
            # loop over atoms
            mass = massatom[n]
            print('            %i ' %n)
            for lam in range(int(3*numatoms)):
                # loop over modes at q = 0
                omegaz = omegazero[lam]
                
                # eigenvectors of q=0 modes
                eigs0   = eigensz[lam,2*n+0]
                eigs1   = eigensz[lam,2*n+1]  
                eigs2   = eigensz[lam,2*n+2]
                
                # check some values that might cause an error message
                if omegaz < 1E-6:
                    preterm =  0
                else:                  
                    preterm = -1.0/(hbar*omegaz)*np.sqrt(hbar/(2.0*mass*omegaz))
                #now for the internal sum over q points and lambda prime!
                if temp == 0 or preterm == 0:
                    #find dudT
                    dudT[t][0][n][lam] = 0
                    dudT[t][1][n][lam] = 0
                    dudT[t][2][n][lam] = 0
                    
                else:
                    for qp in range(len(omega)):
#                        # loop over all q -points
#                        # eigenvectors of finite q modes
#                        eig0   = eigvecre[qp,lam,2*n+0]+1j*eigvecim[qp,lam,2*n+0]
#                        eig1   = eigvecre[qp,lam,2*n+1]+1j*eigvecim[qp,lam,2*n+1]
#                        eig2   = eigvecre[qp,lam,2*n+2]+1j*eigvecim[qp,lam,2*n+2]
#
#                        # find thermal-eigenvector
#                        omeg1 = omega[qp,lam]
#                        
#                        if omeg1 == 0.0:
#                            xterm1 = 0.0
#                            dbose1 = 0.0
#                        else:
#                            xterm1 = (hbar*omeg1)/(kb*temp)
#                            dbose1 = 1.0/(kb*temp**2)*(1.0/(np.cosh(xterm1)-1))
#                        
#                        # TODO check if the eigenvector should be real, imaginary, or squared!
#                        if np.abs(xterm1) < 710: 
#                            thermeig[t][0][n][lam] += np.real(eig0*dbose1)
#                            thermeig[t][1][n][lam] += np.real(eig1*dbose1)
#                            thermeig[t][2][n][lam] += np.real(eig2*dbose1)
#                        else:
#                            thermeig[t][0][n][lam] += 0.0
#                            thermeig[t][1][n][lam] += 0.0
#                            thermeig[t][2][n][lam] += 0.0
                            
                        for lamp in range(int(3*numatoms)):   
                            # loop over phonon modes for V3 summation
                            v3   = coupling[lam,lamp,qp] 
                            omeg = omega[qp,lamp] # number is an angular frequency (Hz)!
                            
                            # calculate the temperature dependence of Pi_1
                            if omeg == 0.0:
                                xterm  = 0.0
                                vdbose = 0.0
                            else:
                                xterm = (hbar*omeg)/(kb*temp)
                                if np.abs(xterm) < 710: # 710 will change depending on the machine
                                    vdbose = v3*xterm/temp*(1.0/(np.cosh(xterm)-1.0)) 
                                    #use cosh function for smoothness
                                                                                       
                            #find dudT
                            dudT[t][0][n][lam] += preterm*eigs0*vdbose
                            dudT[t][1][n][lam] += preterm*eigs1*vdbose
                            dudT[t][2][n][lam] += preterm*eigs2*vdbose
                            
    return dudT

def read_dudT(filename,cell_data):
    """
    Author: Nicholas Pike
    Email : Npike@ues.com
    
    Purpose: Read in a previously calculated dudT file (The full version!)
    
    Return: Returns the matrix dudT
    """
    Tstep      = int(cell_data[25])
    numatoms   = cell_data[4] 
  
    # create an empty array
    dudT =np.zeros(shape=(Tstep,3,numatoms,int(3.0*numatoms)))
    
    # fill the empty array by reading a file
    num = 0 
    with open(filename,'r') as f:
        while num < Tstep:
            for num,line in enumerate(f,0):
                if num % 50 == 0: 
                    print('          . ')
                if not line.startswith('#'):
                    l = line.strip('\n').split()      
                    # file is formatted such that
                    ##temp \tdudT(1,x,0) \tdudT(1,y,0) \tdudT(1,z,0) \tdudT(2,x,lambda = 1) \t etc. \n' 
                    #l[0] is skipped, we do not need temperature only the index
                    l = l[1:]
                    for lam in range(int(3.0*numatoms)):
                        for n in range(numatoms):
                            dudT[num-2][0][n][lam] = float(l[int(0+(numatoms-1)*n+3*numatoms*lam)])
                            dudT[num-2][1][n][lam] = float(l[int(1+(numatoms-1)*n+3*numatoms*lam)])
                            dudT[num-2][2][n][lam] = float(l[int(2+(numatoms-1)*n+3*numatoms*lam)])
                    
    return dudT

def main_pyro(DFT_INPUT,tags):
    """
    Author: Nicholas Pike
    Email:  Nicholas.pike@smn.uio.no
    
    Purpose:Calculation of the pyroelectric coefficient from first principles
    
    Return: None

    """    
    #unpack tags
    if tags != '':
        withntvol = tags[1]
    else:
        withntvol = False
        
    thermlat  = []
        
    print('*'*60)
    print('This program calculates the pyroelectric coefficient as a function\n'\
          'of temperature using information calculated from DFT, DFPT, and TDEP.\n')
    print('This program was written and designed by Nicholas Pike.\n'\
          'Please address all questions to Nicholas.pike@smn.uio.no\n\n')    
    # get data about computer system and number of cores
    print('*'*60)
    
        #start program by checking python version
    print('   1 - Checking current build information.')
    """
    Current build information is checked incase any part of the calculation needs
    to be run in parallel.  Currently, only the coupling constants are calculated 
    in parallel.  
    
    If these are already found in the directory, then this part takes almost no 
    additional time.
    """
    
    #gather information on the number of computer cores being used.
    max_core_number = os.environ.get('SLURM_NTASKS') #check if slurm is used first
    if max_core_number == None:
        max_core_number = cpu_count()
    else:
        max_core_number = int(os.environ.get('SLURM_NTASKS'))
        
    print('       Maximum number of cores: %i' %max_core_number)
    print('       Processor:               %s' %platform.processor())
    print('       System:                  %s' %platform.machine())
    print('       Python version:          %s' %platform.python_version())
    print('       Numpy version            %s' %np.version.version)
    print('       Largest number:          %s' %sys.float_info.max)
    print('       Smallest number:         %s' %sys.float_info.min)
    print('       Max exponent:            %s' %sys.float_info.max_exp)
    print('       Min exponent:            %s\n' %sys.float_info.min_exp)
    
    # get data from TDEP and DFT calculations
    cell_data = READ_INPUT_DFT(DFT_INPUT)  
    """
    Reading in DFT data (and data collected as part of ACTE.py's tdep 
    calculations) takes place here. Step 2 is internal to the previous module. 
    
    """                
    #store the max_core_number for later use
    cell_data[22] = max_core_number

    #extract data from cell_data
    #vol        = cell_data[0]    # DFT volume of unit cell at T= 0 K in m**3
    numatoms   = cell_data[4]    # number of atoms in the unit cell
    
    #determine mass of each atom type
    massatom = []
    for i in range(len(cell_data[5])):
        if i %2 == 1:
            massatom = np.append(massatom,[float(cell_data[5][i])])
            
    #Data from DFPT calculations
    Zstar  = cell_data[6]    # Born effective charge tensor in units of electron charge
    piezo  = cell_data[9]    # piezoelectric tensor
    
    print('   3 - Looking for thermal properties.')
    """
    This calculation requires two files from ACTE.py. 
    
    1) out.expansion_coeffs  -- file containing the non-symmetryic thermal 
        expansion coefficients for the system
    2) out.thermal_expansion  -- by default, the program uses the temperature
        dependent volume of the unit cell in its calculation of the pyroelectric
        coefficient.  Thus, we require the lattice parameters as  a function of
        temperature.
        
    If either of these files are missing, the program aborts. 
    """
    if os.path.isfile('out.expansion_coeffs'):
        print('       Found thermal expansion coefficients. Loading them now.')
        linexp        = read_expansion(cell_data)
        cell_data[10] = linexp
    else:
        print('ERROR: Need expansion coefficients to proceed.')
        print('       Run the ACTE.py program to generate out.expansion_coeffs .')
        sys.exit()
        
    if not withntvol:
        if os.path.isfile('out.thermal_expansion'):
            print('       Found thermal lattice parameters. Loading them now.')
            thermlat = read_thermal_lattice('out.thermal_expansion',cell_data)
        else:
            print('No thermal lattice parameters found in directory!.')
            print('    Run ACTE.py to generate out.thermal_expansion .')
            sys.exit()
    
    print('   4 - Calculation of anharmonic coupling constants.')
    """
    The program will check if the coupling constants have already been calculated 
    for this system.  If they are found, the program reads them in, if they are 
    absent. Then the program launches a parallel calculation to determine them.  
    
    For modest size systems, this calculation can take more than 24 hours with
    40 cores. 
    
    """
    #launch calculation of linear expansion coefficients
    coupling = calc_anharmonic(cell_data)    
    print('       Calculation of coupling constants is complete.')
                    
    """
    Begin calculation of the du/dt
    """
    print('   5 - Start calculation of du/dT')
    """
    With the calculated coupling constants, we can now determine dudT for all 
    atoms, phonon modes, and q points.  This calculation runs in series and takes 
    about 24 hours on a single core.  I am not sure yet if it can be parallelized.
    
    Before starting the calculation, it will check if it was previously calculated.
    """
    Tstep     = int(cell_data[25])
    Tmin      = int(cell_data[23])
    Tmax      = int(cell_data[24])   
    
    # check first if file already exists
    if os.path.isfile('out.dudT_full'):
        print('      Found previous calculation of dudT.')
        print('      Reading file now...')
        dudT = read_dudT('out.dudT_full',cell_data)
        
        print('      Finished read-in of full dudT file.')
    else:
        print('      Starting calculation...')
        dudT = calc_dudT(numatoms,massatom,coupling,cell_data)
        #dudT  -> shape=(Tstep,3,numatoms,int(3.0*numatoms))
        
        print('      Finished du/dT calculation.  Sum over lambda and writing files...')
    
        f1= open('out.dudT_full','w')
        f1.write('#duDT \n#temp \tdudT(1,x,0) \tdudT(1,y,0) \tdudT(1,z,0) \tdudT(2,x,lambda = 1) \t etc. \n' )
        for t in range(dudT.shape[0]):
            temp = Tmin+(Tmax-Tmin)/Tstep*t
            f1.write('%6.2f ' %(temp))
            for n in range(dudT.shape[2]):
                for lam in range(dudT.shape[3]):
                    f1.write('%1.10e %1.10e %1.10e ' %(dudT[t,0,n,lam],dudT[t,1,n,lam],dudT[t,2,n,lam]))
            f1.write('\n')
        f1.close()
        
        # need to sum over lambda first!
        dudTsum = np.zeros(shape=(Tstep,3,numatoms))
        for t in range(Tstep):
            for d in range(3):
                for n in range(numatoms):
                    for lam in range(int(3.0*numatoms)):
                        dudTsum[t,d,n] += dudT[t,d,n,lam]
                        
        f1= open('out.dudT','w')
        f1.write('#duDT \n#temp \tdudT(1,x) \tdudT(1,y) \tdudT(1,z) \tdudT(2,x) \t etc. \n' )
        for t in range(dudT.shape[0]):
            temp = Tmin+(Tmax-Tmin)/Tstep*t
            f1.write('%6.2f ' %(temp))
            for n in range(dudT.shape[2]):
                f1.write('%1.10e %1.10e %1.10e ' %(dudTsum[t,0,n],dudTsum[t,1,n],dudTsum[t,2,n]))
            f1.write('\n')
        f1.close()
         
    
    print('   6 - Starting calculation of the p1, p2, and p3')
    """
    Finally, the program will start the calculation of the pyroelectric coefficients
    
    By this point, the calculation is rather quick and should only take a few 
    more seconds to complete. 
    
    """
    #Define output variables
    ptot     = np.zeros(shape=(int(cell_data[25]),4))
    p1_sigma = np.zeros(shape=(int(cell_data[25]),3))  #array of p1 values
    p2_sigma = np.zeros(shape=(int(cell_data[25]),3))  #array of p2 values
    p3_sigma = np.zeros(shape=(int(cell_data[25]),3))  #array of p3 values
    # note, we only calculate the x,y,z values of each component.
    
    Tstep     = int(cell_data[25])                 
    for t in range(int(Tstep)):
        temp = Tmin+(Tmax-Tmin)/Tstep*t
        ptot[t,0] = temp  # store temperature
        
        if withntvol:
            vol = cell_data[0]  # DFT volume of unit cell at T= 0 K in m**3
        else:
            vol = thermlat[1,t] # This is a volume in meters cubed
            
        for d in range(3):
            for d2 in range(3):
                #calculate p2
                p2_sigma[t,d]  += piezo[d,d2]*linexp[d2+1,t] #here piezo is in C/m^2 and expansion coeff is per decree K
                
                for n in range(numatoms):
                    zatom = np.array([[Zstar[n,0],Zstar[n,1],Zstar[n,2]],
                              [Zstar[n,3],Zstar[n,4],Zstar[n,5]],
                              [Zstar[n,6],Zstar[n,7],Zstar[n,8]]])
                    #dzdu = np.array([dZdu[n][1]],[],[]) # 3 by 3 matrix of BEC derivatives
                    for lam in range(int(3.0*numatoms)):
                        #calculate p1 
                        p1_sigma[t,d] += echarge/vol*zatom[d,d2]*dudT[t][d2][n][lam]
                        
                        #calculate p3
                        #p3_sigma[t,d] += echarge*hbar**2/(2.0*vol)
                        
            # sum up contributions from d2, n, and lam to determine the total pyroelectric coeff.
            ptot[t,d+1] = p1_sigma[t,d]+p2_sigma[t,d] # + p3_sigma[t,d]
            print(t, ptot[t,d+1],p1_sigma[t,d],p2_sigma[t,d])
 

    #add print statement here to print the calculated data out to a file.
    f1= open('pyroelectric_coeff','w')
    f1.write('#pyroelectric coefficients \n#temp \tptx \tpty \tptz \tp1x \tp1y \tp1z \tp2x \tp2y \tp2z \tp3x \tp3y \tp3z\n' )
    for t in range(ptot.shape[0]):
        f1.write('%6.2f %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n'
                 %(ptot[t,0],ptot[t,1],ptot[t,2],ptot[t,3],
                   p1_sigma[t,0],p1_sigma[t,1],p1_sigma[t,2],
                   p2_sigma[t,0],p2_sigma[t,1],p2_sigma[t,2],
                   p3_sigma[t,0],p3_sigma[t,1],p3_sigma[t,2])) 
    f1.close()
        
    print('      Calculation completed! Thank you for using this program.')
    
    return None

def read_expansion(cell_data):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Read in the output file from ACTE for the calculated expansion coefficients
    
    Return: array of expansion coefficients as a function of temperature.
    """
    #get info from cell_data
    Tstep = int(cell_data[25])
    
    #declare arrays
    expan = np.zeros(shape=(10,Tstep))
    
    #open file
    c = 0
    with open('out.expansion_coeffs','r') as f:
        for line in f:
            if not line.startswith('#'):
                l = line.strip().split()
                expan[0][c] = float(l[0])
                expan[1][c] = float(l[1])
                expan[2][c] = float(l[2])
                expan[3][c] = float(l[3])
                expan[4][c] = float(l[4])
                expan[5][c] = float(l[5])
                expan[6][c] = float(l[6])
                expan[7][c] = float(l[7])
                expan[8][c] = float(l[8])
                expan[9][c] = float(l[9])
                c += 1
    return expan

def read_thermal_lattice(filename,cell_data):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Read in the output file from ACTE for the calculated temperature
             dependent lattice parameters
    
    Return: array of vol vs temperature
    """
    #get info from cell_data
    Tstep = int(cell_data[25])
    alpha = cell_data[27]*np.pi/180.
    beta  = cell_data[28]*np.pi/180.
    gamma = cell_data[29]*np.pi/180.
       
    #declare arrays
    expan = np.zeros(shape=(2,Tstep))
    
    #open file
    a = 0
    b = 0
    c = 0
    d = 0
    with open(filename,'r') as f:
        for line in f:
            if not line.startswith('#'):
                l = line.strip().split()
                if len(l) == 2:
                    #isotropic
                    expan[0][d] = float(l[0])
                    a = float(l[1])
                    b = a
                    c = a
                    expan[1][d] = a*b*c*np.sqrt((1.0-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2)+2.0*(np.cos(alpha)*np.cos(beta)*np.cos(gamma)))*ang_to_m**3.0
                elif len(l) == 3:
                    #two unique parameters
                    expan[0][d] = float(l[0])
                    a = float(l[1]) 
                    b = a
                    c = float(l[2]) 
                    expan[1][d] = a*b*c*np.sqrt((1.0-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2)+2.0*(np.cos(alpha)*np.cos(beta)*np.cos(gamma)))*ang_to_m**3.0
                elif len(l) == 4:
                    
                    expan[0][d] = float(l[0])
                    a = float(l[1])       
                    b = float(l[2])     
                    c = float(l[3]) 
                    expan[1][d] = a*b*c*np.sqrt((1.0-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2)+2.0*(np.cos(alpha)*np.cos(beta)*np.cos(gamma)))*ang_to_m**3.0
                d += 1
                
    return expan

def main_coupling(DFT_INPUT):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Precalculate the coupling constants and store information to a file
    
    Return: None.
    """
    if sys.version_info<=(3,0,0):
        print('ERROR: This program requires Python version 3.0.0 or greater.' )
        sys.exit()
        
    #gather information on the number of computer cores being used.
    max_core_number = os.environ.get('SLURM_NTASKS') #check if slurm is used first
    if max_core_number == None:
        max_core_number = cpu_count()
    else:
        max_core_number = int(os.environ.get('SLURM_NTASKS'))
        
    print('Maximum number of cores: %i' %max_core_number)
    print('Processor:               %s' %platform.processor())
    print('System:                  %s' %platform.machine())
    print('Python version:          %s' %platform.python_version())
    print('Numpy version            %s' %np.version.version)
    print('Largest number:          %s' %sys.float_info.max)
    print('Smallest number:         %s' %sys.float_info.min)
    print('Max exponent:            %s' %sys.float_info.max_exp)
    print('Min exponent:            %s\n' %sys.float_info.min_exp)
    
    # get data from TDEP and DFT calculations
    cell_data = READ_INPUT_DFT(DFT_INPUT)
    
    #store the max_core_number for later use
    cell_data[22] = max_core_number

    print('Launching calculation of anharmonic coupling constants')
    #launch calculation of linear expansion coefficients
    coupling = calc_anharmonic(cell_data)    
    print('Calculation of the anharmonic coupling constants is complete\n')
    
    #clears warning message
    coupling = coupling
    
    return None

def gather_outcar():
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Gather information from outcar files to be used in calculations of
             the pyroelectric coefficient.
             
    Return: None
    """
    #properties to get
    printfilename = '../data_extraction'
    alat = 0
    blat = 0
    clat = 0
    vol  = 0
    alpha = 0
    beta  = 0
    gamma = 0
    natom = 0
    dietensor  = np.zeros(shape=(3,3))
    pieztensor = np.zeros(shape=(3,6))
    elatensor  = np.zeros(shape=(6,6))
    bectensor  = []
    
    #check for existance of printfile and its contents
    printavec   = True
    printvol    = True
    printtensor = True
    printpos    = True
    printelas   = True
    
    try:
        with open(printfilename,'r') as p:
            for i, line in enumerate(p):
                if 'alat' in line:
                    printavec = False
                elif 'volume' in line:
                    printvol = False
                elif 'dielectric' in line:
                    printtensor = False
                elif 'atpos' in line:
                    printpos = False
                elif 'elastic' in line:
                    printelas = False

    except:
        print('Output file not found, generating one now...')
        
    # With file located, we will read the file and look for particular things based 
    # on our current directory.
    latvec  = 'length of vectors'
    dirvec  = 'direct lattice vectors'
    volcell = 'volume of cell'
    numions = 'number of ions'
    dietens = ' MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)'
    #need both ionic and displacement contributions to piezoelectric tensor
    pietens = ' PIEZOELECTRIC TENSOR  for field in x, y, z        (C/m^2)'
    pietens2 = 'PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        (C/m^2)'
    bectens = ' BORN EFFECTIVE CHARGES (in e, cummulative output)'
    elatens = ' SYMMETRIZED ELASTIC MODULI (kBar)'

    
    with open('OUTCAR','r') as f:
        for i, line in enumerate(f):
            # By default the last time the lattice parameters and volume are printed 
            # the data is kept
            
            #get lattice vectors
            if latvec in line :
                data = linecache.getline('OUTCAR',i+2).split()
                if data != []:
                    alat = float(data[0])
                    blat = float(data[1])
                    clat = float(data[2])
            
            #get lattice angles
            if dirvec in line :
                data1 = linecache.getline('OUTCAR',i+2).split()
                data2 = linecache.getline('OUTCAR',i+3).split() 
                data3 = linecache.getline('OUTCAR',i+4).split() 
                if data1 != []:
                    avec = np.array([float(data1[0]),float(data1[1]),float(data1[2])])
                    bvec = np.array([float(data2[0]),float(data2[1]),float(data2[2])])
                    cvec = np.array([float(data3[0]),float(data3[1]),float(data3[2])])
                    
                    alpha = np.arccos(np.dot(bvec,cvec)/(np.linalg.norm(bvec)*np.linalg.norm(cvec)))*180.0/np.pi
                    beta  = np.arccos(np.dot(avec,cvec)/(np.linalg.norm(avec)*np.linalg.norm(cvec)))*180.0/np.pi
                    gamma = np.arccos(np.dot(avec,bvec)/(np.linalg.norm(avec)*np.linalg.norm(bvec)))*180.0/np.pi            

            #get unit cell volume
            elif volcell in line and printvol:
                data = linecache.getline('OUTCAR',i+1).split()
                vol  = float(data[len(data)-1])
        
            #get number of ions in the unit cell
            elif numions in line and printtensor:
                data = linecache.getline('OUTCAR',i+1).split()
                natom = int(data[11])
                
            #get dielectric tensor
            elif dietens in line and printtensor: 
                data1 = linecache.getline('OUTCAR',i+3).split()
                data2 = linecache.getline('OUTCAR',i+4).split()
                data3 = linecache.getline('OUTCAR',i+5).split()
                dietensor[0][0] = float(data1[0])
                dietensor[0][1] = float(data1[1])
                dietensor[0][2] = float(data1[2])
                dietensor[1][0] = float(data2[0])
                dietensor[1][1] = float(data2[1])
                dietensor[1][2] = float(data2[2])
                dietensor[2][0] = float(data3[0])
                dietensor[2][1] = float(data3[1])
                dietensor[2][2] = float(data3[2])
    
            #get piezoelectric tensor
            elif pietens in line and printtensor:
                data1 = linecache.getline('OUTCAR',i+4).split()
                data2 = linecache.getline('OUTCAR',i+5).split()
                data3 = linecache.getline('OUTCAR',i+6).split()
                pieztensor[0][0] += float(data1[1])
                pieztensor[0][1] += float(data1[2])
                pieztensor[0][2] += float(data1[3])
                pieztensor[0][3] += float(data1[4])
                pieztensor[0][4] += float(data1[5])
                pieztensor[0][5] += float(data1[6])
                pieztensor[1][0] += float(data2[1])
                pieztensor[1][1] += float(data2[2])
                pieztensor[1][2] += float(data2[3])
                pieztensor[1][3] += float(data2[4])
                pieztensor[1][4] += float(data2[5])
                pieztensor[1][5] += float(data2[6])
                pieztensor[2][0] += float(data3[1])
                pieztensor[2][1] += float(data3[2])
                pieztensor[2][2] += float(data3[3])
                pieztensor[2][3] += float(data3[4])
                pieztensor[2][4] += float(data3[5])
                pieztensor[2][5] += float(data3[6])
                
            elif pietens2 in line and printtensor:
                data1 = linecache.getline('OUTCAR',i+4).split()
                data2 = linecache.getline('OUTCAR',i+5).split()
                data3 = linecache.getline('OUTCAR',i+6).split()
                pieztensor[0][0] += float(data1[1])
                pieztensor[0][1] += float(data1[2])
                pieztensor[0][2] += float(data1[3])
                pieztensor[0][3] += float(data1[4])
                pieztensor[0][4] += float(data1[5])
                pieztensor[0][5] += float(data1[6])
                pieztensor[1][0] += float(data2[1])
                pieztensor[1][1] += float(data2[2])
                pieztensor[1][2] += float(data2[3])
                pieztensor[1][3] += float(data2[4])
                pieztensor[1][4] += float(data2[5])
                pieztensor[1][5] += float(data2[6])
                pieztensor[2][0] += float(data3[1])
                pieztensor[2][1] += float(data3[2])
                pieztensor[2][2] += float(data3[3])
                pieztensor[2][3] += float(data3[4])
                pieztensor[2][4] += float(data3[5])
                pieztensor[2][5] += float(data3[6])
                
            #get born effective charge tensor
            elif bectens in line and printtensor:
                bectensor = np.zeros(shape=(natom,3,3))
                for j in range(natom):
                    data1 = linecache.getline('OUTCAR',i+4+j*4).split()
                    data2 = linecache.getline('OUTCAR',i+5+j*4).split()
                    data3 = linecache.getline('OUTCAR',i+6+j*4).split()
                    bectensor[j][0][0] = float(data1[1])
                    bectensor[j][0][1] = float(data1[2])
                    bectensor[j][0][2] = float(data1[3])
                    bectensor[j][1][0] = float(data2[1])
                    bectensor[j][1][1] = float(data2[2])
                    bectensor[j][1][2] = float(data2[3])
                    bectensor[j][2][0] = float(data3[1])
                    bectensor[j][2][1] = float(data3[2])
                    bectensor[j][2][2] = float(data3[3])    
                    
            #get elastic tensor
            elif elatens in line and printelas:
                data1 = linecache.getline('OUTCAR',i+4).split()
                data2 = linecache.getline('OUTCAR',i+5).split()
                data3 = linecache.getline('OUTCAR',i+6).split()
                data4 = linecache.getline('OUTCAR',i+7).split()
                data5 = linecache.getline('OUTCAR',i+8).split()
                data6 = linecache.getline('OUTCAR',i+9).split()
                if not float(data1[1]) >= 0.0 and float(data2[2]) >= 0.0 and float(data3[3]) >= 0.0:
                    #checks the first three diagonal components... that should be enough
                    print('ERROR: The elastic tensor contains negative diagonal elements!')
                    print('       Diagonal elements are...')
                    print('       %s' %float(data1[1]))
                    print('       %s' %float(data2[2]))
                    print('       %s' %float(data3[3]))
                    print('       %s' %float(data4[4]))
                    print('       %s' %float(data5[5]))
                    print('       %s' %float(data6[6]))
                    print('       Aborting calculation!')
                    print('Suggestion: Check the elastic tensor calculation since your material us not stable.')
                    sys.exit()    
                elatensor[0][0] = float(data1[1])
                elatensor[0][1] = float(data1[2])
                elatensor[0][2] = float(data1[3])
                elatensor[0][3] = float(data1[4])
                elatensor[0][4] = float(data1[5])
                elatensor[0][5] = float(data1[6])
                elatensor[1][0] = float(data2[1])
                elatensor[1][1] = float(data2[2])
                elatensor[1][2] = float(data2[3])
                elatensor[1][3] = float(data2[4])
                elatensor[1][4] = float(data2[5])
                elatensor[1][5] = float(data2[6])
                elatensor[2][0] = float(data3[1])
                elatensor[2][1] = float(data3[2])
                elatensor[2][2] = float(data3[3])
                elatensor[2][3] = float(data3[4])
                elatensor[2][4] = float(data3[5])
                elatensor[2][5] = float(data3[6])
                elatensor[3][0] = float(data4[1])
                elatensor[3][1] = float(data4[2])
                elatensor[3][2] = float(data4[3])
                elatensor[3][3] = float(data4[4])
                elatensor[3][4] = float(data4[5])
                elatensor[3][5] = float(data4[6])
                elatensor[4][0] = float(data5[1])
                elatensor[4][1] = float(data5[2])
                elatensor[4][2] = float(data5[3])
                elatensor[4][3] = float(data5[4])
                elatensor[4][4] = float(data5[5])
                elatensor[4][5] = float(data5[6])
                elatensor[5][0] = float(data6[1])
                elatensor[5][1] = float(data6[2])
                elatensor[5][2] = float(data6[3])
                elatensor[5][3] = float(data6[4])
                elatensor[5][4] = float(data6[5])
                elatensor[5][5] = float(data6[6])                                     
                    
            elif printpos == True:
                with open('POSCAR','r') as f:
                    for i, line in enumerate(f):
                        if i == 5 :
                            atnames = line.split()
                        elif i == 6:
                            atmult = line.split()   
                            
    """
    Now start print out of information to a seperate file. What is printed is 
    determined by the input file and what is already in the seperate file.
    """
                
    #open file
    f1= open(printfilename,'a')
    
    #print data to the output file (if it doesn't already exist)
    if alat != 0 and printavec:
        f1.write('#Calculated data from VASP to TDEP\n')
        f1.write('alat %f\n'%alat)
        f1.write('blat %f\n'%blat)
        f1.write('clat %f\n'%clat)
        f1.write('alpha %f\n'%alpha)
        f1.write('beta  %f\n'%beta)
        f1.write('gamma %f\n'%gamma)
    if vol != 0 and printvol:
        f1.write('volume %f\n'%vol)
        f1.write('TMIN %s\n' %tmin)
        f1.write('TMAX %s\n' %tmax)
        f1.write('TSTEP %s\n' %tsteps)
    if printpos:
        atnamestring = ''
        for i in range(len(atnames)):
            for j in range(int(atmult[i])):
                atnamestring += atnames[i]+ ' ' 
        f1.write('atpos %s\n'%atnamestring)
    if natom != 0 and bectensor[0][0][0] != 0 and printtensor:
        f1.write('natom %i\n'%natom)    
        f1.write('dielectric %f %f %f %f %f %f %f %f %f\n'%(dietensor[0][0],dietensor[0][1],dietensor[0][2],
                                    dietensor[1][0],dietensor[1][1],dietensor[1][2],
                                    dietensor[2][0],dietensor[2][1],dietensor[2][2]))
        f1.write('piezo %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n'
                         %(pieztensor[0][0],pieztensor[0][1],pieztensor[0][2],pieztensor[0][3],pieztensor[0][4],pieztensor[0][5],
                           pieztensor[1][0],pieztensor[1][1],pieztensor[1][2],pieztensor[1][3],pieztensor[1][4],pieztensor[1][5],
                           pieztensor[2][0],pieztensor[2][1],pieztensor[2][2],pieztensor[2][3],pieztensor[2][4],pieztensor[2][5]))
        for i in range(natom):
            if i == 0:
                f1.write('bectensor %i %f %f %f %f %f %f %f %f %f\n'%(i+1,bectensor[i][0][0],bectensor[i][0][1],bectensor[i][0][2],
                                      bectensor[i][1][0],bectensor[i][1][1],bectensor[i][1][2],
                                      bectensor[i][2][0],bectensor[i][2][1],bectensor[i][2][2]))  
            else:
                f1.write('          %i %f %f %f %f %f %f %f %f %f\n'%(i+1,bectensor[i][0][0],bectensor[i][0][1],bectensor[i][0][2],
                                      bectensor[i][1][0],bectensor[i][1][1],bectensor[i][1][2],
                                      bectensor[i][2][0],bectensor[i][2][1],bectensor[i][2][2]))
                
        f1.write('elastic %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n'
                 %(elatensor[0][0],elatensor[0][1],elatensor[0][2],elatensor[0][3],elatensor[0][4],elatensor[0][5],
                   elatensor[1][0],elatensor[1][1],elatensor[1][2],elatensor[1][3],elatensor[1][4],elatensor[1][5],
                   elatensor[2][0],elatensor[2][1],elatensor[2][2],elatensor[2][3],elatensor[2][4],elatensor[2][5],
                   elatensor[3][0],elatensor[3][1],elatensor[3][2],elatensor[3][3],elatensor[3][4],elatensor[3][5],
                   elatensor[4][0],elatensor[4][1],elatensor[4][2],elatensor[4][3],elatensor[4][4],elatensor[4][5],
                   elatensor[5][0],elatensor[5][1],elatensor[5][2],elatensor[5][3],elatensor[5][4],elatensor[5][5]))
     
    #close output filein
    f1.close()
    return None

def gather_files(infilename):
    """
    Author: Nicholas Pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Gather and move files to the root directory of calculation
    
    """       
    printfilename = '../data_extraction'
                       
    if infilename.endswith('dispersion_relations.hdf5'):
        dispfile = 'outfile.dispersion_relations.hdf5'     
        infiledir = infilename.split('/')
        infilestr = ''
        for i in range(len(infiledir)-1):
            infilestr += infiledir[i]+'/'
        strext = 'disp_data.hdf5'
    
        if os.path.isfile(infilestr+'/'+dispfile):
            #move file if it exists
            bashcommand = 'cp '+infilestr+'/'+dispfile+' '+strext
            output = subprocess.Popen([bashcommand],stdout=subprocess.PIPE,shell=True).communicate()[0]
        
        #print file name to directory
            f1 = open(printfilename,'a')
            f1.write('DISPDATA %s\n' %strext)
            f1.close()
                
        elif infilename.endswith('grid_dispersions.hdf5'):
    
            phonfile = 'outfile.grid_dispersions.hdf5'
            infiledir = infilename.split('/')
            infilestr = ''
            for i in range(len(infiledir)-1):
                infilestr += infiledir[i]+'/'
            strext = 'grid_data.hdf5'
            
            if os.path.isfile(infilestr+'/'+phonfile):
                #move file if it exists
                bashcommand = 'cp '+infilestr+'/'+phonfile+' '+strext
                output = subprocess.Popen([bashcommand],stdout=subprocess.PIPE,shell=True).communicate()[0]
                
            #print file name to directory
                f1 = open(printfilename,'a')
                f1.write('PHONDATA %s\n' %strext)
                f1.close()
                
    elif infilename.endswith('free_energy'):
        freefile = 'outfile.free_energy'
        infiledir = infilename.split('/')
        infilestr = ''
        for i in range(len(infiledir)-1):
            infilestr += infiledir[i]+'/'
        strext    = 'free_energy'
    
        if os.path.isfile(infilestr+'/'+freefile):
            #move file if it exists
            bashcommand = 'cp '+infilestr+'/'+freefile+' '+strext
            output = subprocess.Popen([bashcommand],stdout=subprocess.PIPE,shell=True).communicate()[0]
     
        #print file name to directory
            f1 = open(printfilename,'a')
            f1.write('CVDATA %s\n' %strext)
            f1.close() 
            
        batchfile = 'batch_latt.sh'
        with open(batchfile,'r') as f:
            for i, line in enumerate(f):
                 if 'tmin' in line :
                     data = linecache.getline(batchfile,i+1).split('=')
                     tmin = float(data[1])
                 elif 'tmax' in line:
                     data = linecache.getline(batchfile,i+1).split('=')
                     tmax = float(data[1])
                 elif 'tstep' in line :
                     data  = linecache.getline(batchfile,i+1).split('=')
                     tstep = float(data[1])
                     
        #print file name to directory
        f1 = open(printfilename,'a')
        f1.write('TMIN %s\n' %tmin)
        f1.write('TMAX %s\n' %tmax)
        f1.write('TSTEP %s\n' %tstep)

        f1.close() 
            
                
    elif infilename.endswith('third_order'):
        thirdfile = 'outfile.forceconstant_thirdorder '
        infiledir = infilename.split('/')
        infilestr = ''
        for i in range(len(infiledir)-1):
            infilestr += infiledir[i]+'/'
        strext    = 'third_order'
    
        if os.path.isfile(infilestr+'/'+thirdfile):
            #move file if it exists
            bashcommand = 'cp '+infilestr+'/'+thirdfile+' '+strext
            output = subprocess.Popen([bashcommand],stdout=subprocess.PIPE,shell=True).communicate()[0]
                 
        #print file name to directory
            f1 = open(printfilename,'a')
            f1.write('TRIDATA %s\n' %strext)
            f1.close()     
            
    output = output #does nothing
    return None

def determine_volumes():
    """
    Author: Nicholas Pike 
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Determines the number of volumes by reading the poscar file.
    
    Return: number of volumes and the cell identification number
    """
    POSCAR_store = []
    with open('Relaxation/CONTCAR','r') as POSCAR:
       for line in POSCAR:
           POSCAR_store = np.append(POSCAR_store,line)  
        
    #extract lengths of a,b and ,c lattice parameter
    avec = POSCAR_store[2].split()
    bvec = POSCAR_store[3].split()
    cvec = POSCAR_store[4].split()
    
    #find the magnitude of the lattice vectors
    a = np.linalg.norm(avec)
    b = np.linalg.norm(bvec)
    c = np.linalg.norm(cvec)
            
    if ( np.abs(a-b)<=difftol and np.abs(b -c) <= difftol 
        and np.abs(a-c)<=difftol  ):
       #isotropic. 6 different volumes, ax, by, and cz are the only non-zero lattice parameters
       diff_volumes = 6
       speccell = 0
       
    elif ( np.abs(a-b)<= difftol and np.abs(a-c) > difftol and np.abs(b-c) > difftol ):
       #a and b are the same, c is different, 36 volumes 
       diff_volumes = 36
       speccell = 0
       
    elif ( np.abs(a-b)> difftol and np.abs(a-c) <= difftol and np.abs(b-c) > difftol ):
       #a and c are the same, b is different, 36 volumes 
       diff_volumes = 36
       speccell = 1
       
    elif ( np.abs(a-b)> difftol and np.abs(b -c)<= difftol and np.abs(a-c) > difftol ):
       #b and c are the same, a is different, 36 volumes , ax, by, and cz are the only non-zero lattice parameters
       diff_volumes = 36
       speccell = 2
       
    elif a != b and b != c:
       #all vectors are different, 216 volumes and , ax, by, and cz are the only non-zero lattice parameters
       diff_volumes = 216
       speccell = 0
       
    else:
       print('Something bad happened, or differnet symmetry read in, when reading the POSCAR file')
       sys.exit()
       
    return diff_volumes,speccell

def sortflags():
    """
    Author: Nicholas pike
    Email: Nicholas.pike@smn.uio.no
    
    Purpose: Determine the type of executation for this program
    """    
    if len(sys.argv) == 1:
        #check that the user entered a tag
        print('Use python pyroelectric_coeff.py --help to view the help menu.')
        sys.exit()
    else:
        i = 1
        while i < len(sys.argv):
            tagfound = False
            if sys.argv[i] == '--help':
                tagfound = True
                print('--help\t\t Prints this help menu.\n')
                print('--usage\t\t Gives an example of how to use this program.')
                print('--author\t Gives author information.')
                print('--version\t Gives the version of this program.')
                print('--email\t\t Provides an email address of the primary author.')
                print('--bug\t\t Provides an email address for bug reports.\n')
                print('## Executation of main program ##\n')
                print('--start_calc\t Starts calculation of pyroelectric coefficient.')
                print('--post_process \t Processes the vasp files with tdep.')
                print('--launch_pyro \t Calculates the pyroelectric coefficient.\n')
                print('## Internal calls and routines ##\n')
                print('--coupling \t Calculates the third order coupling for each q and phonon branch.')        
                print('--outcar \t Gathers data from outcar file.')
                print('--pyroelectric \t Launches calculation of the pyroelectric coefficient.')
                print('--move_file \t moves a file from one directory to another.')
                print('--copy_file \t copies a file from one directory to another.')
                print('--make_KPOINTS \t generates the KPOINT file based on user input.')
                print('--launch_calc \t launches calculations from a batch script.')
                print('--vasp_converge  determines if a VASP calculation converged.')
                print('--generate_batch generates a batch file based on user input.')
                print('--gather_file\t moves a file from one directory to another. Different than move_file.')
                print('--calc_debye \t calculates the Debye temperature from the elastic tensor.')
                print('--generate_loto writes the TDEP formatted loto splitting file.')
                i+=1
                sys.exit() 
            
            elif sys.argv[i] == '--usage':
                tagfound = True
                print('--usage\t To use this program use the following in the command line.\n python pyroelectric_coeff.py --TAG data_extraction_file')
                i+=1
                sys.exit()
        
            elif sys.argv[i] == '--author':
                tagfound = True
                print('--author\t The author of this program is %s' %__author__)
                i+=1
                sys.exit()
        
            elif sys.argv[i] == '--version':
                tagfound = True
                print('--version\t This version of the program is %s \n\n  This version allows the user'
                      ' to read in information gathered with a\n different python script and calculate\n '
                      ' the pyroelectric coefficients from a first principles calculation.' %__version__)
                i+=1
                sys.exit()
        
            elif sys.argv[i] == '--email':
                tagfound = True
                print('--email\t Please send questions and comments to %s' %__email__)
                i+=1
                sys.exit()
        
            elif sys.argv[i] == '--bug':
                tagfound = True
                print('--bug \t Please send bug reports to %s' %__email__)
                i+=1
                sys.exit()    
                
            elif sys.argv[i] == '--start_calc':
                #launch relaxation, tensor calculations, band structure, and tdep calculations
                print('Launching the calculations for the pyroelectric coefficient.')
                makerelax = False
                makeelastic = False
                print('   First, check if there are previous calculations')
                if os.path.exists('data_extraction'):
                    with open('data_extraction','r') as f:
                        for i,line in enumerate(f):
                            if 'volume' in line:
                                makerelax = False
                            if 'elastic' in line:
                                makeelastic = False
                
                    if makerelax == False and makeelastic == False:
                        copy_file('data_extraction','data_extraction_acte') #move the existing data extraction file
                        remove_file('data_extraction')
                        print('      Rebuilding of the data_extraction file required!')
                        
                else:
                    makerelax = True
                    makeelastic = True
                
                #decide what to do if the files are found.
                if makerelax == True:
                    #since both are true, we need to start the calculation from scratch
                    make_folder('Relaxation')
                    make_folder('Elastic')
                    make_folder('TDEP')
                
                    #generate batch files
                    generate_batch('Relaxation','batch.sh','')
                    move_file('batch.sh','Relaxation',original = False)
                                
                    #INCAR
                    generate_INCAR('Relaxation')
                    move_file('INCAR','Relaxation',original = False)
                    generate_INCAR('Elastic')
                    move_file('INCAR','Elastic',original = False)
                    generate_INCAR('TDEP')
                    move_file('INCAR','TDEP',original = False)
                
                    #KPOINTS
                    generate_KPOINT(kdensity)
                    move_file('KPOINTS','Relaxation',original = True)
                    move_file('KPOINTS','Elastic',original = True)
                    move_file('KPOINTS','TDEP',original = True)
                
                    #POTCAR
                    generate_POTCAR()
                    move_file('POTCAR','Relaxation',original = True)
                    move_file('POTCAR','Elastic',original = True)
                    move_file('POTCAR','TDEP',original = True)
                
                    #POSCAR moved to relaxation first
                    move_file('POSCAR','Relaxation',original = True)
                    
                    #submit relaxation calculation
                    launch_calc('Relaxation','batch.sh')
                    #Note, that relaxation calculation will launch the rest of the calculations
                    #if everything is converged.
                    
                elif makerelax == False and makeelastic == False:
                    print('      Calculations completed for the relaxation and elastic steps... Moving on')
                    #need to generate new data_extraction file
                    print('      Rebuilding the data_extraction file')
                    generate_batch('outcar','gather_outcar.sh','')
                    move_file('gather_outcar.sh','Relaxation',original = True)
                    move_file('gather_outcar.sh','Elastic',original= False)
                    launch_script('Relaxation','gather_outcar.sh')
                    launch_script('Elastic','gather_outcar.sh')
                    #check if TDEP directory exists
                    if os.path.isdir('TDEP'):
                        print('      Calculation of TDEP properties found as well!')
                        sys.exit()
                    else:
                        print('      Calculations for TDEP properties not found, looking for configuration calculations...')
                        if os.path.isdir('lattice_0'):
                            print('      Found configuration calculations. Using the equilibrium configuration for tdep.')
                            sys.exit()
                            
                        else:
                            print('      Configurations not found, launching TDEP calculation.')
                            make_folder('TDEP')
                            
                            #generate batch files in the relaxation folder, code checks for completion anyway
                            generate_batch('Relaxation','batch.sh','')
                            move_file('batch.sh','Relaxation',original = False)
                            
                            #generate incar file
                            generate_INCAR('TDEP')
                            move_file('INCAR','TDEP',original = False)
                            
                            #generate kpoint file
                            generate_KPOINT(kdensity)
                            move_file('KPOINTS','TDEP',original = True)
                            
                            #generate potcar file
                            move_file('POTCAR','TDEP',original = True)
                            
                            #launch calculation
                            launch_calc('Relaxation','batch.sh')
                            sys.exit()
                    
                else:
                    print('Something bad happened in sortflag --start_calc contact developer.')
                    print('makerelax: %s' %makerelax)
                    print('makeelastic: %s' %makeelastic)
                    sys.exit()
                    
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--post_process':
                tagfound   = True
                withsolver = False
                withntvol   = False
                ii= i+1
                while ii < len(sys.argv):
                    if sys.argv[ii] == '--solver':
                        withsolver = True
                        ii +=1
                    else:
                        print('%s is not a valid tag. See help menu.' %sys.argv[ii])
                        sys.exit()
                tags = [withsolver,withntvol]

                if os.path.isdir('TDEP'):
                    generate_batch('script','gather.sh',tags)
                    move_file('gather.sh','TDEP/configs',original = False)
                    print('Launching TDEP post-processing')
                    launch_script('TDEP/configs/','gather.sh')
                    
                elif os.path.isdir('lattice_0'):
                    a0 = 0
                    b0 = 0
                    c0 = 0
                    minlatticefile = ''
                    diff_volumes,speccell = determine_volumes()
                    with open('data_extraction','r') as f:
                        for line in f:
                            if 'alat'  in line:
                                l = line.split()
                                a0 = float(l[1])
                            elif 'blat' in line:
                                l = line.split()
                                b0 = float(l[1])
                            elif 'clat' in line:
                                l = line.split()
                                c0 = float(l[1])
                    for i in range(diff_volumes):
                        POSCAR = open('lattice_'+str(i)+'/POSCAR','r')
                        POSCAR.readline() #name 
                        scale = float(POSCAR.readline().strip().split()[0])
                        avec  = POSCAR.readline().strip().split()
                        bvec  = POSCAR.readline().strip().split()
                        cvec  = POSCAR.readline().strip().split()
                        POSCAR.close()
                        
                        #find norm of vectors
                        a = find_norm(avec,scale)
                        b = find_norm(bvec,scale)
                        c = find_norm(cvec,scale)
                        if np.abs(a-a0) < difftol and np.abs(b-b0) < difftol and np.abs(c-c0) < difftol:
                            minlatticefile = 'lattice_'+str(i)
                            
                    if os.path.isdir(minlatticefile):
                        generate_batch('script','gather.sh','')
                        move_file('gather.sh',minlatticefile+'/configs',original = False)
                        print('Launching TDEP post-processing')
                        launch_script(minlatticefile+'/configs','gather.sh')
                    else:
                        print('The directory corresponding to the minimum lattice parameters is not found! Contact developer')
                        sys.exit()
 
                i += 1
                sys.exit()
                
            elif sys.argv[i] == '--launch_pyro':
                tagfound = True
                withntvol = False
                ii= i+1
                while ii < len(sys.argv):
                    if sys.argv[ii] == '--ntvol':
                        withntvol = True
                        ii +=1
                    else:
                        print('%s is not a valid tag. See help menu.' %sys.argv[ii])
                        sys.exit()
                tags = ['',withntvol]
                
                generate_batch('pyroelectric','batch_pyro.sh',tags)
                launch_calc_root('batch_pyro.sh')
                i += 1
                sys.exit()
                
            elif sys.argv[i] == '--pyroelectric':
                tagfound = True      
                withntvol = False
                ii= i+1
                while ii < len(sys.argv):
                    if sys.argv[ii] == '--ntvol':
                        withntvol = True
                        ii +=1
                    else:
                        print('%s is not a valid tag. See help menu.' %sys.argv[ii])
                        sys.exit()
                tags = ['',withntvol]
                main_pyro('data_extraction',tags)
                i+=1
                sys.exit()
            
            elif sys.argv[i] == '--coupling':
                tagfound = True
                DFT_INPUT = sys.argv[2]
                main_coupling(DFT_INPUT)
                i+=1
                sys.exit()  
    
            elif sys.argv[i] == '--outcar':
                tagfound = True
                gather_outcar()
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--gather_files':
                tagfound = True
                gather_files(sys.argv[i+1])
                i+=1
                sys.exit()
            
            elif sys.argv[i] == '--copy_file':
                tagfound = True
                copy_file(sys.argv[i+1],sys.argv[i+2])
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--move_file':
                tagfound = True
                move_file(sys.argv[i+1],sys.argv[i+2],sys.argv[i+3])
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--generate_batch':
                tagfound = True
                generate_batch(sys.argv[i+1],sys.argv[i+2],'')
                i+=1
                sys.exit()
            
            elif sys.argv[i] == '--vasp_converge':
                tagfound = True
                vasp_converge(sys.argv[i+1])
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--make_KPOINTS':
                tagfound = True
                generate_KPOINT(float(sys.argv[i+1]))
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--launch_calc':
                tagfound = True
                launch_calc(sys.argv[i+1],sys.argv[i+2])
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--generate_loto':
                tagfound = True
                generate_LOTO()
                i+=1
                sys.exit()
                
            elif sys.argv[i] == '--calc_debye':
                tagfound = True
                find_debye()
                i+=1
                sys.exit()
            
            if tagfound != True:
                print('ERROR: The tag %s is not reconized by the program.  Please use --help \n to see available tags.' %sys.argv[i])
                sys.exit()    
        i += 1
        
    return None

"""
Run main program for windows machine, others will work automatically
"""
#starts main program for windows machines... has no effect for other machine types
if __name__ == '__main__':
    __author__     = 'Nicholas Pike'
    __copyright__  = 'none'
    __credits__    = 'none'
    __license__    = 'none'
    __version__    = '0.0'
    __maintainer__ = 'Nicholas Pike'
    __email__      = 'Nicholas.pike@sintef.no'
    __status__     = 'experimental'
    __date__       = 'October 2017'
    
    sortflags()
    
"""
Ends program

"""
