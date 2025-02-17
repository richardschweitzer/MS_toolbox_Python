#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translation of the Microsaccade Toolbox 0.9 from R to Python (original authors 
Ralf Engbert, Petra Sinn, Konstantin Mergenthaler, and Hans Trukenbrod) by 
Richard Schweitzer. 

vecvel(): 2d smooth velocity space
microsacc(): saccade detection 
binsacc(): checks whether saccade timings match
microsacc_merge(): custom algorithm to merge extremely close saccade candidates
                    (likely post-saccadic oscillations)


Created on Thu Aug  1 17:55:38 2024

@author: richard schweitzer
"""

import numpy as np



def vecvel(x, y, SAMPLING_RATE=1000, FIVE_POINT_SMOOTH=True):
    """
    Computes a 2D velocity vector. Code is translated from the original R function 
    by Ralf Engbert, Petra Sinn, Konstantin Mergenthaler, and Hans Trukenbrod
    
    Output: two-column (x,y) velocity vector
    """
    # create a Nx2 matrix to match the existing code
    xy = np.matrix(np.vstack((np.array(x), np.array(y)))) 
    xy = np.matrix.transpose(xy) 
    # determine size and preallocate the velocity vector
    d = np.shape(xy)
    N = d[0] # Python, unlike R, starts at 0 as first vector index - this is fixed below
    v = np.matrix(np.zeros(d))
    # compute velocity
    if FIVE_POINT_SMOOTH==True:
        v[2:(N-3),] = SAMPLING_RATE/6 * (xy[4:(N-1),] + xy[3:(N-2),] - xy[1:(N-4),] - xy[0:(N-5),])
        v[1,] = SAMPLING_RATE/2 * (xy[2,] - xy[0,])
        v[(N-2),] = SAMPLING_RATE/2 * (xy[(N-1),] - xy[(N-3),])   
        # this has to be added for python compatibility, as indexing does not include the last element
        v[(N-3),] = SAMPLING_RATE/6 * (xy[(N-1),] + xy[(N-2),] - xy[(N-4),] - xy[(N-5),])
    else:
        v[1:(N-2),] = SAMPLING_RATE/2 * (xy[2:(N-1),] - xy[0:(N-3),])
        # this has to be added for python compatibility:
        v[(N-2),] = SAMPLING_RATE/2 * (xy[(N-1),] - xy[(N-3),])
    return(v)



def microsacc(x, y, VFAC=5, MINDUR=3, SAMPLING=1000):
    #============================================================
    #  microsacc() INPUT:
    #  x[,1:2]		position vector
    #  VFAC			  relative velocity threshold
    #  MINDUR		  minimal saccade duration
    #
    #  microsacc() OUTPUT:
    #  $table[,1:7]		[(1) onset, (2) end, (3) peak velocity, 
    #                 (4) horizontal component, (5) vertical component,
    #			            (6) horizontal amplitude, (6) vertical amplitude ] 	
    #  $radius		    parameters of elliptic threshold
    #---------------------------------------------------------------------
    
    # Compute velocity
    v = vecvel(x, y, SAMPLING_RATE=SAMPLING, FIVE_POINT_SMOOTH=True)
    
    # Compute threshold
    medx = np.median(v[:,0], axis=0)[0,0]
    msdx = np.sqrt( np.median(np.square((v[:,0]-medx)), axis=0) )[0,0]
    medy = np.median(v[:,1], axis=0)[0,0]
    msdy = np.sqrt( np.median(np.square((v[:,1]-medy)), axis=0) )[0,0]
    if msdx<1e-10:
        msdx = np.sqrt( np.mean(np.square(v[:,0])) - np.square(np.mean(v[:,0])) )
        if msdx<1e-10:
            raise ValueError("msdx<realmin in microsacc()")
    if msdy<1e-10:
        msdy = np.sqrt( np.mean(np.square(v[:,1])) - np.square(np.mean(v[:,1])) )
        if msdy<1e-10:
            raise ValueError("msdy<realmin in microsacc()")
    radiusx = VFAC*msdx
    radiusy = VFAC*msdy
    radius = np.array([radiusx,radiusy])
    
    # Apply test criterion: elliptic treshold
    test = np.square(v[:,0]/radiusx) + np.square(v[:,1]/radiusy)
    indx = np.argwhere(test>1)[:,0]
    
    # Determine saccades
    N = len(indx) 
    nsac = 0 # counter for saccades
    sac = np.array([])
    dur = 1 # counter for single saccade duration
    a = 0 # 
    k = 0 # iterator over samples
    
    # Loop over saccade candidates
    while k<(N-1):
        if indx[k+1]-indx[k]==1: # onset
            dur = dur + 1
        else:
            # Minimum duration criterion (exception: last saccade)
            if dur>=MINDUR:
                nsac += 1
                b = k
                if len(sac)==0:
                    sac = np.concatenate((sac, np.array([indx[a],indx[b],0,0,0,0,0]) ), axis=0)
                else:
                    sac = np.vstack((sac, np.array([indx[a],indx[b],0,0,0,0,0]) ))
            a = k+1
            dur = 1
        k = k + 1
    
    # Check minimum duration for last microsaccade
    if dur>=MINDUR:
        nsac = nsac + 1
        b = k
        sac = np.vstack((sac, np.array([indx[a],indx[b],0,0,0,0,0])))
    
    # determine metrics of last saccade
    if nsac>0:
      # Compute peak velocity, horiztonal and vertical components
      for s in range(nsac):
        # Onset and offset for saccades
        a = int(sac[s,0])
        b = int(sac[s,1])
        idx = np.array(np.arange(a, b+1), dtype='int') # a:b in R
        # Saccade peak velocity (vpeak)
        vpeak = np.max( np.sqrt( np.square(v[idx,0]) + np.square(v[idx,1]) ) )
        sac[s,2] = vpeak
        # Saccade vector (dx,dy)
        dx = x[b]-x[a] 
        dy = y[b]-y[a] 
        sac[s,3:5] = np.array([dx,dy])
        # Saccade amplitude (dX,dY)
        minx = np.min(x[idx])
        maxx = np.max(x[idx])
        miny = np.min(y[idx])
        maxy = np.max(y[idx])
        ix1 = np.argmin(x[idx])
        ix2 = np.argmax(x[idx])
        iy1 = np.argmin(y[idx])
        iy2 = np.argmax(y[idx])
        dX = np.sign(ix2-ix1)*(maxx-minx)
        dY = np.sign(iy2-iy1)*(maxy-miny)
        sac[s,5:7] = np.array([dX,dY])
    else:
        sac = np.array([])
    # return saccade table and threshold info
    return(sac, radius)


def absmax(x):
    return(x[abs(x)==max(abs(x))][0])


def binsacc(sacl,sacr):
    #============================================================
    #  INPUT:
    #  msl$table		monocular saccades left eye
    #  msr$table		monocular saccades right eye  
    #
    #  OUTPUT:
    #  sac$N[1:3]		      number of microsaccades (bin, monol, monor)
    #  sac$bin[:,1:14]    binocular microsaccades (right eye/left eye)
    #  sac$monol[:,1:7]   monocular microsaccades of the left eye
    #  sac$monor[:,1:7]   monocular microsaccades of the right eye
    #  Basic saccade parameters: (1) onset, (2) end, (3) peak velocity, 
    #  (4) horizontal component, (5) vertical component, (6) horizontal 
    #  amplitude, (7) vertical amplitude
    #---------------------------------------------------------------------
    numr = len(sacr[:,0])
    numl = len(sacl[:,0])
    NB = 0
    NR = 0
    NL = 0
    if numr*numl>0:
      # Determine saccade clusters
      TR = np.max(sacr[:,1])
      TL = np.max(sacl[:,1])
      TB = np.int32(np.max([TL,TR])+1) # 
      s = np.zeros(TB+1) #rep(0,(TB+1))
      for i in range(np.shape((sacl))[0]):
          left = np.arange(sacl[i,0], sacl[i,1]+1, dtype='int')
          s[left] = 1
      for i in range(np.shape((sacr))[0]):
          right = np.arange(sacr[i,0], sacr[i,1]+1, dtype='int')
          s[right] = 1
      s[0] = 0
      s[TB] = 0
      
      # Find onsets and offsets of microsaccades
      onoff = np.argwhere(np.diff(s)!=0)
      assert(len(onoff)%2==0)
      m = onoff.reshape((round(len(onoff)/2), 2))
      N = np.shape(m)[0]
      
      # Determine binocular saccades
      bino = []
      monol = []
      monor = []
      for i in range(N):
        left  = np.argwhere( (m[i,0]<=sacl[:,0]) & (sacl[:,1]<=m[i,1]) )
        right = np.argwhere( (m[i,0]<=sacr[:,0]) & (sacr[:,1]<=m[i,1]) )
        # Binocular saccades
        if (len(left)*len(right)) > 0:
            ampr = np.sqrt( np.square(sacr[right,5]) + np.square(sacr[right,6]) )
            ampl = np.sqrt( np.square(sacl[left,5])  + np.square(sacl[left,6]) )
            # Determine largest event in each eye
            ir = np.argmax(ampr)
            il = np.argmax(ampl)
            NB = NB + 1
            if len(bino)==0:
                bino = np.hstack(([sacl[left[il],:], sacr[right[ir],:]])) 
            else:
                bino = np.vstack((bino, np.hstack(([sacl[left[il],:], sacr[right[ir],:]])) )) 
        else:
            # Determine monocular saccades
            if len(left)==0:
                NR += 1
                ampr = np.sqrt( np.square(sacr[right,5]) + np.square(sacr[right,6]) )
                ir = np.argmax(ampr)
                if len(monor)==0:
                    monor = sacr[right[ir],:]
                else:
                    monor = np.vstack((monor, sacr[right[ir],:]))
            if len(right)==0:
                NL += 1
                ampl = np.sqrt( np.square(sacl[left,5])  + np.square(sacl[left,6]) )
                il = np.argmax(ampl)
                if len(monol)==0:
                    monol = sacl[left[il],:]
                else:
                    monol = np.vstack((monol, sacl[left[il],:]))
    else: # numr*numl>0
        # Special case of exclusively monocular saccades
        if numr==0:
            bino = []
            monor = []
            monol = sacl
        if numl==0:
            bino = []
            monol = []
            monor = sacr
    #sac = list(N=c(NB,NL,NR),bino=bino,monol=monol,monor=monor)
    sac_N = [NB, NL, NR]
    return(sac_N, bino, monol, monor)


def microsacc_merge(s, n_samples_between=20, remove_subsequent=False):
    """
    Input: Engbert-Kliegl algorithm table, that is, output from microsacc()
    Output: shortened EK table, where extremely close saccade candidates are merged or removed
    """
    # what kind of input do we have?
    if s.shape[1] > 7:
        is_binocular = True # saccade table: binocular case (14 cols)
    else:
        is_binocular = False # saccade table: monocular case (7 cols)
    # pre-allocate
    sac_i = -1
    sacs = np.zeros((np.shape(s)[0]))
    # 1. scan for individual saccades (they must be separated by n_samples_between)
    for r in range(np.shape(s)[0]):
        if (sac_i<0 or (s[r,0]-s[r-1,1])>n_samples_between) or \
            (is_binocular==True and (sac_i<0 or (s[r,0+7]-s[r-1,1+7])>n_samples_between)):
            sac_i += 1
        sacs[r] = sac_i
    n_individual_sacs = sac_i+1 # since index starts at 0
    # 2. compute summary statistics for each sac_i
    new_s = np.zeros((n_individual_sacs, np.shape(s)[1]))
    for sac_i in range(n_individual_sacs):
        row_index = sacs==sac_i
        if np.sum(row_index)>1: # do we have more than 1 row per saccade candidate?
            if not remove_subsequent: # default option: merge candidates
                individual_on = np.min(s[row_index, 0])
                individual_off = np.max(s[row_index, 1])
                individual_vpeak = np.max(s[row_index, 2])
                individual_compx = np.sum(s[row_index, 3])
                individual_compy = np.sum(s[row_index, 4])
                individual_ampx = absmax(s[row_index, 5])
                individual_ampy = absmax(s[row_index, 6])
                if is_binocular:
                    individual_on_2 = np.min(s[row_index, 0+7])
                    individual_off_2 = np.max(s[row_index, 1+7])
                    individual_vpeak_2 = np.max(s[row_index, 2+7])
                    individual_compx_2 = np.sum(s[row_index, 3+7])
                    individual_compy_2 = np.sum(s[row_index, 4+7])
                    individual_ampx_2 = absmax(s[row_index, 5+7])
                    individual_ampy_2 = absmax(s[row_index, 6+7])
            else: # conservative option: remove second (or third, or fourth) candidates
                individual_on = s[row_index, 0][0]
                individual_off = s[row_index, 1][0]
                individual_vpeak = s[row_index, 2][0]
                individual_compx = s[row_index, 3][0]
                individual_compy = s[row_index, 4][0]
                individual_ampx = s[row_index, 5][0]
                individual_ampy = s[row_index, 6][0]
                if is_binocular:
                    individual_on_2 = s[row_index, 0+7][0]
                    individual_off_2 = s[row_index, 1+7][0]
                    individual_vpeak_2 = s[row_index, 2+7][0]
                    individual_compx_2 = s[row_index, 3+7][0]
                    individual_compy_2 = s[row_index, 4+7][0]
                    individual_ampx_2 = s[row_index, 5+7][0]
                    individual_ampy_2 = s[row_index, 6+7][0]
            sac_temp = np.array([individual_on, individual_off, 
                                        individual_vpeak, 
                                        individual_compx, individual_compy, 
                                        individual_ampx, individual_ampy])
            if is_binocular:
                sac_temp = np.hstack((sac_temp, 
                                      np.array([individual_on_2, individual_off_2, 
                                                individual_vpeak_2, 
                                                individual_compx_2, individual_compy_2, 
                                                individual_ampx_2, individual_ampy_2])))
            new_s[sac_i, :] = sac_temp
        elif np.sum(row_index)==1: # just one saccade candidate
            new_s[sac_i, :] = s[row_index, :]
        else:
            raise ValueError("In microsacc_merge np.sum(row_index) should be at least 1!")
    return(new_s)
    
    

if __name__ == '__main__': 
    
    # packages we need for the demo
    import matplotlib.pyplot as plt
    import pandas as pd
    
    ### read a single saccade
    sac_data = pd.read_csv("some_saccade.csv")
    x = sac_data.x
    y = sac_data.y
    t = sac_data.time
    samp_rate = 1000.0/np.median(sac_data.samp_freq)
    print("sampling rate:", samp_rate, "Hz")
    # plot the x-y saccade trajectory
    plt.plot(t, np.sqrt(x**2 + y**2))
    plt.show()
    
    # compute velocity space
    v = vecvel(x, y, SAMPLING_RATE=samp_rate)
    plt.plot(v[:,0], v[:,1])
    
    # run saccade detection
    sac_tab, thres = microsacc(x, y, MINDUR = 5, SAMPLING=samp_rate)
    print(sac_tab)
    print(thres)
    
    # simulate an additional cluster to check the new algorithm
    # a three-saccade cluster:
    s = np.vstack((sac_tab, np.array([54, 60, 2000.0, 11.0, 1.25, 11.0, 1.25])))
    # independent saccades
    s = np.vstack((np.array([1, 10, 10000.0, 300.0, 10, 300, 10]), s))
    s = np.vstack((s, np.array([300, 330, 30000.0, 600.0, 60, 600, 60])))
    
    # test saccade cluster algorithm real quick
    s_reduced = microsacc_merge(s, n_samples_between=10, remove_subsequent=False)
    
    # run binocular detection, just for checks
    n_sac, bin_tab, monol_tab, monor_tab = binsacc(sac_tab, sac_tab)
    
    ### load real binocular data (provided by Engbert et al)
    bino_data = pd.read_csv("f01.005.dat", sep = '\t', header = None)
    bino_data.rename(columns={0: 't', 1: 'xl', 2: 'yl', 3: 'xr', 4: 'yr'}, inplace=True)
    
    # perform saccade detection for each eye
    l_sac_tab, l_radius = microsacc(x = bino_data.xl, y = bino_data.yl, SAMPLING=500)
    l_sac_tab_reduced = microsacc_merge(l_sac_tab)
    r_sac_tab, r_radius = microsacc(x = bino_data.xr, y = bino_data.yr, SAMPLING=500)
    r_sac_tab_reduced = microsacc_merge(r_sac_tab)
    
    # perform binocular detection
    n_sac, bin_tab, monol_tab, monor_tab = binsacc(l_sac_tab, r_sac_tab)
    
    # -> this is the same output as the R version of the algorithm
    
    