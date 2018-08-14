# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 11:17:13 2017

@author: wfok007
"""

    
import scipy, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import itertools
from scipy import io
from sklearn.neighbors.kde import KernelDensity
import matplotlib
from scipy.fftpack import fft
import pickle
from sklearn import neighbors


def get(fileName, input_dir, key):
    """
    this function reads Matlab mat files
    inputs:
        fileName, input_dir = mat with linear and angular accelerations files in the directory 
        key = tags
    outputs:
        mat array
    """
    f = os.path.join(input_dir, fileName)
    # dictionary
    mat = io.loadmat(f)
    return mat[key]
    
def goGet(itemKey, input_dir, mats):
    """
    This funtion loads mat files and put the arrays in a list
    inputs:
        mats = list of mat files
        input_dir = storage directory
        itemKey = tag used to retrieve the right type of information
    outputs:
        list of arrays
    """
    accs = [get(mat, input_dir, itemKey) for mat in mats]
    return accs



def printout(a):
    print ('min: %f' %a.min())
    print ('max: %f' %a.max())
    print ('mean: %f' %a.mean())
    print ('median: %f' %np.median(a)) 

def findMeSomethingInteresting(acc, euler):
    """
    This function detects "interesting" signal
    Inputs:
        acc = linear accelerations
        euler = angular accelerations
    Outputs:
        a = signals
        mean_out = average of linear and angular accelerations
        std_out = their standard deviations
    """
    
    X = np.hstack([acc, euler])

    
    ''' forward '''
    mean = pd.rolling_mean(X, window_size)

    std = pd.rolling_std(X, window_size)

    ''' backward '''
    X2 = X[::-1] # create a view, not actually reverse
    mean2 = pd.rolling_mean(X2, window_size)

    std2 = pd.rolling_std(X2, window_size)
    ''' backward backward = forward '''
    mean3 = mean2[::-1]
    std3 = std2[::-1]
    
 
    mean[np.isnan(mean)] = np.inf
    std[np.isnan(std)] = np.inf
    mean3[np.isnan(mean3)] = np.inf
    std3[np.isnan(std3)] = np.inf
    
    # min out
    combo = np.dstack([mean, mean3])
    mean_out = np.min(combo, axis=2)
    
    combo2 = np.dstack([std, std3])
    std_out = np.min(combo2, axis=2)
    
    #check
    assert X.shape[0] == mean_out.shape[0]
    
    z = (X - mean_out)/std_out
    p = st.norm.cdf(z)
    # two sided test becomes one sided test, flipped over
    right_hand_side = p >= 0.5
    p[right_hand_side] = 1- p[right_hand_side]
    
    #printout(p)
    # fix, min p = 0
    # stability , too close to zero
    eps = 1e-15 # cutoff
    printout(p+eps)
    # log rule
    logP = np.log(p+eps)
    
    #printout(logP)
    
    # use the mean from acceleration or rotation, less sensitive to random outliers
    family = [np.mean(logP[:,0:3], axis=1), np.mean(logP[:,3::], axis=1)]
    together = np.vstack(family).transpose()
    # log rule
    a = np.sum( together, axis=1)
    #printout(a)
    return a, mean_out, std_out


def plotAccEulerInterestingNess(k, tag, acc, euler, a,  mean_out, std_out, videoT, velocity):
    """
    This function puts accelerations with their means and standard deviation in a chart.
    The interesting signals are plotted below
    
    Inputs:
        k = kth number of sample
        tag = types
        acc = linear accelerations
        euler = angular accelerations
        a = signals
        mean_out = average of linear and angular accelerations
        std_out = their standard deviations
        videoT = video time series
        velocity = pixel velocity (deprecated; will be removed in the next version)
    
    Outputs:
        rollingB = convolution on signals
        t[start:] = time series
        mask_a, mask_g = the accelerations that are labelled as signals
    """
    
    # chart style
    width = 1.5
    plt.figure(figsize=(18, 30))
    ax= plt.subplot(511)
    num_samples, num_dims = acc.shape
    
    t = [i/float(Hz) for i in range(num_samples)]
    # only have corresponding video from 100 onwards
    start = t.index(100)
    for i, label in enumerate(['x', 'y','z']):
        ax.plot(t[start:], acc[start:,i], linewidth=width, label=label)
        ax.fill_between(t[start:], (mean_out[start:,i]+2*std_out[start:,i]),
                    (mean_out[start:,i]-2*std_out[start:,i]),alpha=0.5)

    plt.legend()
    ax.set_ylabel('acceleration \n m/$\mathregular{s^2}$')
    plt.grid()
    plt.ylim(-200,200)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax2 = plt.subplot(512, sharex=ax)
    for i, label in enumerate([r'$\phi$', r'$\theta$', r'$\psi$',]):
        ax2.plot(t[start:], euler[start:,i], linewidth=width, label=label)
        ax2.fill_between(t[start:], (mean_out[start:,i+3]+std_out[start:,i+3]),
                        (mean_out[start:,i+3]-std_out[start:,i+3]),alpha=0.5)
    plt.legend()
    ax2.set_ylabel('angular \n acceleration \n $^\circ$/$\mathregular{s^2}$')
    plt.grid()

    plt.setp(ax2.get_xticklabels(), visible=False)
    ax3 = plt.subplot(513, sharex=ax)
    
    # turn log back to probability
    # unlikely to be random flips to how likelihood to be meaningful
    a = 1- np.exp(a)
    a = np.log(a)
    
    
    ax3.plot(t[start:],a[start:], linewidth=width)
 
    ax3.set_ylabel('log(probability)')
    plt.grid()
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    ax4 = plt.subplot(514, sharex=ax)
    bins = a[start:] > -np.exp(-11)
    rollingB = pd.rolling_sum(bins, 20*Hz)/Hz
    ax4.plot(t[start:], rollingB, linewidth=width)
    ax4.set_ylabel('number of steps/s')
    ax4.set_ylim(0,10)
    plt.grid()
    plt.setp(ax4.get_xticklabels(), visible=True)
    
    mask_a  = bins[:, np.newaxis]*acc[start:,:]
    mask_g  = bins[:, np.newaxis]*euler[start:,:]
    
    
    ax5 = plt.subplot(515, sharex=ax)
    ax5.plot(videoT, velocity,linewidth=width)
    ax5.set_ylim(0,0.25)
    plt.grid()
    ax5.set_ylabel('pixel \n velocity')
    plt.xlabel('time in seconds')
    plt.xlim(xmin=150, xmax=450) # adjust the min leaving max unchanged
    #plt.show()
    
    plt.savefig(os.path.join(output_dir, str(k)+tag), dpi=200)

    return rollingB, t[start:], mask_a, mask_g

def readMAT(input_dir):
    """
    This function reads Mat files and returns arrays 
    """
    
    mats = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('mat')]
    itemKey = 'acc'
    accs = goGet(itemKey,input_dir, mats)
    
    itemKey = 'euler'
    eulers = goGet(itemKey,input_dir, mats)
    return accs, eulers, mats

def findItems(which, items):
    return [i for i, item in enumerate(items) if item == which]

def stepStatistics(files, accsPos, eulersPos, tag, videoT, velocity):
    """
    This function estimates the average step in each category or each time point
    """
    
    steps_stat = []
    sheep_steps = []
    masks = []
    masks_g = []
    for i, (acc, euler) in enumerate(zip(accsPos, eulersPos)):
        
        print (i)
        # get signals    
        a,  mean_out, std_out = findMeSomethingInteresting(acc, euler)
        # plot signals
        number_steps, period, mask_a, mask_g  = plotAccEulerInterestingNess(i, tag, acc, euler, a,  mean_out, std_out,
                                    videoT, velocity)
        # replace nan
        number_steps = np.nan_to_num(number_steps)
        #total = sum(number_steps)
        #cumsum = np.cumsum(number_steps)
        sheep_steps.append(number_steps)
        # average step/ s
        steps_stat.append(np.mean(number_steps))
        masks.append(mask_a)
        masks_g.append(mask_g)
    
    # display the average sheep step per second
    sheepRecoveryDates = [ decoder[f.split('_')[1][0]] for f in files]
    
    for key in decoder.keys():
        print (decoder[key])
        day1 = findItems(decoder[key], sheepRecoveryDates)
        mean_stat = np.mean([steps_stat[i] for i in day1])
        print ('average sheep step per second %f' %mean_stat)
    
    return steps_stat, sheep_steps, masks, masks_g
    
    
def fftBox(steps):
    
    N = steps.shape[0] # sample points
    yf = fft(steps)
    xf = np.linspace(0, 1/(2*T), N//2)
    response = 2.0/N * np.abs(yf[0:N//2])
    
    return xf, response
    
def interpolateFFT(xf, response):
    
    knn=neighbors.KNeighborsRegressor()
    knn.fit(xf.reshape(-1,1), response)
    newY = knn.predict(newX)
    
    # verify outputs
#    plt.figure(figsize=(18, 16))
#    width = 1
#    ax = plt.subplot(111)
#    ax.set_xscale("log", nonposx='clip')
#    ax.set_yscale("log", nonposy='clip')
#
#    
#    ax.plot(newX, newY, linewidth=width)
#    plt.grid()
#    #ax.set_ylim(10e-5, 10e-1)
#    #ax.set_xlim(0,50) # plotting slow walking and trotting
#    plt.xlabel('frequency (Hz)')
#    plt.ylabel('frequency response')
#    #plt.title('t_'+decoder[key])
#    #plt.savefig(os.path.join(output_dir, 't_'+decoder[key]+'frequency_herding.png'))
#    plt.show()
    
    return newY

def sortRank(n, t_files, t_mask):
    """
    This function sorts and ranks the largest magnitude in the recordings.
    Inputs:
        n = nth ranks
        t_files = the group of files that show rehabilitation time points
        t_mask = recordings on different dates
    Outputs:
        allMax_t = magnitude distribution of n largest recordings
        timeStamps = rehabilitation time points
    """ 
    sheepRecoveryDates = [ decoder[f.split('_')[1][0]] for f in t_files]
    
    # find top values
    allMax_t = []
    timeStamps = []
    for key in decoder.keys():
        print (decoder[key])
        day1 = findItems(decoder[key], sheepRecoveryDates)
        maxValues = []
        for i in day1:
            here = np.abs(t_mask[i].flatten())
            index = np.argsort(here)
            maxValues += here[index[-n:]].tolist()

        allMax_t.append(maxValues)
        timeStamps.append(decoder[key])
    
    return allMax_t, timeStamps
    
def maxSpread(t_mask, c_mask, t_files, c_files):
    """
    This function identifies the largest magnitude in recordings.
    
    
    
    
    """
    n = 150 # top n number of candidates

    allMax_t, timeStamps = sortRank(n, t_files, t_mask)
    allMax_c, timeStamps = sortRank(n, c_files, c_mask)
    
    return allMax_c, allMax_t, timeStamps
        
       
if __name__ == '__main__':
    input_dir = r'C:\Users\wfok007\google drive on D\Docear\projects\PhD\scripts\IMU\output'
    output_dir = r'E:\working\activityText'
    treated_dir = r'Z:\postOpSheep\treated'
    control_dir = r'Z:\postOpSheep\control'
    
    plt.close('all')
    
    
    
    decoder = {'2':'day 1',
               '3':'week 1',
               '4':'week 2',
               '5':'week 3',
               '6':'week 4'}
    
    
    ''' global variables'''
    takeout = 4*60 # in seconds
    # sampling frequency
    Hz = 500
    window_size = int(Hz*takeout)

    font = {'family' : 'Arial',
        'weight' : 'bold',
        'size'   : 22}

    
    matplotlib.rc('font', **font)
    
    # get the video frames
    # deprecated; will be removed in the next version
    videoT = pickle.load( open(os.path.join(output_dir, "t.p"), "rb" ) )
    videoT= videoT+10
    
    velocity = np.load(os.path.join(output_dir, "velocity.npy"))
    
    
    tag = 'treated.png'
    
    accsPos, eulersPos, t_files = readMAT(treated_dir)
    t_steps_stat, t_sheep_steps, t_mask, tg_mask = stepStatistics(t_files, accsPos, eulersPos, tag, videoT, velocity)
    tag = 'control.png'
    accsPos, eulersPos, c_files = readMAT(control_dir)
    c_steps_stat, c_sheep_steps, c_mask, cg_mask = stepStatistics(c_files, accsPos, eulersPos, tag, videoT, velocity)
    
    
    # the histogram of the data
    allMax_c, allMax_t, timeStamps = maxSpread(t_mask, c_mask, t_files, c_files)
    fig = plt.figure(figsize=(18, 16))
    ax1 = fig.add_subplot(111)    # The big subplot
    axes = [fig.add_subplot(i,sharey=ax1,sharex=ax1) for i in range(511, 516)]
#    f, axes = plt.subplots(5)
    for ax, c,t, timeStamp in zip(axes, allMax_c, allMax_t, timeStamps):
        ax.hist(c, 20, label='control')
        ax.hist(t, 20, label='post op')
        print (timeStamp)
        print ('max for control %f' %np.max(c))
        print ('max for post op %f' %np.max(t))
        
        ax.set_title(timeStamp)
    ax1.set_xlabel('acceleration \n m/$\mathregular{s^2}$')
    ax1.set_ylabel('occurrences')
    plt.suptitle('the acceleration distribution between control and post Op sheep')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'linearHist.png'), dpi=200)
    
    print ('time \t control  \t treated  \t  percentage fall')
    for c,t, timeStamp in zip(allMax_c, allMax_t, timeStamps):
        c2 = np.max(c)
        t2 = np.max(t)
        print ('%s \t %f \t  %f \t %f' %(timeStamp, c2,t2, 100*(c2-t2)/c2))
    
 
    
    pickle.dump( t_steps_stat, open( os.path.join(output_dir, "t_steps_stat.p"), "wb" ) )
    pickle.dump( t_sheep_steps, open( os.path.join(output_dir, "t_sheep_stpes.p"), "wb" ) )
    pickle.dump( c_steps_stat, open( os.path.join(output_dir, "c_steps_stat.p"), "wb" ) )
    pickle.dump( c_sheep_steps, open( os.path.join(output_dir, "c_sheep_steps.p"), "wb" ) )
    
    pickle.dump( c_files, open( os.path.join(output_dir, "c_files.p"), "wb" ) )
    pickle.dump( t_files, open( os.path.join(output_dir, "t_files.p"), "wb" ) )
    
    plt.close('all')
    
#    t_steps_stat = pickle.load(open( os.path.join(output_dir, "t_steps_stat.p"), "rb" ) )
#    t_sheep_steps = pickle.load(open( os.path.join(output_dir, "t_sheep_stpes.p"), "rb" ) )
#    c_steps_stat = pickle.load(open( os.path.join(output_dir, "c_steps_stat.p"), "rb" ) )
#    c_sheep_steps = pickle.load( open( os.path.join(output_dir, "c_sheep_steps.p"), "rb" ) )
#    
#    c_files = pickle.load(open( os.path.join(output_dir, "c_files.p"), "rb" ) )
#    t_files = pickle.load(open( os.path.join(output_dir, "t_files.p"), "rb" ) )
    
    # fourier transform
    T = 1/Hz
    newX = np.linspace(0, 100, 25000).reshape(-1,1)
    
    # different time point for post op
    sheepRecoveryDates = [ decoder[f.split('_')[1][0]] for f in c_files]
    
    c_days = {}
    for key in decoder.keys():
        print (decoder[key])
        day1 = findItems(decoder[key], sheepRecoveryDates)
        responses = []
        
        
        plt.figure(figsize=(18, 16))
        width = 1
        ax = plt.subplot(111)
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
    
        for i in day1:
            steps = c_sheep_steps[i]
            xf, response = fftBox(steps)
            response = interpolateFFT(xf, response)
            responses.append(response)
            
            ax.plot(newX, response, linewidth=width)
        
        c_days[decoder[key]] = responses
        plt.grid()
        plt.xlabel('frequency (Hz)')
        plt.ylabel('frequency response')
        plt.title('c_'+decoder[key])
        plt.savefig(os.path.join(output_dir, 'c_'+decoder[key]+'frequency_herding.png'))
        #plt.show()
    plt.close('all')
    
    sheepRecoveryDates = [ decoder[f.split('_')[1][0]] for f in t_files]
    
    t_days = {}
    for key in decoder.keys():
        print (decoder[key])
        day1 = findItems(decoder[key], sheepRecoveryDates)
        responses = []
        xfs = []
        
        plt.figure(figsize=(18, 16))
        width = 1
        ax = plt.subplot(111)
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
    
        for i in day1:
            steps = t_sheep_steps[i]
            xf, response = fftBox(steps)
            response = interpolateFFT(xf, response)
            responses.append(response)
            
            ax.plot(newX, response, linewidth=width)
        t_days[decoder[key]] = responses
        
        plt.grid()
        plt.xlabel('frequency (Hz)')
        plt.ylabel('frequency response')
        plt.title('t_'+decoder[key])
        plt.savefig(os.path.join(output_dir, 't_'+decoder[key]+'frequency_herding.png'))
        #plt.show()
    plt.close('all')
    
    pickle.dump( t_days, open( os.path.join(output_dir, "t_days.p"), "wb" ) )
    pickle.dump( c_days, open( os.path.join(output_dir, "c_days.p"), "wb" ) )
    
#    t_days = pickle.load(open( os.path.join(output_dir, "t_days.p"), "rb" ) )
#    c_days = pickle.load(open( os.path.join(output_dir, "c_days.p"), "rb" ) )
    
    # plot the standard deviations as shaded area
    # issue: values approaching zero would lead to numerical instability on a log scale plot
    plt.figure(figsize=(18, 16))
    width = 1.5
    ax = plt.subplot(111)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    
    avg_days = {}    
    for key in c_days.keys():
        c = np.vstack(c_days[key])
        t = np.vstack(t_days[key])
        cBar = np.mean(c, axis=0)
        tBar = np.mean(t, axis=0)
        cError = np.std(c, axis=0)/5
        tError = np.std(t, axis=0)/5
        upper = cBar + cError
        lower = cBar - cError
        ax.fill_between(np.squeeze(newX), lower, upper, label='control-'+key)
        upper = tBar + tError
        lower = tBar - tError
        ax.fill_between(np.squeeze(newX), lower, upper, label='postOp-'+key)
        ax.set_xlim(0,50)
        #ax.errorbar(newX, cBar, yerr=cError, fmt='-o', label='control-'+key)
        #ax.errorbar(newX, tBar, yerr=tError, fmt='-x', label='postOp-'+key)
#        ax.plot(newX, cBar, linewidth=width, label='control-'+key)
#        ax.plot(newX, tBar, linewidth=width, label='postOp-'+key)
        
        
    plt.grid()
    plt.xlabel('frequency (Hz)')
    plt.ylabel('frequency response')
    
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'frequency_postOp.png'))
    plt.show()
    
