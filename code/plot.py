import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

from utility import *

DATASET_LIST = ['wikivot', 'referendum', 'slashdot', 'wikicon'] + ['p2pgnutella31', 'youtube', 'roadnetCA', 'fb-artist']
Density = {'p2pgnutella31':2.3630204838142714, 'youtube':2.632522975795011, 'roadnetCA':1.4077949080147323, 'fb-artist':16.21906364446204} # edge density, required by computation of modularity score
PCA_LIST = ['Type1', 'Type2', 'Type3', 'Type4']

def plot_q_Task(fname, qs, MODE='adj', ERR=False):
    df = pd.read_csv('{}.csv'.format(fname))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6,3), constrained_layout=True)
    x1, x2 = df[df["algo"]=='RSVD'], df[df["algo"]=='RSum']
    if MODE=='adj': c, ctitle, otitle, plot_baseline = "R", "$R(\hat{u})$", "polarity", True
    else: c, ctitle, otitle, plot_baseline = "eigval", "$\hat{u}^TA\hat{u}$", "modularity", False
    for j,dname in enumerate(DATASET_LIST):
        tx1, tx2 = x1[x1["dataset"]==dname], x2[x2["dataset"]==dname]
        if len(tx1)==0: continue
        rs1, rs2, re1, re2, os1, os2 = [],[],[],[],[],[]
        for q in qs:
            rs1 += [tx1[tx1["q"]==q][c].mean()]
            rs2 += [tx2[tx2["q"]==q][c].mean()]
            re1 += [tx1[tx1["q"]==q][c].std()]
            re2 += [tx2[tx2["q"]==q][c].std()]
            o1, o2 = tx1[tx1["q"]==q]["obj"].mean(), tx2[tx2["q"]==q]["obj"].mean()
            if MODE=='mod': o1, o2 = o1/(4*Density[dname]), o2/(4*Density[dname])
            os1 += [o1]
            os2 += [o2]
        if ERR:
            axs[0].errorbar(qs, rs1, yerr=re1, label=dname, ls='-', color='C{}'.format(j))
            axs[0].errorbar(qs, rs2, yerr=re2, ls='-.', color='C{}'.format(j))
        else:
            axs[0].plot(qs, rs1, label=dname, ls='-', color='C{}'.format(j))
            axs[0].plot(qs, rs2, ls='-.', color='C{}'.format(j))
        axs[1].plot(qs, os1, label=dname, ls='-', color='C{}'.format(j))
        axs[1].plot(qs, os2, ls='-.', color='C{}'.format(j))
        # baseline scipy
        if plot_baseline:
            tx0 = df[(df["algo"]=='eigsh')&(df["dataset"]==dname)]
            o0 = tx0["obj"].mean()
            axs[1].hlines(o0, 1, qs[-1], label='Lanczos', ls='dotted', color='C{}'.format(j), linewidth=1)
    axs[0].set_title(ctitle)
    axs[0].set_xlabel("$q$")
    axs[1].set_title(otitle)
    axs[1].set_xlabel("$q$")
    plt.savefig('{}.pdf'.format(fname), bbox_inches='tight')
def plot_d_Task(fname, ds, MODE='adj', ERR=False):
    df = pd.read_csv('{}.csv'.format(fname))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6,3), constrained_layout=True)
    x1, x2 = df[df["algo"]=='RSVD'], df[df["algo"]=='RSum']
    if MODE=='adj': c, ctitle, otitle, plot_baseline = "R", "$R(\hat{u})$", "polarity", True
    else: c, ctitle, otitle, plot_baseline = "eigval", "$\hat{u}^TA\hat{u}$", "modularity", False
    for j,dname in enumerate(DATASET_LIST):
        tx1, tx2 = x1[x1["dataset"]==dname], x2[x2["dataset"]==dname]
        if len(tx1)==0: continue
        rs1, rs2, re1, re2, os1, os2 = [],[],[],[],[],[]
        for d in ds:
            rs1 += [tx1[tx1["d"]==d][c].mean()]
            rs2 += [tx2[tx2["d"]==d][c].mean()]
            re1 += [tx1[tx1["d"]==d][c].std()]
            re2 += [tx2[tx2["d"]==d][c].std()]
            o1, o2 = tx1[tx1["d"]==d]["obj"].mean(), tx2[tx2["d"]==d]["obj"].mean()
            if MODE=='mod': o1, o2 = o1/(4*Density[dname]), o2/(4*Density[dname])
            os1 += [o1]
            os2 += [o2]
        if ERR:
            axs[0].errorbar(ds, rs1, yerr=re1, label=dname, ls='-', color='C{}'.format(j))
            axs[0].errorbar(ds, rs2, yerr=re2, ls='-.', color='C{}'.format(j))
        else:
            axs[0].plot(ds, rs1, label=dname, ls='-', color='C{}'.format(j))
            axs[0].plot(ds, rs2, ls='-.', color='C{}'.format(j))
        axs[1].plot(ds, os1, label=dname, ls='-', color='C{}'.format(j))
        axs[1].plot(ds, os2, ls='-.', color='C{}'.format(j))
        # baseline scipy
        if plot_baseline:
            tx0 = df[(df["algo"]=='eigsh')&(df["dataset"]==dname)]
            o0 = tx0["obj"].mean()
            axs[1].hlines(o0, 1, ds[-1], label='Lanczos', ls='dotted', color='C{}'.format(j), linewidth=1)
    axs[0].set_title(ctitle)
    axs[0].set_xlabel("$d$")
    axs[1].set_title(otitle)
    axs[1].set_xlabel("$d$")
    plt.savefig('{}.pdf'.format(fname), bbox_inches='tight')
def plot_d_q_Task(fdname, fqname, oname, ds, qs, MODE='adj'):
    df1, df2 = pd.read_csv('{}.csv'.format(fdname)), pd.read_csv('{}.csv'.format(fqname))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6,3), constrained_layout=True)
    x11, x12, x21, x22 = df1[df1["algo"]=='RSVD'], df1[df1["algo"]=='RSum'], df2[df2["algo"]=='RSVD'], df2[df2["algo"]=='RSum']
    if MODE=='adj': c, otitle = "obj", "polarity"
    elif MODE=='mod': c, otitle = "obj", "modularity"
    else: c, otitle = "eigval", "$R(\hat{u})$"
    # plot d
    for j,dname in enumerate(DATASET_LIST):
        tx1, tx2 = x11[x11["dataset"]==dname], x12[x12["dataset"]==dname]
        if len(tx1)==0: continue
        rs1, rs2 = [],[]
        for d in ds:
            r1, r2 = tx1[tx1["d"]==d][c].mean(), tx2[tx2["d"]==d][c].mean()
            if MODE=='mod': r1, r2 = r1/(4*Density[dname]), r2/(4*Density[dname])
            rs1 += [r1]
            rs2 += [r2]
        axs[0].plot(ds, rs1, label=dname, ls='-', color='C{}'.format(j))
        axs[0].plot(ds, rs2, ls='-.', color='C{}'.format(j))
    # plot q
    for j,dname in enumerate(DATASET_LIST):
        tx1, tx2 = x21[x21["dataset"]==dname], x22[x22["dataset"]==dname]
        if len(tx1)==0: continue
        rs1, rs2 = [],[]
        for q in qs:
            r1, r2 = tx1[tx1["q"]==q][c].mean(), tx2[tx2["q"]==q][c].mean()
            if MODE=='mod': r1, r2 = r1/(4*Density[dname]), r2/(4*Density[dname])
            rs1 += [r1]
            rs2 += [r2]
        axs[1].plot(qs, rs1, label=dname, ls='-', color='C{}'.format(j))
        axs[1].plot(qs, rs2, ls='-.', color='C{}'.format(j))
    axs[0].set_title(otitle)
    axs[0].set_xlabel("$d$")
    axs[1].set_title(otitle)
    axs[1].set_xlabel("$q$")
    plt.savefig('{}.pdf'.format(oname), bbox_inches='tight')

def plot_q_PCA(fname, qs, ERR=False):
    df = pd.read_csv('{}.csv'.format(fname))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6,3), constrained_layout=True)
    x1, x2 = df[df["algo"]=='RSVD'], df[df["algo"]=='RSum']
    for j,dname in enumerate(PCA_LIST):
        tx1, tx2 = x1[x1["dataset"]==dname], x2[x2["dataset"]==dname]
        if len(tx1)==0: continue
        tx0 = df[(df["algo"]=='eigsh')&(df["dataset"]==dname)]
        t0 = tx0["time"].mean()
        rs1, rs2, re1, re2, ts1, ts2 = [],[],[],[],[],[]
        for q in qs:
            rs1 += [tx1[tx1["q"]==q]["R"].mean()]
            rs2 += [tx2[tx2["q"]==q]["R"].mean()]
            re1 += [tx1[tx1["q"]==q]["R"].std()]
            re2 += [tx2[tx2["q"]==q]["R"].std()]
            ts1 += [t0/tx1[tx1["q"]==q]["time"].mean()]
            ts2 += [t0/tx2[tx2["q"]==q]["time"].mean()]
        if ERR:
            axs[0].errorbar(qs, rs1, yerr=re1, label=dname, ls='-', color='C{}'.format(j))
            #axs[0].errorbar(qs, rs2, yerr=re2, ls='-.', color='C{}'.format(j))
        else:
            axs[0].plot(qs, rs1, label=dname, ls='-', color='C{}'.format(j))
            #axs[0].plot(qs, rs2, ls='-.', color='C{}'.format(j))
        axs[1].plot(qs, ts1, ls='-', color='C{}'.format(j))
        #axs[1].plot(qs, ts2, ls='-.', color='C{}'.format(j))
    axs[0].set_title("$R(\hat{u})$")
    axs[0].set_xlabel("$q$")
    axs[1].set_title("Speedup")
    axs[1].set_xlabel("$q$")
    axs[1].hlines(1, 1, qs[-1], ls='dotted', color='C9', linewidth=1)
    plt.savefig('{}.pdf'.format(fname), bbox_inches='tight')
def plot_d_PCA(fname, ds, ERR=False):
    df = pd.read_csv('{}.csv'.format(fname))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6,3), constrained_layout=True)
    x1, x2 = df[df["algo"]=='RSVD'], df[df["algo"]=='RSum']
    for j,dname in enumerate(PCA_LIST):
        tx1, tx2 = x1[x1["dataset"]==dname], x2[x2["dataset"]==dname]
        if len(tx1)==0: continue
        tx0 = df[(df["algo"]=='eigsh')&(df["dataset"]==dname)]
        t0 = tx0["time"].mean()
        rs1, rs2, re1, re2, ts1, ts2 = [],[],[],[],[],[]
        for d in ds:
            rs1 += [tx1[tx1["d"]==d]["R"].mean()]
            rs2 += [tx2[tx2["d"]==d]["R"].mean()]
            re1 += [tx1[tx1["d"]==d]["R"].std()]
            re2 += [tx2[tx2["d"]==d]["R"].std()]
            ts1 += [t0/tx1[tx1["d"]==d]["time"].mean()]
            ts2 += [t0/tx2[tx2["d"]==d]["time"].mean()]
        if ERR:
            axs[0].errorbar(ds, rs1, yerr=re1, label=dname, ls='-', color='C{}'.format(j))
            #axs[0].errorbar(ds, rs2, yerr=re2, ls='-.', color='C{}'.format(j))
        else:
            axs[0].plot(ds, rs1, label=dname, ls='-', color='C{}'.format(j))
            #axs[0].plot(ds, rs2, ls='-.', color='C{}'.format(j))
        axs[1].plot(ds, ts1, ls='-', color='C{}'.format(j))
        #axs[1].plot(ds, ts2, ls='-.', color='C{}'.format(j))
    axs[0].set_title("$R(\hat{u})$")
    axs[0].set_xlabel("$d$")
    axs[1].set_title("Speedup")
    axs[1].set_xlabel("$d$")
    axs[1].hlines(1, 1, ds[-1], ls='dotted', color='C9', linewidth=1)
    plt.savefig('{}.pdf'.format(fname), bbox_inches='tight')

def plot_SyntheticEigvals(N):
    fig = plt.figure(figsize=(3,3))
    for j,type in enumerate(PCA_LIST):
        Sigma = get_eigvals(type, N=N)
        plt.plot(np.sort(Sigma)[::-1], label=type, color='C{}'.format(j))
        print("kappa={:.4f}".format(np.sum(np.array(Sigma)**3) / np.sum(np.abs(np.array(Sigma)**3))))
    plt.xlabel('$i$')
    plt.yscale('symlog', linthreshy=0.01)
    plt.title('$\lambda_i$')
    plt.savefig('synthetic-eigvals_n{}.pdf'.format(N), bbox_inches='tight')

plot_d_q_Task("SCG-d_q1-R", "SCG-q_d10-R", "SCG-real_dq", [1,5,10,25,50], [1,2,4,8,16], MODE='adj')
plot_d_q_Task("MOD-d_q1-S", "MOD-q_d10-S", "MOD-real_dq", [1,5,10,25,50], [1,2,4,8,16], MODE='mod')

plot_SyntheticEigvals(10000)
plot_d_PCA("SYN_d_q1_n10000", [1,5,10,25])
plot_q_PCA("SYN_q_d10_n10000", [1,2,4,8])
