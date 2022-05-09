## Analyze the dimension, time, and mflop data (under different thread numbers and different server)

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import glob, os

ind = 1

if ind == 1:
    os.chdir('./data/trial5')
    filelst = []
    idxlst = []
    for file in glob.glob("*.txt"):
        filelst.append(file)
        thread = file[:-4].split('-')[-1]+" threads"
        idxlst.append(thread)

    print(idxlst)

    for thefile in filelst:
        f1 = pd.read_csv(thefile, names=["N","time","MFLOP"], sep=";")
        x = f1["N"].tolist()
        x = np.log2(x)
        # mflop = f1["MFLOP"].tolist()
        time = f1["time"].tolist()
        time = np.log2(time)
        # plt.plot(x,mflop,'s-')
        plt.plot(x,time,'o-')

    plt.legend(idxlst)
    plt.xlabel("Dimension (in log2 scale)")
    plt.ylabel("Time (in log2 scale)")
    plt.title("Dimension vs. Time under different # threads (SNAPPY1)")

    plt.savefig("../../fig/dimension-time-3.jpg")


elif ind == 2:
    os.chdir('./data/trial4')
    filelst = []
    idxlst = []
    for file in glob.glob("*.txt"):
        filelst.append(file)
        thread = file[:-6]
        idxlst.append(thread)

    filelst.sort()
    idxlst.sort()
    print(idxlst)

    for thefile in filelst:
        f1 = pd.read_csv(thefile, names=["N","time","MFLOP"], sep=";")
        x = f1["N"].tolist()
        x = np.log2(x)
        # mflop = f1["MFLOP"].tolist()
        time = f1["time"].tolist()
        time = np.log2(time)
        # plt.plot(x,mflop,'s-')
        plt.plot(x,time,'o-')

    plt.legend(idxlst)
    plt.xlabel("Dimension (in log2 scale)")
    plt.ylabel("Time (in log2 scale)")
    plt.title("Dimension vs. Time under different servers (# threads=8)")

    plt.savefig("../../fig/dimension-time-server-2.jpg")