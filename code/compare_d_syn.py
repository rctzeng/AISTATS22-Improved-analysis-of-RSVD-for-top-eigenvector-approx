import optparse
import pandas as pd

from utility import *

opt = parse_arg()
DIM, T, q = int(opt.N), int(opt.T), 1
np.random.seed(1234)

df = pd.DataFrame({"d":[], "dataset":[], "algo":[], "R":[], "eigval":[], "time":[]})
# real-world datasets
for dname in ['Type1', 'Type2', 'Type3', 'Type4']:
    print("[{}]".format(dname))
    N, A = gen_matrix(dname, N=DIM) # test
    # baseline: scipy
    classic_eigval, classic_eigvec, classic_time = classic(A, N)
    df = df.append({"d":-1, "dataset":dname, "algo":'eigsh', "R":1.0, "eigval":classic_eigval, "time":classic_time}, ignore_index=True)
    for d in [1,5,10,25,50]:
        print("[d={}, T={}]".format(d, T))
        # randomized svd
        vs2, ts2, vs3, ts3 = [],[],[],[]
        for t in range(T):
            # RSVD
            RSVD_eigval, RSVD_eigvec, RSVD_time = RSVD(A, N, d, q)
            vs2 += [RSVD_eigval]
            ts2 += [RSVD_time]
            df = df.append({"d":d, "dataset":dname, "algo":'RSVD', "R":vs2[-1]/classic_eigval, "eigval":vs2[-1], "time":ts2[-1]}, ignore_index=True)
            # RandSum
            RSum_eigval, RSum_eigvec, RSum_time = RandSum(A, N, d, q)
            vs3 += [RSum_eigval]
            ts3 += [RSum_time]
            df = df.append({"d":d, "dataset":dname, "algo":'RSum', "R":vs3[-1]/classic_eigval, "eigval":vs3[-1], "time":ts3[-1]}, ignore_index=True)
        print(" | d={}, T={} |".format(d, T))
        print("(R-SVD):\tR={:.1f}\tspeedup={:.1f}x".format(np.mean(vs2)/classic_eigval, classic_time/np.mean(ts2)))
        print("(RandSum):\tR={:.1f}\tspeedup={:.1f}x".format(np.mean(vs3)/classic_eigval, classic_time/np.mean(ts3)))
        df.to_csv('SYN_d_q{}_n{}.csv'.format(q, DIM), index=False)
