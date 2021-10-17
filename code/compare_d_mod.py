import optparse
import pandas as pd

from utility import *

opt = parse_arg()
T, MODE, ROUND, q = int(opt.T), 'mod', 'S', 1
np.random.seed(1234)

df = pd.DataFrame({"d":[], "dataset":[], "algo":[], "R":[], "eigval":[], "obj":[], "time":[]})
# real-world datasets
for dname in ['fb-artist', 'p2pgnutella31', 'youtube', 'roadnetCA']:
    print("[{}]".format(dname))
    N, A = read_dataset("datasets/{}.txt".format(dname))
    for d in [1,5,10,25,50]:
        print("[d={}, T={}, M={}]".format(d, T, MODE))
        # randomized svd
        vs2, ts2, ps2, vs3, ts3, ps3 = [],[],[],[],[],[]
        for t in range(T):
            # RSVD
            RSVD_eigval, RSVD_eigvec, RSVD_time = RSVD(A, N, d, q, MODE)
            vs2 += [RSVD_eigval]
            ts2 += [RSVD_time]
            RSVD_r = rounding(RSVD_eigvec, A, N, ROUND=ROUND)
            ps2 += [compute_Obj(A, RSVD_r, MODE)]
            df = df.append({"d":d, "dataset":dname, "algo":'RSVD', "R":-1, "eigval":vs2[-1], "obj":ps2[-1], "time":ts2[-1]}, ignore_index=True)
            # RandSum
            RSum_eigval, RSum_eigvec, RSum_time = RandSum(A, N, d, q, MODE)
            vs3 += [RSum_eigval]
            ts3 += [RSum_time]
            RSum_r = rounding(RSum_eigvec, A, N, ROUND=ROUND)
            ps3 += [compute_Obj(A, RSum_r, MODE)]
            df = df.append({"d":d, "dataset":dname, "algo":'RSum', "R":-1, "eigval":vs3[-1], "obj":ps3[-1], "time":ts3[-1]}, ignore_index=True)
        print(" | d={}, T={} |".format(d, T))
        print("(R-SVD):\teigval={:.1f}\tobj={:.1f}".format(np.mean(vs2), np.mean(ps2)))
        print("(RandSum):\teigval={:.1f}\tobj={:.1f}".format(np.mean(vs3), np.mean(ps3)))
        df.to_csv('MOD-d_q{}-{}.csv'.format(q, ROUND), index=False)
