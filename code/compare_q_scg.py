import optparse
import pandas as pd

from utility import *

opt = parse_arg()
T, MODE, ROUND, d = int(opt.T), 'adj', 'R', 10
np.random.seed(1234)

df = pd.DataFrame({"q":[], "dataset":[], "algo":[], "R":[], "eigval":[], "obj":[], "time":[]})
# real-world datasets
for dname in ['wikivot', 'referendum', 'slashdot', 'wikicon']:
    print("[{}]".format(dname))
    N, A = read_dataset("datasets/{}.txt".format(dname))
    # baseline: scipy
    classic_eigval, classic_eigvec, classic_time = classic(A, N)
    classic_r = rounding(classic_eigvec, A, N, ROUND=ROUND)
    classic_obj = compute_Obj(A, classic_r, MODE)
    df = df.append({"q":-1, "dataset":dname, "algo":'eigsh', "R":1.0, "eigval":classic_eigval, "obj":classic_obj, "time":classic_time}, ignore_index=True)
    for q in [1,2,4,8,16]:
        print("[q={}, T={}, M={}]".format(q, T, MODE))
        # randomized svd
        vs2, ts2, ps2, vs3, ts3, ps3 = [],[],[],[],[],[]
        for t in range(T):
            # RSVD
            RSVD_eigval, RSVD_eigvec, RSVD_time = RSVD(A, N, d, q, MODE)
            vs2 += [RSVD_eigval]
            ts2 += [RSVD_time]
            RSVD_r = rounding(RSVD_eigvec, A, N, ROUND=ROUND)
            ps2 += [compute_Obj(A, RSVD_r, MODE)]
            df = df.append({"q":q, "dataset":dname, "algo":'RSVD', "R":vs2[-1]/classic_eigval, "eigval":vs2[-1], "obj":ps2[-1], "time":ts2[-1]}, ignore_index=True)
            # RandSum
            RSum_eigval, RSum_eigvec, RSum_time = RandSum(A, N, d, q, MODE)
            vs3 += [RSum_eigval]
            ts3 += [RSum_time]
            RSum_r = rounding(RSum_eigvec, A, N, ROUND=ROUND)
            ps3 += [compute_Obj(A, RSum_r, MODE)]
            df = df.append({"q":q, "dataset":dname, "algo":'RSum', "R":vs3[-1]/classic_eigval, "eigval":vs3[-1], "obj":ps3[-1], "time":ts3[-1]}, ignore_index=True)
        print(" | q={}, T={} |".format(q, T))
        print("(scipy.eigsh):\tR={:.1f}\tobj={:.1f}".format(1, classic_obj))
        print("(R-SVD):\tR={:.1f}\tobj={:.1f}".format(np.mean(vs2)/classic_eigval, np.mean(ps2)))
        print("(RandSum):\tR={:.1f}\tobj={:.1f}".format(np.mean(vs3)/classic_eigval, np.mean(ps3)))
        df.to_csv('SCG-q_d{}-{}.csv'.format(d, ROUND), index=False)
