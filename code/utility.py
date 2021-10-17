import optparse
import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
from scipy.linalg import qr
from scipy.sparse.linalg import eigsh
import time

EPS=1E-10
INF=1E10

################################ Methods for finding the leading eigenvector ################################
def classic(A, n, MODE='adj'): # baseline
    st = time.time()
    _, eigvec = eigsh(A, k=1, which='LA')
    eigval = (compute_prod(A, eigvec, MODE).T).dot(eigvec)[0,0]
    ed = time.time()
    return eigval, eigvec, ed-st

def RandSum(A, n, d, q, MODE='adj'):
    st = time.time()
    n1 = d//2
    Omega = np.hstack((np.random.binomial(1, 0.5, (n,n1)), np.random.normal(0, 1, size=(n, d-n1))))
    Y = Omega.copy()
    for i in range(q): Y = compute_prod(A, Y, MODE)
    if Y.shape[1]>1:
        Q, _ = qr(Y, mode='economic')
        pA = (A.dot(Q).T).dot(Q)
        D, V = eigsh(pA, k=1, which='LA')
        D, u = D[0], np.matmul(Q,V)
    else: u = Y/norm(Y)
    ed = time.time()
    eigval = (compute_prod(A, u, MODE).T).dot(u)[0,0]
    return eigval, u, ed-st

def RSVD(A, n, d, q, MODE='adj'): # randomized SVD
    st = time.time()
    Omega = np.random.normal(0, 1, size=(n, d))
    Y = Omega.copy()
    for i in range(q): Y = compute_prod(A, Y, MODE)
    if Y.shape[1]>1:
        Q, _ = qr(Y, mode='economic')
        pA = (A.dot(Q).T).dot(Q)
        D, V = eigsh(pA, k=1, which='LA')
        D, u = D[0], np.matmul(Q,V)
    else: u = Y/norm(Y)
    ed = time.time()
    eigval = (compute_prod(A, u, MODE).T).dot(u)[0,0]
    return eigval, u, ed-st

################################ Read Input/Arguments ################################
def parse_arg():
    """ utility function to parse argument from command line """
    parser = optparse.OptionParser()
    parser.add_option('-T', dest='T', default='100', help='specify the repeated rounds')
    parser.add_option('-N', dest='N', default='3000', help='N: dimension of the matrix')
    (options, args) = parser.parse_args()
    return options

def read_dataset(path, undirected=True):
    """ read the input graph and cast into scipy.sparse.csr_matrix format """
    with open(path, "r") as f:
        data = f.read()
    N = int(data.split('\n')[0].split(' ')[1])
    # adjacency matrix
    A = sp.lil_matrix((N,N), dtype='d')
    for line in data.split('\n')[1:]:
        es = line.split('\t')
        if len(es)!=3: continue
        a, b, w = int(es[0]), int(es[1]), int(es[2])
        A[b,a] = w
        if undirected: A[a,b] = w
    return N, A.tocsr()

def compute_prod(A, Omega, MODE='adj'):
    """ matrix product with Omega """
    N = A.shape[0]
    Y = A.dot(Omega)
    if MODE == 'adj': # (signed) adjacency matrix
        return Y
    elif MODE == 'mod': # (signed) modularity matrix
        D_0 = A.sum(axis=1).A.reshape((N,1))
        D_1 = abs(A).sum(axis=1).A.reshape((N,1))
        D_p = (D_0+D_1)/2.0
        D_n = (D_1-D_0)/2.0
        Y_p = D_p * np.tile((D_p * Omega).sum(axis=0), (N,1)) / D_p.sum()
        if D_n.sum() > 0:
            Y_n = D_n * np.tile((D_n * Omega).sum(axis=0), (N,1)) / D_n.sum()
            return Y - (Y_p-Y_n)
        else: return Y - Y_p
    return Y

def compute_Obj(A, v, MODE='adj'): # Rayleigh quotient with v in task dependent set
    return (compute_prod(A, v, MODE).T).dot(v)[0,0] / (v*v).sum()

def gen_matrix(type, N=10000): # generate matrix with different spectral decays
    Sigma = get_eigvals(type, N=N)
    Q, _ = qr(np.random.normal(0, 1, size=(N, N)), mode='economic')
    A = Q@np.diag(Sigma)@(Q.T)
    return N, sp.csr_matrix(A)

def get_eigvals(type, N=10000):
    Sigma = []
    if type=='Type1':
        for i in range(N): Sigma += [(i+1)**(-1)]
    elif type=='Type2':
        for i in range(N): Sigma += [(i+1)**(-1/7)]
    elif type=='Type3':
        for i in range(N): Sigma += [(i+1)**(-1/3) if i<N*2/3 else -(i-N*2/3+1)**(-1)]
    elif type=='Type4':
        for i in range(N): Sigma += [(i+1)**(-1/2) if i<N/2 else -0.9*(i-N//2+1)**(-1/2)]
        for i in range(N-N//100,N): Sigma[i] = -0.9*(i+1)**(-0.01)
    else: None
    for i in range(1,N//100): Sigma[i] = (i+1)**(-0.01)
    return Sigma


################################ Rounding Algorithms ################################
def rounding(v, A, N, ROUND="R"):
    if ROUND == "R":
        return round_by_randomized_vector(v, 1, -1, A, N)
    if ROUND == "S":
        return round_by_sign(v, N)
    return None

def round_by_sign(v, N): # rounding by the sign of the entry
    return np.array([np.sign(x) for x in v.reshape((-1))]).reshape((N,1))

def round_by_randomized_vector(v_in, pos, neg, A, N): # randomized proportional to the magnitude
    """ sample a randomized vector T in {0,-1,z}^N given the specific vector v or -v """
    def randomized_vector(v):
        def bernoulli_sample(x):
            if x>0: return pos*np.random.choice([0,1], 1, p=[max(1.0-x/pos,0),min(x/pos,1.0)])[0]
            elif x<0: return neg*np.random.choice([0,1], 1, p=[max(1.0-x/neg,0),min(x/neg,1.0)])[0]
            else: return 0
        v *= np.abs(v).sum()
        T = np.array([bernoulli_sample(v[i]) for i in range(N)])
        return T
    v_in = v_in.reshape((-1))
    return randomized_vector(v_in).reshape((N,1))
