import numpy as np
import cdd
from scipy.spatial import ConvexHull
from .qhull_local_interface import run_qhull

External_Qhull_Path = '/home/hongce/summer19/qhull-2019.1/build/qhull'

epsilon = 1e-5 # please adjust this for finding dimensions

def toHullScipy(low_dim_pts_reduct, timeout, externQhull=False):
    if low_dim_pts_reduct.shape[1] == 1:
        lb = np.min(low_dim_pts_reduct)
        ub = np.max(low_dim_pts_reduct)
        # x <= ub  -x >= -lb
        A = np.array([[1.0],[-1.0]])
        b = np.array([ub, -lb])
    else:
        # convex hull will get you Ax + b <= 0
        #try:
        if not externQhull:
            hull = ConvexHull(points=low_dim_pts_reduct)
            equations = hull.equations
        else:
        #except:
        #    print ('W: Qhull failed, turn on Q14 (merge) option')
            #equations = run_qull(low_dim_pts_reduct, '/home/hongce/data/qhull-2019.1/build/qhull', 'Qx Qt')
            equations = run_qhull(low_dim_pts_reduct, External_Qhull_Path, 'Qx Qt', timeout=timeout)

        A = equations[:,:-1]
        b = -equations[:,-1]
    return A,b



def toHullIdxScipy(low_dim_pts_reduct):
    if low_dim_pts_reduct.shape[1] == 1:
        lb_idx = np.argmin(low_dim_pts_reduct)
        ub_idx = np.argmax(low_dim_pts_reduct)
        return [lb_idx, ub_idx]
    else:
        # convex hull will get you Ax + b <= 0
        try:
            hull = ConvexHull(points=low_dim_pts_reduct)
        except:
            print ('W: Qhull failed, turn on QJ (Joggle) option')
            hull = ConvexHull(points=low_dim_pts_reduct, qhull_options = 'QJ')
        return hull.vertices


def toHullCdd(low_dim_pts_reduct):
    if low_dim_pts_reduct.shape[1] == 1:
        lb = np.min(low_dim_pts_reduct)
        ub = np.max(low_dim_pts_reduct)
        # x <= ub  -x >= -lb
        A = np.array([[1.0],[-1.0]])
        b = np.array([ub, -lb])
    else:
        npoints = low_dim_pts_reduct.shape[0]
        low_dim_pts_reduct = np.hstack([np.ones((npoints,1)),low_dim_pts_reduct])
        mat = cdd.Matrix(low_dim_pts_reduct, number_type='float')
        mat.rep_type = cdd.RepType.GENERATOR
        P = cdd.Polyhedron(mat)
        H = P.get_inequalities()
        H = np.array(H)
        A = -H[:,1:]
        b = H[:,0]
    return A,b


def FullDimension(v):
    _, ndim = v.shape
    return np.linalg.matrix_rank(v-v[0]) == ndim

def pnt2hresp_Scipy(v, timeout:int, externQhull=False):
    if FullDimension(v):
        A,b = toHullScipy(v, externQhull = externQhull, timeout = timeout)
        return A,b
    m_ = v[0]
    v  = v - m_ # m_ will be the origin
    U,S,V = np.linalg.svd(v)
    npoints, ndim = v.shape
    
    if not (np.abs(S) < epsilon).any():
        dim_to_remove = S.shape[0]
    else:
        dim_to_remove = np.argmax(np.abs(S) < epsilon)
    #print ( 'QHull # dim =' , dim_to_remove)
    
    diag_l = S.shape[0]
    reconstruct_diag = np.zeros((npoints, ndim))
    reconstruct_diag[:diag_l,:diag_l] = np.diag(S)

    new_pts = np.matmul(U,reconstruct_diag)
    low_dim_pts_reduct = new_pts[:, :dim_to_remove]

    if dim_to_remove != 0:
        #try:
        A,b = toHullScipy(low_dim_pts_reduct, externQhull = externQhull,  timeout = timeout)
        #except Exception as e:
        #    print ('--------- DEBUG ----------')
        #    print ("S = ")
        #    print (S)
        #    print ('dim_to_remove = ', dim_to_remove)
        #    print ("Points = ")
        #    for vp in low_dim_pts_reduct:
        #        print (vp)
        #    raise e

        # complete to the full dimension
        n_rows = A.shape[0]
        n_cols = A.shape[1]

        num_var_to_add = ndim - (dim_to_remove)

        zero4 = np.zeros((n_rows,num_var_to_add))
        rows_to_append = np.zeros((2*num_var_to_add, n_cols+num_var_to_add))
        for idx in range(num_var_to_add):
            rows_to_append[idx*2,n_cols+ idx] = 1
            rows_to_append[idx*2+1,n_cols+ idx] = -1

        trueA = np.vstack([np.hstack([A, zero4]), rows_to_append])
        trueB = np.append(b, [0.0, 0.0]*num_var_to_add)
        #trueB = np.append(b, [epsilon, epsilon]*num_var_to_add)
    else:
        #num_var_to_add = ndim # v.shape[1] - (dim_to_remove)  # v.shape[1] 
        # directly construct trueA trueB
        trueA = np.zeros((ndim*2, ndim))
        for idx in range(ndim):
            trueA[idx*2,idx] = 1
            trueA[idx*2+1,idx] = -1
        trueB = np.zeros(ndim*2)

    finalA = np.matmul(trueA, V)
    finalB = np.matmul(finalA, m_) + trueB
    return finalA, finalB



def get_pnt_idx_from_hull_Scipy(v):
    if FullDimension(v):
        print ('F')
        idxs = toHullIdxScipy(v)
    else:
        m_ = v[0]
        v  = v - m_ # m_ will be the origin
        U,S,_ = np.linalg.svd(v)
        npoints, ndim = v.shape
        
        if not (np.abs(S) < epsilon).any():
            dim_to_remove = S.shape[0]
        else:
            dim_to_remove = np.argmax(np.abs(S) < epsilon)
        #print (dim_to_remove)
        
        diag_l = S.shape[0]
        reconstruct_diag = np.zeros((npoints, ndim))
        reconstruct_diag[:diag_l,:diag_l] = np.diag(S)

        new_pts = np.matmul(U,reconstruct_diag)
        low_dim_pts_reduct = new_pts[:, :dim_to_remove]
        idxs = toHullIdxScipy(low_dim_pts_reduct)
    return idxs

 
def pnt2hresp_Cdd(v):
    if FullDimension(v):
        print ('F')
        return toHullCdd(v)
    m_ = v[0]
    v  = v - m_ # m_ will be the origin
    U,S,V = np.linalg.svd(v)
    npoints, ndim = v.shape
    
    if not (np.abs(S) < epsilon).any():
        dim_to_remove = S.shape[0]
    else:
        dim_to_remove = np.argmax(np.abs(S) < epsilon)
    #print (dim_to_remove)
    
    diag_l = S.shape[0]
    reconstruct_diag = np.zeros((npoints, ndim))
    reconstruct_diag[:diag_l,:diag_l] = np.diag(S)

    new_pts = np.matmul(U,reconstruct_diag)
    low_dim_pts_reduct = new_pts[:, :dim_to_remove]
    
    if dim_to_remove != 0:
        A,b = toHullCdd(low_dim_pts_reduct)
        # complete to the full dimension
        n_rows = A.shape[0]
        n_cols = A.shape[1]

        num_var_to_add = ndim - (dim_to_remove)

        zero4 = np.zeros((n_rows,num_var_to_add))
        rows_to_append = np.zeros((2*num_var_to_add, n_cols+num_var_to_add))
        for idx in range(num_var_to_add):
            rows_to_append[idx*2,n_cols+ idx] = 1
            rows_to_append[idx*2+1,n_cols+ idx] = -1

        trueA = np.vstack([np.hstack([A, zero4]), rows_to_append])
        trueB = np.append(b, [0.0, 0.0]*num_var_to_add)
        #trueB = np.append(b, [epsilon, epsilon]*num_var_to_add)
    else:
        #num_var_to_add = ndim # v.shape[1] - (dim_to_remove)  # v.shape[1] 
        # directly construct trueA trueB
        trueA = np.zeros((ndim*2, ndim))
        for idx in range(ndim):
            trueA[idx*2,idx] = 1
            trueA[idx*2+1,idx] = -1
        trueB = np.zeros(ndim*2)

    finalA = np.matmul(trueA, V)
    finalB = np.matmul(finalA, m_) + trueB
    return finalA, finalB


def test():
  v1 = np.array([[0.0, 0.0,0.0,0.0],[1.0, 1.0,1.0,1.0], [1.0/2, 1.0/2,1.0/2,1.0/2], [1.0/4, 1.0/4,1.0/4,1.0/4]])
  v2 = np.array([[0.0, 0.0,0.0,0.0],[1.0, 1.0,1.0,1.0]]) # good
  v3 = np.array([[1.0, 1.0,0.0,1.0],[1.0, 0.0,2.0,1.0]])
  v4 = np.array([[1.0,0.0,1.0],[ 0.0,2.0,1.0], [ 0.0,0.0,1.0], [ 1.0,2.0,1.0]])
  v5 = np.zeros((2,4))
  v6 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])

  for v in [v1,v2,v3,v4,v5]:
    assert (not FullDimension(v))
  assert (FullDimension(v6))

  for v in [v1, v2, v3 ,v4, v5, v6]:
    print ('------------------')
    print ('v:', v)
    A, b = pnt2hresp_Scipy(v, timeout = 30, externQhull=False)

    A[np.abs(A) < epsilon] = 0.0
    b[np.abs(b) < epsilon] = 0.0

    print (A)
    print (b)

    A, b = pnt2hresp_Cdd(v)

    A[np.abs(A) < epsilon] = 0.0
    b[np.abs(b) < epsilon] = 0.0

    print (A)
    print (b)


def test2():
  v6 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])
  print (toHullCdd(v6))
  print (toHullScipy(v6))

if __name__ == '__main__':
  test()
    
    
