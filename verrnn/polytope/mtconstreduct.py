import numpy as np
import multiprocessing
import functools
import operator
import time

"""
# --------------- to cope with the numerical erros ------------------- #
def remove_similar_constraints(coefs, b, epsilon) -> None:
    idxs_to_remove = []
    #print ('coefs=')
    #print (np.array(coefs))
    #print ('b=')
    #print (np.array(b))
    for i in range(len(coefs)):
        found = False
        for j in range(i+1, len(coefs)):
            if np.sum(np.abs(coefs[i] - coefs[j])) < epsilon and np.sum(np.abs( b[i] - b[j] )) < epsilon:
                idxs_to_remove.append(i)
                found = True
                break
        if found:
            continue
    idxs_to_remove.reverse()
    print ('to remove idx: ', idxs_to_remove)
    for i in idxs_to_remove:
        del coefs[i]
        del b[i]

def check_problematic_neg_constraints(coefs, b, epsilon) -> None:
    problematic_pair = []
    neg_coefs = [-l for l in coefs]
    neg_b = [-l for l in b]
    print (coefs, b, neg_coefs, neg_b)
    for i in range(len(coefs)):
        for j in range(i+1, len(coefs)):
            print (coefs[i], b[i], neg_coefs[j], neg_b[j])
            if np.sum(np.abs(coefs[i] - neg_coefs[j])) < epsilon and np.sum(np.abs( b[i] - neg_b[j] )) < epsilon:
                if np.sum(np.abs(coefs[i] - neg_coefs[j])) != 0 or  np.sum(np.abs( b[i] - neg_b[j] )) != 0:
                    problematic_pair.append((i,j))
                else:
                    #print ((i,j))
                    pass
    if problematic_pair:
        print ('problematic pair:',problematic_pair )
    for (i,j) in problematic_pair:
        coefs[j] = -coefs[i]
        b[j] = -b[i]

"""
# ------------------------------------------------------------------------------------------
# will hold the (implicitly mem-shared) data
data_array_coefs = None
data_array_b = None
data_epsilon = None

# child worker function
def job_handler_remove_similar_constraint(i):
    idxs_to_remove = []
    for j in range(i+1, len(data_array_coefs)):
        if np.sum(np.abs( data_array_b[i] - data_array_b[j] )) < data_epsilon and np.sum(np.abs(data_array_coefs[i] - data_array_coefs[j])) < data_epsilon:
            idxs_to_remove.append(i)
            break
    return idxs_to_remove

# -----------------------------------------------------------------------------------------

def launch_jobs_remove_similar_constraint(coefs, b, epsilon, num_worker=6):
    global data_array_coefs
    global data_array_b
    global data_epsilon
    data_array_coefs = coefs
    data_array_b = b
    data_epsilon = epsilon

    pool = multiprocessing.Pool(num_worker)
    res = pool.map(job_handler_remove_similar_constraint, range(len(coefs)))
    pool.close()

    data_array_coefs = None
    data_array_b = None
    data_epsilon = None
    return functools.reduce(operator.iconcat, res, [])

def remove_similar_constraints_parallel(coefs, b, epsilon) -> None:
    t0 = time.time()
    idxs_to_remove = launch_jobs_remove_similar_constraint(coefs, b, epsilon)
    idxs_to_remove.sort(reverse=True)
    for i in idxs_to_remove:
        del coefs[i]
        del b[i]
    t1 = time.time()
    if t1-t0 > 60:
        print ('remove_similar_constraints_parallel takes : ',t1-t0,'s')

# -----------------------------------------------------------------------------------------

data_array_neg_coefs = None
data_array_neg_b = None

def job_handler_check_problematic_neg_constraints(i):
    problematic_pair = []
    for j in range(i+1, len(data_array_coefs)):
        if np.sum(np.abs( data_array_b[i] - data_array_neg_b[j] )) < data_epsilon and np.sum(np.abs(data_array_coefs[i] - data_array_neg_coefs[j])) < data_epsilon:
            if np.sum(np.abs( data_array_b[i] - data_array_neg_b[j] )) != 0 or np.sum(np.abs(data_array_coefs[i] - data_array_neg_coefs[j])) != 0:
                problematic_pair.append((i,j))
            else:
                #print ((i,j))
                pass
    return problematic_pair

def launch_jobs_check_problematic_neg_constraints(coefs, b, epsilon, num_worker=6):
    global data_array_coefs
    global data_array_b
    global data_epsilon
    global data_array_neg_coefs
    global data_array_neg_b
    data_array_coefs = coefs
    data_array_b = b
    data_epsilon = epsilon
    data_array_neg_coefs = [-l for l in coefs]
    data_array_neg_b = [-l for l in b]

    pool = multiprocessing.Pool(num_worker)
    res = pool.map(job_handler_check_problematic_neg_constraints, range(len(coefs)))
    pool.close()
    data_array_coefs = None
    data_array_b = None
    data_epsilon = None
    data_array_neg_coefs = None
    data_array_neg_b = None
    return functools.reduce(operator.iconcat, res, [])

def check_problematic_neg_constraints_parallel(coefs, b, epsilon) -> None:
    t0 = time.time()
    problematic_pair = launch_jobs_check_problematic_neg_constraints(coefs, b, epsilon)
    if problematic_pair:
        print ('problematic pair:',problematic_pair )
        problematic_pair.sort()
        rjset = set([])
        for (i,j) in problematic_pair:
            if j in rjset: continue
            coefs[j] = -coefs[i]
            b[j] = -b[i]
            rjset.add(j)
    t1 = time.time()
    if t1-t0 > 60:
        print ('check_problematic_neg_constraints_parallel takes : ',t1-t0,'s')

# ---------------------------------------------------------------------------------------------


def remove_similar_points_serial(vs, epsilon) -> np.ndarray:
    """we assume vs is a ndarray: 1.  change to lst"""
    t0 = time.time()
    vs = list(vs)
    idxs_to_remove = []
    #print ('coefs=')
    #print (np.array(self.coefs))
    #print ('b=')
    #print (np.array(self.b))
    for i in range(len(vs)):
        found = False
        for j in range(i+1, len(vs)):
            if np.sum(np.abs(vs[i] - vs[j])) < epsilon:
                idxs_to_remove.append(i)
                found = True
                break
        if found:
            continue
    #print ('to remove idx: ', idxs_to_remove)
    idxs_to_remove.reverse()
    for i in idxs_to_remove:
        del vs[i]
    assert (len(vs) != 0)
    t1 = time.time()
    if t1-t0 > 30:
        print ('remove_similar_points_serial takes : ',t1-t0,'s')
    return np.array(vs)

def job_handler_remove_similar_points(i):
    idxs_to_remove = []
    for j in range(i+1, len(data_array_coefs)):
        if np.sum(np.abs(data_array_coefs[i] - data_array_coefs[j])) < data_epsilon:
            idxs_to_remove.append(i)
            break
    return idxs_to_remove

def launch_jobs_remove_similar_points(vs, epsilon, num_worker=6):
    global data_array_coefs
    global data_epsilon
    data_array_coefs = vs
    data_epsilon = epsilon

    pool = multiprocessing.Pool(num_worker)
    res = pool.map(job_handler_remove_similar_points, range(len(vs)))
    pool.close()

    data_array_coefs = None
    data_epsilon = None
    return functools.reduce(operator.iconcat, res, [])


def remove_similar_points_parallel(vs, epsilon) -> np.ndarray:
    t0 = time.time()
    vs = list(vs)
    idxs_to_remove = launch_jobs_remove_similar_points(vs, epsilon)
    idxs_to_remove.sort(reverse=True)
    #print (idxs_to_remove)
    for i in idxs_to_remove:
        del vs[i]
    assert (len(vs) != 0)
    t1 = time.time()
    if t1-t0 > 40:
        print ('remove_similar_points_parallel takes : ',t1-t0,'s')
    return np.array(vs)


# -----------------------------------------------------------------------------------------



def test():
    # create some random data and execute the child jobs
    coefs = [np.ones(10),np.ones(10),-np.ones(10)+0.0001, -np.ones(10)+0.0002]
    b = [1.0,1.0,-1.0, -1.0]
    remove_similar_constraints_parallel(coefs, b, 1)
    check_problematic_neg_constraints_parallel(coefs, b, 1)
    print (coefs, b)
    #coefs1 = list(np.random.rand(400,10))
    #b1 = list(np.random.rand(400))
    #remove_similar_constraints_parallel(coefs1, b1, 1) 
    
    coefs1 = np.random.rand(1000,10)
    coefs2 = coefs1.copy()
    remove_similar_points_parallel(coefs1,1)
    remove_similar_points_serial(coefs2,1)
    
    

if __name__ == "__main__":
    test()
