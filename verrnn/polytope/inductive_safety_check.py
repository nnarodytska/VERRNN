import smtbmc
from .pnt2hresp import get_pnt_idx_from_hull_Scipy
import numpy as np

def check_safety(pltp ,  weightsObj, ilb, iub, output_sign, nsteps = 3):
    slb,sub = pltp.get_range_box()
    if slb.shape[0] == 8:
        slb = slb[0:7]
    if sub.shape[0] == 8:
        sub = sub[0:7]
    range_estimator = smtbmc.RangeEstimation(weightsObj)
    #range_estimator.populate_polytope_state_range(50-Nlayer,slb, sub, stimulus_lower = ilb, stimulus_upper = iub, response_step = 30)
    range_estimator.populate_polytope_state_range(nsteps, state_ub_list = sub,state_lb_list = slb , stimulus_lower = ilb, stimulus_upper = iub, response_step = 30)
    oub, olb = range_estimator.get_previous_populate_output_status()
    #print ('  olb , oub = ',olb,oub)
    assert (oub >= olb-0.00001)  # epsilon
    #if stable and olb*oub >= 0 and olb*output_sign >= 0:
    if olb*oub >= 0 and olb*output_sign >= 0:
        return True # safe
    return False

# encode constraints in the form of 
# sat?  !( (A_c x <= B_c) -> (A_p x <= B_p) )

def check_inductiveness_smt(parent, child, precision):
    
    pass


def remove_similar_points(vs, epsilon):
    """we assume vs is a ndarray: 1.  change to lst"""
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
    return np.array(vs)

def remove_similar_points_j(vs, nstart, epsilon):
    """we assume vs is a ndarray: 1.  change to lst"""
    vs = list(vs)
    idxs_to_remove = []
    #print ('coefs=')
    #print (np.array(self.coefs))
    #print ('b=')
    #print (np.array(self.b))
    for i in range(nstart):
        for j in range(nstart, len(vs)):
            if np.sum(np.abs(vs[i] - vs[j])) < epsilon:
                idxs_to_remove.append(j)
    idxs_to_remove = list(set(idxs_to_remove))
    idxs_to_remove.sort(reverse = True)
    #print ('to remove idx: ', idxs_to_remove)
    for i in idxs_to_remove:
        del vs[i]
    assert (len(vs) != 0)
    return np.array(vs)

def check_inductiveness_qhull(parent_vertices, child_vertices):
    parent_vertices = remove_similar_points(parent_vertices, 1e-6)
    child_vertices = remove_similar_points(child_vertices, 1e-6)
    parent_idx = parent_vertices.shape[0]
    all_vertices = np.vstack([parent_vertices,child_vertices])
    all_vertices = remove_similar_points_j(all_vertices, parent_idx, 1e-6)
    all_idx = get_pnt_idx_from_hull_Scipy(all_vertices)
    inductiveness = (all_idx < parent_idx).all()

    #if not inductiveness:
    #    print ('---- DEBUG -------')
    #    print ('parent = ', parent_idx)
    #    print (parent_vertices)
    #    print ('child = ')
    #    print (child_vertices)
    #    print ('idx = ', all_idx)
    #    exit(1)

    return inductiveness



# do some test here:
def test_ics1():
    pv = np.array([[2.0,2.0,2.0],[2.0,0.0,2.0],[0.0,2.0,2.0],[0.0,0.0,2.0],[2.0,2.0,0.0],[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,0.0]])
    cv = np.array([[1.0,1.0,1.0],[1.0,0.0,1.0],[0.0,1.0,1.0],[0.0,0.0,1.0],[1.0,1.0,0.0],[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,0.0]])
    print (check_inductiveness_qhull(pv, cv))

def test_ics2():
    pv = np.array([[0.0,1.0],[1.0,3.0],[3.0,0.0]])
    cv = np.array([[1.0,1.0],[1.0,2.0],[2.0,1.0]])
    print (check_inductiveness_qhull(pv, cv))

def test_ics3():
    cv = np.array([[0.0,1.0],[1.0,3.0],[3.0,0.0]])
    pv = np.array([[1.0,1.0],[1.0,2.0],[2.0,1.0]])
    print (check_inductiveness_qhull(pv, cv))


if __name__ == "__main__":
    test_ics1()
    test_ics2()
