from .ilp import cplex, ilp_encoder, np
import cdd
import math
import time
from typing import Sequence, List
from .mtconstreduct import check_problematic_neg_constraints_parallel, remove_similar_constraints_parallel, remove_similar_points_parallel
from .pnt2hresp import pnt2hresp_Scipy, pnt2hresp_Cdd
from .timeoutalarm import TimeoutException

# number_type = 'float' or 'fraction'
def compute_polytope_vertices(A, b, number_type):
    """
    Compute the vertices of a polytope given in halfspace representation by
    :math:`A x \\leq b`.
    Parameters
    ----------
    A : array, shape=(m, k)
        Matrix of halfspace representation.
    b : array, shape=(m,)
        Vector of halfspace representation.
    Returns
    -------
    vertices : list of arrays
        List of polytope vertices.
    """
    b = b.reshape((b.shape[0], 1))
    mat = cdd.Matrix(np.hstack([b, -A]), number_type=number_type)
    mat.rep_type = cdd.RepType.INEQUALITY
    P = cdd.Polyhedron(mat)
    g = P.get_generators()
    V = np.array(g)
    vertices = []
    #print (g)
    for i in range(V.shape[0]):
        if V[i, 0] != 1:  # 1 = vertex, 0 = ray
            #print ('---- DEBUG ----')
            #print ('A = ')
            #print (A)
            #print ('b = ')
            #print (b)
            #print ('V = ')
            #print (V)
            raise RuntimeError("Polyhedron is not a polytope")  # Exception
        elif i not in g.lin_set:
            if number_type == 'float':
                vertices.append(V[i, 1:])
            else:
                vertices.append([float(v) for v in V[i, 1:]])
        else:
            print ('i in g.lin_set')
    return vertices

def pts_remove_small_coef(vs, epsilon) -> np.ndarray:
    vs[np.abs(vs) < epsilon] = 0.0
    return vs

class dummydataholder(object):
    def __init__(self, num_state, num_input, W_rec, W_in, b_rec):
        self.num_state = num_state
        self.num_input = num_input
        self.W_rec = W_rec
        self.W_in = W_in
        self.b_rec = b_rec

class polytopeAttributes(object):
    def __init__(self):
        self.has_bound_box_constraints = False
        self.is_abstract = False
        self.has_input_constraints = 0 # 1 : yes, 2 : unknown
        self.abs_poly_ref = None
        self.abs_poly_layer = None
    def copy(self):
        tmp = polytopeAttributes()
        tmp.has_bound_box_constraints = self.has_bound_box_constraints
        tmp.is_abstract = self.is_abstract
        tmp.has_input_constraints = self.has_input_constraints
        tmp.abs_poly_layer = self.abs_poly_layer
        tmp.abs_poly_ref = self.abs_poly_ref
        return tmp

class convex_polytope(object):
    def __init__(self, vars : Sequence[str]) -> None:
        self.vars = vars # s0-s6, x0
        self.coefs : Sequence[np.ndarray] = []
        self.b : Sequence[float] = []
        self.vertices_holder : Sequence[np.ndarray] = []
        # attributes
        self.attribute = polytopeAttributes()
        self.no_input_vs = None
    
    def copy(self):
        cvp = convex_polytope(self.vars)
        cvp.coefs = self.coefs[:]
        cvp.b = self.b[:]
        cvp.vertices_holder = []
        cvp.attribute = self.attribute.copy()
        return cvp

    # H-representations
    def add_vi_range(self, idx,lb, ub) -> None: # you can use it for vi == ??
        assert (idx >= 0 and idx <= len(self.vars))
        assert (lb <= ub)
        coef = [0.0] * len(self.vars)
        coef[idx] = 1.0
        self.coefs.append(coef[:])
        self.b.append(ub)

        coef[idx] = -1.0
        self.coefs.append(coef[:])
        self.b.append(-lb)
        # clear vertices holder
        self.vertices_holder = []
        # it has
        self.attribute.has_input_constraints = True

    def add_relu_constraints(self, relu_binary_indicator, Wb, b_rec ) -> None:
        assert (len(self.vars) == len(relu_binary_indicator) + 1)
        assert (len(b_rec) == len(relu_binary_indicator))
        for idx in range(len(relu_binary_indicator)):
            coef = Wb[:,idx].flatten()#.tolist()
            assert (coef.shape[0] == len(self.vars))
            if relu_binary_indicator[idx] < 0.5: # == 0
                self.coefs.append(coef)
                self.b.append(-b_rec[idx])
            else: # == 1.0 
                self.coefs.append(-coef)
                self.b.append(b_rec[idx])
        # clear vertices holder
        self.vertices_holder = []
        self.no_input_vs = None

    def print_eqs(self) -> None:
        assert (len(self.b) == len(self.coefs))
        num_eqs = len(self.b)
        #print (self.coefs)
        #print (self.b)
        for idx in range(num_eqs):
            coef =  self.coefs[idx]
            b = self.b[idx]
            assert (len(coef) == len(self.vars))
            for vidx in range(len(coef)):
                print (coef[vidx], '*',self.vars[vidx], end = ' + ')
            print ('<=', b)

    def dump_to_file(self, f) -> None:
        np.savez(f, **{'vars':self.vars, 'coefs':self.coefs, 'b':self.b, \
            'is_abstract': self.attribute.is_abstract, \
            'has_bound_box_constraints':self.attribute.has_bound_box_constraints, \
            'has_input_constraints':self.attribute.has_input_constraints, \
            'abs_poly_layer':self.attribute.abs_poly_layer, \
            'ref_vars':self.attribute.abs_poly_ref.vars if self.attribute.is_abstract and self.attribute.abs_poly_ref is not None else None , \
            'ref_coefs':self.attribute.abs_poly_ref.coefs if self.attribute.is_abstract and self.attribute.abs_poly_ref is not None else None,
            'ref_b':self.attribute.abs_poly_ref.b if self.attribute.is_abstract and self.attribute.abs_poly_ref is not None else None, \
            'ref_is_abstract': self.attribute.abs_poly_ref.attribute.is_abstract if self.attribute.is_abstract and self.attribute.abs_poly_ref is not None else None, \
            'ref_has_bound_box_constraints':self.attribute.abs_poly_ref.attribute.has_bound_box_constraints if self.attribute.is_abstract and self.attribute.abs_poly_ref is not None else None, \
            'ref_has_input_constraints':self.attribute.abs_poly_ref.attribute.has_input_constraints if self.attribute.is_abstract and self.attribute.abs_poly_ref is not None else None, \
            'ref_abs_poly_layer':self.attribute.abs_poly_ref.attribute.abs_poly_layer if self.attribute.is_abstract and self.attribute.abs_poly_ref is not None else None \
            })
            
    def load_from_file(self,f) -> None:
        vals = np.load(f, allow_pickle=True)
        self.vars = vals['vars']
        self.coefs = list(vals['coefs'])
        self.b = list(vals['b'])
        self.attribute.is_abstract = vals['is_abstract']
        self.attribute.has_bound_box_constraints = vals['has_bound_box_constraints']
        self.attribute.has_input_constraints = vals['has_input_constraints']
        self.attribute.abs_poly_layer = vals['abs_poly_layer']
        if self.attribute.is_abstract:
            ref_vars = vals['ref_vars']
            self.attribute.abs_poly_ref = convex_polytope(ref_vars)
            self.attribute.abs_poly_ref.coefs = list(vals['ref_coefs'])
            self.attribute.abs_poly_ref.b = list(vals['ref_b'])
            self.attribute.abs_poly_ref.attribute.is_abstract = vals['ref_is_abstract']
            self.attribute.abs_poly_ref.attribute.has_bound_box_constraints = vals['ref_has_bound_box_constraints']
            self.attribute.abs_poly_ref.attribute.ref_has_input_constraints = vals['ref_has_input_constraints']
            
            
        
        print (self.attribute)
        print (isinstance(self.attribute, polytopeAttributes))
        print (self.attribute.shape)
        print (self.attribute.has_input_constraints)
        assert (len(self.b) == len(self.coefs))
        assert (np.array(self.coefs).shape[1] == len(self.vars))

    # V-representation
    def get_vertices(self, assert_no_buffered_v = False, assert_has_buffered_v = False, force_no_point_check = False) -> List[np.ndarray]:
        #self.print_eqs()
        if assert_no_buffered_v:
            assert (len(self.vertices_holder) == 0)
        if assert_has_buffered_v:
            assert (len(self.vertices_holder) != 0)

        if len(self.vertices_holder) != 0: # a buffer
            return self.vertices_holder

        if not self.attribute.has_input_constraints:
            print ('Warning: no input constraint!')
        
        t0 = time.time()
        
        precision = 0
        useFractionFlag = False
        succeed = False
        while not succeed:
            try:
                if not useFractionFlag and precision != 0:
                    self.truncate_precision(2**precision)
                vs = compute_polytope_vertices(np.array(self.coefs), np.array(self.b), number_type = 'fraction' if useFractionFlag else 'float')
                #if len(vs) == 0 and precision != 10:
                if force_no_point_check and len(vs) == 0:
                    print ('-------------------- ERROR no points!')
                    raise RuntimeError
                #if precision != 10 and len(vs) == 0:
                #    print ('-------------------- ERROR no points!')
                #    raise RuntimeError
                succeed = True
            except RuntimeError as runtime:
                if useFractionFlag:
                    print ('---- DEBUG ----')
                    print ('A = ')
                    print (np.array(self.coefs))
                    print ('b = ')
                    print (self.b)
                    self.dump_to_file('polytope.dump')
                    raise runtime
                elif precision > 0 and precision < 11:
                    useFractionFlag = True
                    print ('(!!!UseFraction!!!)', end = '', flush=True)
                elif precision == 0:
                    precision = 16
                else:
                    precision = precision - 1 
                succeed = False
                print ('(precision:',precision,')', end = ' ', flush=True)
            except Exception as e:
                print ('polytope dumped to polytope.dump')
                self.dump_to_file('polytope.dump')
                raise e
        t1 = time.time()
        if t1 - t0 > 60: print ('cdd takes ', t1-t0,'s')

        self.vertices_holder = vs
        if len(vs) != 0 and not self.check_points(vs, epsilon=1e-5):
            print ("points not in")
            raise Exception("points not in")
        return vs



    def from_vertices(self, vertices, num_input : int, FacetApproxBound : int, ReduceDupFacet : bool, RdFacetBound : int, externQhull:bool, timeout:int) -> None: # the number of input variables to append
        """
        if the #facet > FacetApproxBound, will raise exception
        if ReduceDupFacet and #facet < RdFacetBound, will reduce bound, else do nothing
        """
        assert (len(self.coefs) == 0) # you must start from empty
        assert (len(self.b) == 0)
        vertices = remove_similar_points_parallel(vertices, 1e-5)
        t1 = time.time()
        A, b = pnt2hresp_Scipy(v = np.array(vertices), externQhull = externQhull, timeout=timeout)
        t2 = time.time()
        if t2-t1 > 30: print ('qhull takes ', t2-t1, 's')

        neqs = A.shape[0]
        A = np.hstack([A, np.zeros((neqs, num_input))])
        self.coefs = list(A)
        self.b = list(b)
        if len(self.coefs) > 1000:
            print ('W: many facets before #:', len(self.coefs))
        self.clear_small_coef(1e-6)
        if ReduceDupFacet and ( RdFacetBound == 0 or RdFacetBound > len(self.coefs) ):
            remove_similar_constraints_parallel(self.coefs, self.b, 1e-6)
            check_problematic_neg_constraints_parallel(self.coefs, self.b, 1e-6)
        else:
            print ('Will not reduce facet, # facet (%d) > # rdBound (%d)' %(len(self.coefs) , RdFacetBound) )
        if len(self.coefs) > 1000:
            print ('W: many facets after #:', len(self.coefs))
        if len(self.coefs) > FacetApproxBound and FacetApproxBound != 0:
            msg = "Will not construct exact polytope for facet (%d) > FacetApproxBound(%d) " % (len(self.coefs), FacetApproxBound)
            print (msg)
            raise Exception(msg)

        self.vertices_holder = vertices
        self.no_input_vs = vertices

        self.attribute.has_input_constraints = False
        self.attribute.has_bound_box_constraints = False
        
        if len(vertices) != 0 and not self.check_points(np.hstack([vertices,np.zeros((vertices.shape[0], num_input))]), epsilon=1e-5):
            print ("points not in")
            raise Exception("points not in")

    def is_null(self) -> bool:
        if len(self.vertices_holder) == 0:
            self.get_vertices(force_no_point_check=True)
        if len(self.vertices_holder) == 0:
            print (' $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ Unexpected Null')
            raise Exception('Unexpected Null')
        return len(self.vertices_holder) == 0
        

    # --------------- to cope with the numerical erros ------------------- #
    def get_range_box(self) -> np.ndarray:
        if self.no_input_vs is not None:
            slb = np.amin(self.no_input_vs, axis=0)
            sub = np.amax(self.no_input_vs, axis=0)
            return slb,sub
            
        if len(self.vertices_holder) == 0:
            self.get_vertices()
        slb = np.amin(self.vertices_holder, axis=0)
        sub = np.amax(self.vertices_holder, axis=0)
        #print (self.vertices_holder)
        #print (slb)
        #print (sub)
        return slb,sub


    
    def clear_small_coef(self, epsilon) -> None:
        tmp_b = np.array(self.b)
        tmp_b[np.abs(tmp_b) < epsilon] = 0.0
        self.b = list(tmp_b)

        tmp_coef = np.array(self.coefs)
        tmp_coef[np.abs(tmp_coef) < epsilon] = 0.0
        self.coefs = list(tmp_coef)

    def truncate_precision(self, precision) -> None:
        new_coefs = []
        for coef in self.coefs: # make more points into the polytope
            new_coefs.append([math.floor(x*precision)/float(precision) for x in coef])
        self.coefs = list(np.array(new_coefs))
        self.b = [math.ceil(x*precision)/float(precision) for x in self.b]

    def check_points(self, vs, epsilon = 0.0) -> bool:
        if isinstance(vs, list):
            vs = np.array(vs)
        assert (vs.shape[1] == len(self.vars))
        A = np.array(self.coefs)
        lhs = np.matmul(A, vs.T)
        for idx in range(lhs.shape[1]):
            check = lhs[:,idx] <= np.array(self.b) + epsilon
            if not check.all():
                print ('Points not in!')
                print ('--- DEBUG ---')
                print ('A = ')
                print (A)
                print ('lhs = ')
                print (lhs[:,idx])
                print ('violation point #%d = ' % idx)
                print (vs[idx, :])
                print ('b = ')
                print (self.b)
                print ('polytope Dumped to point_not_in')
                self.dump_to_file('point_not_in')
                print ('pnts dumpt in not_in_point')
                np.savez("not_in_point" ,vs)
                return False
        return True
    
    def check_points_in(self, vs, num_input = 1, epsilon = 0.0, polytope_no_input_constr = False) -> bool:
        A = np.array(self.coefs)
        if polytope_no_input_constr:
            assert (np.sum(np.abs(A[:,-1*num_input])) == 0.0)
            smallA = A[:,0:-1*num_input]
            smallB = np.array(self.b)
        else:
            assert (np.sum(np.abs(A[:-2*num_input,-1*num_input])) == 0.0)
            assert (np.sum(np.abs(A[-2*num_input:,:-1*num_input])) == 0.0)
            smallA = A[0:-2*num_input,0:-1*num_input]
            smallB = np.array(self.b[0:-2*num_input])

        if isinstance(vs, list):
            vs = np.array(vs)

        assert (vs.shape[1] == len(self.vars)-num_input) # we are extending correctly

        lhs = np.matmul(smallA, vs.T)
        for idx in range(lhs.shape[1]):
            check = lhs[:,idx] <= np.array(smallB) + epsilon
            if not check.all():
                return False
        return True


class polytopes(object):
    def __init__(self, dataholder, input_idx = 0):
        self.num_state = dataholder.num_state
        self.num_input = 1 # dataholder.num_input
        self.dataholder= dataholder # the true dataholder (w.o. change on num_input)
        self.W_rec = dataholder.W_rec
        self.W_in = dataholder.W_in[input_idx] # for the first input
        self.b_rec = dataholder.b_rec
        self.init_state = dataholder.init_state
        self.holder_for_ilp = dummydataholder( \
            num_state = self.num_state, num_input = self.num_input, W_rec = self.W_rec , W_in = np.array([self.W_in]) , b_rec = dataholder.b_rec )
        # the variables are either s0-s6 or s0-s6,x0
        # we should compute Wb matrix
        self.Wb = np.vstack([self.W_rec, self.W_in])
        self.vars = ['s%d' % s for s in range(self.num_state)] + ['x%d' % x for x in range(self.num_input)]

        

        #self.W_in1 = dataholder.W_in[1]
    
    def _transform_matrix(self,relu_binary_indicator ):
        W_rec = (self.Wb * (np.array(relu_binary_indicator) >= 0.5))  #(== 1.0)
        b_rec = (self.b_rec * (np.array(relu_binary_indicator) >= 0.5))
        return W_rec, b_rec

    def merge_polytopes_by_convex_hull(self, vs : Sequence[np.ndarray], FacetApproxBound : int, ReduceDupFacet : bool, RdFacetBound : int, externQhull : bool, timeout:int):
        #vs = np.vstack(vs)
        vs = remove_similar_points_parallel(vs, epsilon = 0.001)
        t1 = time.time()
        vs = pts_remove_small_coef(vs, epsilon = 1e-6)
        t2 = time.time()

        if t2-t1 > 30: print ('pts_remove_small_coef takes ', t2-t1, 's')

        # here we create convex hull of them
        new_pltp = convex_polytope(self.vars)
        try:
            new_pltp.from_vertices(vs, self.num_input, FacetApproxBound, ReduceDupFacet, RdFacetBound, externQhull, timeout=timeout)
        except Exception:
            print ('Will loosen instead of construct exact convex hull')
            raise RuntimeError
        new_pltp.attribute.is_abstract = True
        return new_pltp, vs
    
    def loosen_poly_based_on_vertices(self, pltp, vs, include_old_points = False):
        if isinstance(vs, list):
            vs = np.array(vs)
        assert (vs.shape[1] == len(self.vars)-self.num_input) # we are extending correctly
        
        old_vl = pltp.get_vertices()
        old_v = np.array(old_vl)
        if include_old_points and len(old_vl) > 0:
            if old_v.shape[1] > self.num_state:
                old_v = old_v[:,:-1]
            all_v = np.vstack([old_v, vs])
        else:
            all_v = vs
        #all_v = remove_similar_points_parallel(all_v, epsilon = 1e-5 )
        # compute bound box constraints
        vlb = np.amin(all_v,axis = 0)
        vub = np.amax(all_v,axis = 0)
        bbox_A = np.vstack([np.eye(self.num_state), -np.eye(self.num_state)])
        bbox_B = np.hstack([vub, -vlb]) # this is to add an additional bound box

        newp = pltp.copy()
        A = np.array(newp.coefs)
        # it must have been supplied with input constraints
        assert (pltp.attribute.has_input_constraints)
        assert (np.sum(np.abs(A[:-2*self.num_input,-1*self.num_input])) == 0.0)
        assert (np.sum(np.abs(A[-2*self.num_input:,:-1*self.num_input])) == 0.0)
        smallA = A[0:-2*self.num_input,0:-1*self.num_input]
        smallB = np.array(newp.b[0:-2*self.num_input])
        
        lhs = np.matmul(smallA, vs.T)
        lhs_max = np.amax(lhs, axis = 1)
        
        mask = lhs_max > smallB
        if include_old_points:
            new_rhs = lhs_max*mask + (1-mask)*smallB
        else:
            new_rhs = lhs_max
        
        # bound box constraint added here:
        smallA = np.vstack([smallA, bbox_A])
        new_rhs = np.hstack([new_rhs, bbox_B])
        #print (new_rhs, bbox_B)

        assert (lhs_max.shape[0] == len(pltp.b) - 2*self.num_input)
        #print ('adjust ', newp.b, '--->', new_rhs, 'removing input cs')

        neqs = smallA.shape[0]
        newp.coefs = list(np.hstack([smallA, np.zeros((neqs, self.num_input))]))
        newp.b = list(new_rhs)
        newp.vertices_holder = all_v # to avoid compute without input constraints
        newp.no_input_vs = all_v
        remove_similar_constraints_parallel(newp.coefs, newp.b,1e-6)
        newp.attribute.has_input_constraints = False
        newp.attribute.is_abstract = True
        if not newp.check_points_in(all_v,num_input = self.num_input,epsilon = 1e-6, polytope_no_input_constr = True):
            print ('Error!')
            exit(1)
        return newp        

    def propagate_split_polytope(self, pltp, ilb, iub, FacetApproxBound : int, ReduceDupFacet : bool, RdFacetBound : int, externQhull:bool , qhullTimeout : int, toPointsOnly = False):
        # if toPointsOnly : will only return vertices
        # 1. check for possible relu assignment
        # if FacetApproxBound == 0, will always construct exact polytope
        # if FacetApproxBound != 0 and #facet > FacetApproxBound, will not construct but raise exception
        # if ReduceDupFacet and # facet < RdFacetBound, will reduce the facet, ow. do nothing
        assert (ilb <= iub)
        BSols = []
        solve_idx = 0
        while True:

            ilp_enc = ilp_encoder(self.holder_for_ilp)
            solver = cplex.Cplex()
            solver.set_log_stream(None)
            solver.set_error_stream(None)
            solver.set_warning_stream(None)
            solver.set_results_stream(None)
            SI, XI , _, _, B = ilp_enc.encode_single_frame(0, solver, ilb, iub)
            # 1.2 add currently polytope's restrictions
            assert (len(pltp.b) == len(pltp.coefs))
            lin_expr = []
            for idx in range(len(pltp.coefs)):
                val = [float(v) for v in pltp.coefs[idx]]
                lin_expr.append( cplex.SparsePair(ind = SI + XI, val = val) )
            rhs = [float(v) for v in pltp.b]
            solver.linear_constraints.add(lin_expr=lin_expr,senses = 'L' * len(pltp.coefs), rhs = rhs )

            for bs in BSols:
                ilp_enc.encode_block_binary_assignment(solver, B, bs)
            #solver.solve()
            #print (solver.solution.get_values())

            solver.populate_solution_pool()
            num_sols = solver.solution.pool.get_num()

            if num_sols == 0:
                break
            localSolSet = set([])
            for idx in range(num_sols):
                sol = solver.solution.pool.get_values(idx, B)
                localSolSet.add(tuple(sol))
            localSol = list(localSolSet)
            BSols += localSol
            solve_idx += 1

        if len(BSols) == 0:
            print ('No relu assignment can be found! This is probably a bug!')
            pltp.dump_to_file('polytope.dump')
            solver.write('./ilp_polytope_relu_assign_cplex.lp')
            # assert (False)  # no solution found ...

        #print ('--> relu assignments:', BSols)

        new_pltps = []
        all_vertices = []
        # BSols contains all feasible relu assignments
        Exception_tmp = None
        for relu_assign in BSols:
            sub_pltp = pltp.copy()
            sub_pltp.add_relu_constraints(relu_assign, self.Wb, self.b_rec)
            vs = sub_pltp.get_vertices(assert_no_buffered_v=True)
            #print (vs)
            if len(vs) == 0:
                #print ('W: numerical difference between cplex and Cdd')
                continue
            # do the transformation
            W,b = self._transform_matrix(relu_assign)
            #print (W)
            #print (b)
            new_vertices = np.matmul (vs, W) + b
            #print (new_vertices)
            #print ('--> :',vs, ' =====>> ',  new_vertices)
            if (toPointsOnly):
                all_vertices.append(new_vertices)
            else:
                new_pltp = convex_polytope(self.vars)
                try:
                    new_pltp.from_vertices(new_vertices, self.num_input, FacetApproxBound, ReduceDupFacet, RdFacetBound, externQhull=externQhull, timeout=qhullTimeout)
                except Exception as e:
                    #print ('. # sol =', len(BSols))
                    print (e)
                    Exception_tmp = e
                    all_vertices.append(new_vertices)
                    continue
                new_pltps.append(new_pltp)
        if not toPointsOnly and len(new_pltps) + len(all_vertices) == 0:
            print ('Polytope --> map to none ... this is probably a bug')
        if toPointsOnly:
            assert (len(new_pltps) == 0)
            #print ('1 poly --> %d vs' % len(all_vertices) )
        else:
            print ('1 poly --> p%d, bv%d' % (len(new_pltps), len(all_vertices)) )
        return new_pltps, all_vertices
        
    


def test1():
    # do some test here
    class dh(object):
        def __init__(self):
            self.num_input = 1
            self.num_state = 2
            self.W_rec = np.array([[0.1,0.2],[0.3,-0.4]]).T
            self.W_in = np.array([[0.2],[-0.1]]).T
            self.b_rec = (np.array([[-0.2],[0.5]]).T)[0]
            self.init_state = np.array([0.1,1.2])
    

    dataholder = dh()

    vars = ['s%d' % s for s in range(dataholder.num_state)] + ['x%d' % x for x in range(dataholder.num_input)]

    initial_polytope = convex_polytope(vars)
    for idx, v in enumerate(dataholder.init_state):
        initial_polytope.add_vi_range(idx,v, v)
    initial_polytope.add_vi_range(dataholder.num_state, -1.0, 1.0) # input range

    #initial_polytope.add_vi_range(0, -1.0, 1.0)
    #initial_polytope.add_vi_range(1, -1.0, 1.0)
    #initial_polytope.add_vi_range(2, -1.0, 1.0)


    print (initial_polytope.get_vertices())

    pltp_mgr = polytopes(dataholder)
    new_polys,_ = pltp_mgr.propagate_split_polytope(initial_polytope,-1.0,1.0,0,False, 0, externQhull=False, qhullTimeout = 30)
    for p in new_polys:
        print ('-------------------------')
        p.add_vi_range(dataholder.num_state, -1.0, 1.0) # input range
        print ('ineqs:')
        p.print_eqs()
        print ('points:')
        print (p.get_vertices())



def test2():
    # do some test here
    class dh(object):
        def __init__(self):
            self.num_input = 1
            self.num_state = 2
            self.W_rec = np.array([[0.1,0.2],[0.3,-0.4]]).T
            self.W_in = np.array([[0.2],[-0.1]]).T
            self.b_rec = (np.array([[-0.2],[0.5]]).T)[0]
            self.init_state = None # should have no use
            self.init_range = [(0.1, 0.2),(1.0,1.2)]
            self.input_range = [(-1.0, 1.0)]
    

    dataholder = dh()

    vars = ['s%d' % s for s in range(dataholder.num_state)] + ['x%d' % x for x in range(dataholder.num_input)]

    initial_polytope = convex_polytope(vars)
    for idx, (lb, ub) in enumerate(dataholder.init_range):
        initial_polytope.add_vi_range(idx,lb, ub) # state range
    for idx, (lb, ub) in enumerate(dataholder.input_range):
        initial_polytope.add_vi_range(dataholder.num_state + idx,lb, ub) # input range

    #initial_polytope.add_vi_range(0, -1.0, 1.0)
    #initial_polytope.add_vi_range(1, -1.0, 1.0)
    #initial_polytope.add_vi_range(2, -1.0, 1.0)


    print (initial_polytope.get_vertices())

    pltp_mgr = polytopes(dataholder)
    new_polys,_ = pltp_mgr.propagate_split_polytope(initial_polytope,-1.0,1.0,0,False, 0, externQhull=False, qhullTimeout = 30)
    for p in new_polys:
        print ('-------------------------')
        p.add_vi_range(dataholder.num_state, -1.0, 1.0) # input range
        print ('ineqs:')
        p.print_eqs()
        print ('points:')
        print (p.get_vertices())

        

if __name__ == "__main__":
    test1()
        
