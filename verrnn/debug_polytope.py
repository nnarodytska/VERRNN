import polytope
import numpy as np



def compute_polytope_vertices_fraction(A, b):
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
    mat = polytope.cdd.Matrix(np.hstack([b, -A]), number_type='fraction')
    mat.rep_type = polytope.cdd.RepType.INEQUALITY
    P = polytope.cdd.Polyhedron(mat)
    g = P.get_generators()
    V = np.array(g)
    print ('---------------')
    print ('V = ')
    print (V)
    print ('---------------')
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
            vertices.append(V[i, 1:])
    return vertices


class polytopeDebug(polytope.convex_polytope):
    def __init__(self, f):
        super().__init__([])
        self.load_from_file(f)

    def convert_to_lp(self):
        nvars = len(self.vars)
        solver = polytope.ilp.cplex.Cplex()
        S = ['s_%d'  % idx for idx in range(nvars)]
        solver.variables.add(names = S)
        lin_exprs = []
        rhss = []
        for idx in range(len(self.coefs)):
            lin_expr = [float(c) for c in self.coefs[idx]]
            lin_exprs.append(polytope.ilp.cplex.SparsePair(ind = S, val = lin_expr))
            rhss.append(float(self.b[idx]))

        #print (lin_exprs)
        #print (rhss)
        solver.linear_constraints.add(lin_expr=lin_exprs, senses= 'L'*len(self.coefs), rhs = rhss )
        solver.solve()
        sol = solver.solution.get_values()
        print ('solution = ',sol)
        Ax = np.matmul(np.array(self.coefs), sol)
        print ('Ax=')
        print (Ax)
        print ('b=')
        print (self.b)
        print (Ax < np.array(self.b))

    def get_vertices_fraction(self):
        vs = compute_polytope_vertices_fraction(np.array(self.coefs), np.array(self.b))
        return vs

    def check_zeros(self, pts, epsilon):
        ptsum = np.sum(np.abs(pts), axis=0)
        print (np.argwhere(ptsum < epsilon))

def debug():
    p = polytopeDebug('polytope.dump.npz')

    print ('---- BEFORE ----')
    print ('A = ')
    print (np.array(p.coefs))
    print ('b = ')
    print (np.array(p.b))
    print (p.attribute)
    #print (p.attribute[:])
    p.attribute.has_input_constraints = True
    p.clear_small_coef(1e-7)
    print ('---- AFTER ----')
    print ('A = ')
    print (np.array(p.coefs))
    print ('b = ')
    print (np.array(p.b))

    # ---------------------------- 
    #pts = np.load('original_pts.npz')['arr_0']
    #print ('constructed from pts:')
    #print (pts)
    #p.check_zeros(pts, epsilon = 0.00001)

    #p.add_vi_range(0,-10.0,10.0)
    #p.add_vi_range(1,-10.0,10.0)
    #p.add_vi_range(2,0.0,0.0)
    #p.add_vi_range(3,0.0,0.0)
    #p.add_vi_range(4,-10.0,10.0)
    #p.add_vi_range(5,-10.0,10.0)
    #p.add_vi_range(6,0.0,0.0)
    #p.add_vi_range(7,-10.0,10.0)
    #p.convert_to_lp()
    print (p.get_vertices_fraction())
    print (p.attribute)
    print (p.get_vertices())
    
    #print (p.get_vertices())


if __name__ == '__main__':
    debug()