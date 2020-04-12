import polytope
import numpy as np



class polytopeRange(polytope.convex_polytope):
    def __init__(self, f):
        super().__init__([])
        self.load_from_file(f)


def debug():
    p = polytopeRange('polytope.dump.npz')

    print ('---- BEFORE ----')
    print ('A = ')
    print (np.array(p.coefs))
    print ('b = ')
    print (np.array(p.b))
    p.clear_small_coef(1e-7)
    p.remove_similar_constraints(1e-7)
    p.check_problematic_neg_constraints(1e-7)
    p.truncate_precision(2**16)
    print ('---- AFTER ----')
    print ('A = ')
    print (np.array(p.coefs))
    print ('b = ')
    print (np.array(p.b))

    p.convert_to_lp()
    print (p.get_vertices())


if __name__ == '__main__':
    debug()