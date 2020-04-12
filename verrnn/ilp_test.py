import dataload
import polytope
import numpy as np

# -----------------------------------------------
# configurations

modelpath = "../deep_monkey/results/N7_L1_r11/"

def test():
    weightsObj = dataload.LOADER(modelpath, 10)
    weightsObj.num_input = 1
    weightsObj.W_in = np.array([weightsObj.W_in[0]])
    ilp_enc = polytope.ilp_encoder(weightsObj)
    solver = polytope.cplex.Cplex()
    SI, XI , SO, SL, B = ilp_enc.encode_single_frame(0, solver, 0.5, 2.0)
    iv = [x if x >= 0.0 else 0.0 for x in weightsObj.init_state ]
    ilp_enc.encode_fixed_val(solver, SI, iv)
    
    solver.write('./ilptest.lp')
    solver.populate_solution_pool()
    num_sols = solver.solution.pool.get_num()
    print ('--->Feasible solutions:', num_sols)


def test2():
    # do some test here
    class dh(object):
        def __init__(self):
            self.num_input = 1
            self.num_state = 2
            self.W_rec = np.array([[0.1,0.2],[0.3,-0.4]]).T
            self.W_in = np.array([[0.2],[-0.1]]).T
            self.b_rec = (np.array([[-0.2],[0.5]]).T)[0]
            print (self.b_rec)
    
    BSols = []
    solve_idx = 0
    while True:

        dataholder = dh()
        ilp_enc = polytope.ilp_encoder(dataholder)
        solver = polytope.cplex.Cplex()
        SI, XI , SO, SL, B = ilp_enc.encode_single_frame(0, solver, -1.0, 1.0)
        ilp_enc.encode_fixed_val(solver, SI, [-0.1,1.2])
    
        for bs in BSols:
            print ('--->block:',bs)
            ilp_enc.encode_block_binary_assignment(solver, B, bs)

        solver.write("./ilptest-t2.lp")
        solver.populate_solution_pool()

        num_sols = solver.solution.pool.get_num()
        print ('--->Feasible solutions:', num_sols)

        if num_sols == 0:
            print ('--->No more solutions')
            break

        for idx in range(num_sols):
            print (solver.solution.pool.get_values(idx))
            print (solver.solution.pool.get_values(idx, B))
            BSols.append(solver.solution.pool.get_values(idx, B))
        solve_idx += 1
        break
        if solve_idx == 10:
            break

if __name__ == "__main__":
    #test()
    test2()
