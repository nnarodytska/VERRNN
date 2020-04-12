import cplex
import numpy as np

class ilp_encoder(object):
    def __init__(self, dataholder):
        self.num_state = dataholder.num_state
        self.num_input = dataholder.num_input
        self.dataholder= dataholder
        self.frames = []
        self.solver = None
    
    def encode_fixed_val(self, slv, SI, vlist):
        assert (len(SI) == len(vlist))
        lin_expr = []
        rhs = []
        for idx in range(len(SI)):
            lin_expr.append( cplex.SparsePair(ind = [SI[idx]], val = [1.0]) )
            rhs.append( float(vlist[idx]) )
        slv.linear_constraints.add(lin_expr = lin_expr, rhs = rhs, senses = 'E' * (len(SI)))

    def encode_block_binary_assignment(self, slv, Bvars, Bsol ):
        val = []
        rhs = 0.0
        for sol in Bsol:
            if sol <= 0.3 : # 0
                # we want to see it to be 1
                val.append(1.0)
            elif sol >= 0.9: # 1
                # we want to see it to be 0 \neg(B) -> 1-B
                val.append(-1.0)
                rhs += 1.0 # -neg please
            else:
                print ('sol:', sol)
                assert False
        rhs = 1.0-rhs

        #print ('--->', Bvars, '*' , val, '>=', rhs)
        slv.linear_constraints.add(lin_expr = [cplex.SparsePair(ind = Bvars, val = val)], rhs = [rhs], senses = ['G'] )
    
    def encode_all_frames(self, num_stimulus_layer, num_resp_layer, W_in_st, W_in_resp, W_out, b_out, st_ilb, st_iub, resp_i ):
        # requires dataholder to have 

        # encode stimulus layer
        state_var = [] # list -> list
        input_var = [] # list -> list
        
        #for 


    def encode_single_frame(self, layer, slv, ilb, iub):
        SI = ['s_%d_%d'  % (layer,idx) for idx in range(self.num_state)] # SI
        XI = ['x_%d_%d'  % (layer,idx) for idx in range(self.num_input)] # input
        SO = ['so_%d_%d' % (layer,idx) for idx in range(self.num_state)] # state output
        SL = ['sl_%d_%d' % (layer,idx) for idx in range(self.num_state)] # slack
        B  = ['b_%d_%d'  % (layer,idx) for idx in range(self.num_state)] # binary indicator

        #self.solver = cplex.Cplex()
        slv.variables.add(names = SI, lb = [0.0]* self.num_state, types=[slv.variables.type.continuous]*self.num_state)
        slv.variables.add(names = XI,  lb = [ilb]* self.num_input, ub = [iub]* self.num_input, types=[slv.variables.type.continuous]*self.num_input)
        slv.variables.add(names = SO, lb = [0.0]* self.num_state, types=[slv.variables.type.continuous]*self.num_state)
        slv.variables.add(names = SL, lb = [0.0]* self.num_state, types=[slv.variables.type.continuous]*self.num_state)
        slv.variables.add(names = B,  types = [slv.variables.type.binary] * self.num_state, lb = [0]* self.num_state, ub = [1]* self.num_state)
        lin_expr = []
        rhs = []
        for idx in range(self.num_state):
            SIc = self.dataholder.W_rec[:,idx].flatten().tolist()
            XIc = self.dataholder.W_in[:,idx].flatten().tolist()
            #print ('--->', SIc + XIc + [-1.0, 1.0],'*', SI + XI + [SO[idx], SL[idx]], '=', -self.dataholder.b_rec[idx] )
            val = SIc + XIc + [-1.0, 1.0]
            val = [float(x) for x in val]
            lin_expr.append( cplex.SparsePair(ind = SI + XI + [SO[idx], SL[idx]], \
                val = val ) )
            rhs.append( float(-self.dataholder.b_rec[idx]) )
            # check here!!!
        # SO >= 0, SL >= 0
        
        slv.linear_constraints.add( \
            lin_expr = lin_expr, senses='E'*self.num_state, rhs = rhs )

        slo_lin_expr = []
        slo_rhs = []
        for idx in range(self.num_state):
            slo_lin_expr.append( cplex.SparsePair(ind = [SL[idx]], val=[1.0]) )
            slo_lin_expr.append( cplex.SparsePair(ind = [SO[idx]], val=[1.0]) )
            slo_rhs.append( 0.0 )
            slo_rhs.append( 0.0 )
        
        slv.linear_constraints.add( \
            lin_expr = slo_lin_expr, senses='GG'*self.num_state, rhs = slo_rhs )

        for idx in range(self.num_state):
            lin_expr_s = cplex.SparsePair(ind = [SO[idx]], val = [1.0])
            lin_expr_sl= cplex.SparsePair(ind = [SL[idx]], val = [1.0])
            slv.indicator_constraints.add( \
                lin_expr = lin_expr_s, sense = 'L', rhs = 0.0, indvar = B[idx], complemented = 1)
            slv.indicator_constraints.add( \
                lin_expr = lin_expr_sl, sense = 'L', rhs = 0.0, indvar = B[idx], complemented = 0)

        return (SI, XI, SO, SL, B)



def test1():
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
        ilp_enc = ilp_encoder(dataholder)
        solver = cplex.Cplex()
        SI, XI , SO, SL, B = ilp_enc.encode_single_frame(0, solver, -1.0, 1.0)
        ilp_enc.encode_fixed_val(solver, SI, [0.1,1.2])
    
        for bs in BSols:
            print ('--->block:',bs)
            ilp_enc.encode_block_binary_assignment(solver, B, bs)

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
        if solve_idx == 10:
            break
        

if __name__ == "__main__":
    test1()
        
