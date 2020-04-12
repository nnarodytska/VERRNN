import z3
import numpy as np

"""the model is like 

state' = relu(state) * W_rec + in * W_in + b_rec 

output = relu(state) * W_out + b_out

property 

(rule (
    => ()
))

"""

class ChcEncoder(object):
    def rounding(self,v):
        if self.precision == 0:
            return v
        else:
            return int(v*self.precision)/float(self.precision)

    def __init__(self, dataholder, precision = 0, relu_extra_var = True):
        z3.set_option(**{"fp.engine":"spacer"})
        
        self.relu_extra_var = relu_extra_var
        self.precision = precision
        self.num_state = dataholder.num_state
        self.num_input = dataholder.num_input
        self.dataholder= dataholder
        
        # create variables here
        self.state_vars = []
        self.state_vars_prime = []
        for idx in range(self.num_state):
            self.state_vars.append( z3.Real('s' + str(idx) ) )
            self.state_vars_prime.append( z3.Real('sp' + str(idx) ) )
            
        self.input_vars = []
        self.input_vars_prime = []
        for idx in range(self.num_input):
            self.input_vars.append( z3.Real('i' + str(idx) ) )
            self.input_vars_prime.append( z3.Real('ip' + str(idx) ))
        
        # create index variable
        self.index_var = z3.Real('idx')
        self.index_var_prime = z3.Real('idxp')
        self.precision = precision # (int (x * precision)/precision)
            
        self.fp = None # will be available after you call some Encode****
        # --------- preprocessing -------------------
        self.additional_assumptions = []
        self.RELUs = []
        for idx,v in enumerate(self.state_vars):
            if self.relu_extra_var:
                relu = z3.Real('RELU' + str(idx))
                self.RELUs.append( relu  ) # make the relus z3.If(v > 0, v, 0)
                self.additional_assumptions.append( z3.Or( z3.And(v < 0, relu == 0) ,  z3.And(v >= 0, relu == v) ) )
            else:
                self.RELUs.append( z3.If(v > 0, v, 0) )
                self.RELUs.append(v)


    def encode_init(self, init_vector): # --> set self.init_contraint : list
        assert (init_vector.shape[0] == self.num_state)
        init_constraint = [] # initially a list 
        for idx in range(self.num_state):
            var = self.state_vars[idx]
            iv  = self.rounding(init_vector[idx] )
            init_constraint.append(var == iv)
        init_constraint.append(self.index_var == 0)
        return init_constraint

    
    def encode_transition(self,W_rec,W_in, b_rec): # --> set self.transition_constraint
        assert (W_rec.shape[0] == W_rec.shape[1] and W_rec.shape[0] == self.num_state )
        assert (W_in.shape[0] == self.num_input and W_in.shape[1] == self.num_state)
        assert (b_rec.shape[0] == self.num_state)

        transition = []
        for idx_prime,sp in enumerate(self.state_vars_prime):
            elem = 0
            for idx in range(self.num_state):
                elem += self.rounding(W_rec[idx,idx_prime]) * self.RELUs[idx]
            for idx in range(self.num_input):
                elem += self.rounding (W_in[idx,idx_prime]) * self.input_vars[idx]
            elem += self.rounding(b_rec[idx_prime])
            transition.append(sp == elem)
        transition.append(self.index_var_prime == self.index_var + 1)
        return transition
    
    def encode_output(self, W_out, b_out): # assuming only one output
        assert (W_out.shape[0] == self.num_state)
        assert (b_out.shape[0] == 1)
        self.output_var = 0
        for idx,s in enumerate(self.RELUs):
            self.output_var += self.rounding(W_out[idx]) * s
        self.output_var += self.rounding(b_out)

    def encode_input_stimulus_bound(self, ilb, iub):
        return [self.input_vars[0] >= ilb, self.input_vars[0] <= iub , self.input_vars[1] == 0]
    
    def add_bounded_relu_stable_info(self, sign_lists, first_n_layers):
        for lidx in range(first_n_layers):
            slist = sign_lists[lidx]
            consq_list = []
            for sidx, sign in enumerate(slist):
                if sign == '0':
                    cnstr = self.state_vars[sidx] <= 0
                    consq_list.append(cnstr)
                elif sign == '+':
                    cnstr = self.state_vars[sidx] >= 0
                    consq_list.append(cnstr)
            if consq_list:
                self.additional_assumptions.append( z3.Implies(self.index_var == lidx, z3.And(consq_list)) )
                

    def EncodeReluStableProperty(self, relu_zero_position_list, ilb, iub, start = 0):
        # self.additional_assumptions
        init_cnstrs = self.encode_init(self.dataholder.init_val)
        trans_cnstrs = self.encode_transition(self.dataholder.W_rec, self.dataholder.W_in, self.dataholder.b_rec)
        input_cnstrs = self.encode_input_stimulus_bound(ilb, iub)
        #if start != 0:
        property_cnstrs = z3.Implies( self.index_var > start , z3.And([self.state_vars[i] <= 0 for i in relu_zero_position_list]))
        #else:
        #    property_cnstrs = z3.And([self.state_vars[i] <= 0 for i in relu_zero_position_list])

        self.fp = z3.Fixedpoint()
        boolsort = z3.BoolSort()
        # INV (s,i)
        # INV = z3.Function('INV', [v.sort() for v in self.state_vars + self.input_vars + [self.index_var] + (self.RELUs if self.relu_extra_var else []) ] + [boolsort])
        INV = z3.Function('INV', [v.sort() for v in self.state_vars + self.input_vars + [self.index_var] ] + [boolsort])
        fail = z3.Function('fail', [boolsort])

        self.fp.declare_var( \
            self.state_vars + self.state_vars_prime + \
            self.input_vars + self.input_vars_prime + \
            [self.index_var  , self.index_var_prime] + \
            (self.RELUs if self.relu_extra_var else []) )
        
        self.fp.register_relation(INV)
        self.fp.register_relation(fail)
        
        inv_on_state = INV( \
            self.state_vars + \
            self.input_vars + \
            [self.index_var] \
            )
            #+ (self.RELUs if self.relu_extra_var else []) )
        inv_on_next_state = INV( \
            self.state_vars_prime + \
            self.input_vars_prime + \
            [self.index_var_prime] \
            )
            #+ (self.RELUs if self.relu_extra_var else []))

        self.fp.rule( inv_on_state , init_cnstrs + self.additional_assumptions + input_cnstrs  )
        self.fp.rule( inv_on_next_state ,  \
            [inv_on_state] + trans_cnstrs + self.additional_assumptions + input_cnstrs )

        self.fp.rule( fail(), [inv_on_state] + self.additional_assumptions + [z3.Not(property_cnstrs) ]  )
        # not p as a bad state

        print (self.fp) # dump the rules
        print (self.fp.query(fail()))
        print (self.fp.get_answer())