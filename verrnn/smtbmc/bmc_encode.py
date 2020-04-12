import numpy as np

from pysmt.shortcuts import Symbol, Or, GE, LT, LE, Real, Plus, And, Not, Equals, Ite, Times, Solver, serialize
from pysmt.typing import REAL
from pysmt.oracles import get_logic
# 1. We want to confirm our bound estimation result. This can be checked with bmc. And it is done
# 2. We want to check if the naive IA is able to prove?
#
#


class BMC(object):
    def __init__(self, init_vector, W_rec, b_rec, W_in, W_out, b_out, precision):
        """Please use some other's capability of computing the init vector"""
        self.init_vector = init_vector
        self.W_rec = W_rec
        self.b_rec = b_rec
        self.W_in  = W_in
        self.W_out = W_out
        self.b_out = b_out
        self.precision = precision

        self.num_state = W_rec.shape[0]
        self.num_input = W_in.shape[1]
        # checks
        assert (W_rec.shape[0] == W_rec.shape[1] and W_rec.shape[0] == self.num_state )
        assert (W_in.shape[1] == self.num_input and W_in.shape[0] == self.num_state)
        assert (b_rec.shape[0] == self.num_state)

        self.input_vars = []
        self.smt_var_prefix = ''
        self.prefix_num = 0

        # input constraints
        self.input_constraints = []
        self.iv_cnstr_func = None

        # frames
        self.frames = []
    
    def _encode_const(self,v):
        if self.precision == 0:
            if isinstance(v, int) or isinstance(v, float):
                return Real(v)
            else:
                return Real(v.item())
        return (Real((int(v * self.precision), int(self.precision) )))

    def _add_init_frame(self, init_v = None):
        assert (len(self.frames) == 0) # we must start from an empty frame
        #assert (len(self.relus) == 0) # the relu of the frames
        self.frames.append ([])
        #self.relus.append ([])
        if init_v is not None:
            for idx in range(self.num_state): # add concrete init frame
                v = self._encode_const(init_v[idx])
                self.frames[-1].append( v )
                #self.relus[-1].append( v if init_v[idx] > 0 else self._encode_const(0) )
        else:
            for idx in range(self.num_state): # add symbolic init frame
                v = Symbol(self.smt_var_prefix + "s_0_" + str(idx), REAL)
                self.frames[-1].append( v )
                #self.relus[-1].append( Ite( GE(v, self._encode_const(0) ), v, self._encode_const(0) )  )
        if len(self.input_vars) != 0:
            print ('>>> Warning: the input vars are not emptied!')
    
    def _add_transition_frame(self, n_layer, relu_stable_layer_list, is_stimulus_layer): 
        """relu_stable_layer_list : list of ' '/'0'/'+', is_stimulus_layer : bool (T for stimulus_layer and F for response_layer)"""
        def get_relu(idx, v):
            if not relu_stable_layer_list or relu_stable_layer_list[idx] == ' ':
                return Ite( GE(v, self._encode_const(0) ) , v, self._encode_const(0))
            if relu_stable_layer_list[idx] == '+':
                return v
            if relu_stable_layer_list[idx] == '0':
                return self._encode_const(0)
            assert False
        def create_input(idx): # return a tuple (i0,i1)
            if is_stimulus_layer:
                iv = Symbol(self.smt_var_prefix + "i_" + str(idx) + "_0", REAL )
                self.input_vars.append(iv)
                return ( iv , self._encode_const(0) )
            # else:
            return (self._encode_const(0), self._encode_const(1))

        assert (len(self.frames) > 0)
        #assert (len(self.relus) > 0)
        #prev_frame = self.relus[-1] # seems not used at all
        relu_stablize = []
        for idx in range(self.num_state):
            relu_stablize.append( \
                get_relu(idx, self.frames[-1][idx]))

        input_vec = create_input(n_layer)
        assert (len(input_vec) == self.num_input)

        self.frames.append([])
        for idx_prime in range(self.num_state): # next state
            elem = []
            for idx in range(self.num_state): # prev state
                elem.append( Times( self._encode_const(self.W_rec[idx_prime, idx]) , relu_stablize[idx] ) )
            for idx in range(self.num_input):
                elem.append( Times( self._encode_const(self.W_in[idx_prime, idx]) , input_vec[idx] ) )
            elem.append( self._encode_const(self.b_rec[idx_prime]) )
            elem = Plus(*elem)
            self.frames[-1].append( elem )

    def _clear_frames(self):
        self.frames = [] # 
        self.prefix_num += 1
        self.smt_var_prefix = 'iter' + str(self.prefix_num)
        self.input_constraints = []
        self.iv_cnstr_func = None

    def AddInputConstraints(self, iv_cnstr_func):
        """ iv_cnstr_func : idx, var -> expression """
        for idx,var in enumerate(self.input_vars):
            self.input_constraints.append( iv_cnstr_func(idx,var) )

    def RegisterRangeFunc(self, upper, lower):
        def _rangefunc(idx, v):
            return And( GE(v, self._encode_const(lower)), LE(v, self._encode_const(upper)) )
        self.iv_cnstr_func = _rangefunc

    def RegisterInputConstraintFunction(self, iv_cnstr_func):
        """ iv_cnstr_func : idx, var -> expression, 
        a function will be called per each frame (when
        new input variable is created) """
        self.iv_cnstr_func = iv_cnstr_func

    def CheckReluStabilityWhileUnrolling(self, n_stimulus_step, relu_stable_list):
        epsilon = 0
        def _check_frame(relu_v_list, stable_list, io_assumptions): # relu variables, stable information list
            s = Solver()
            for assumption in io_assumptions:
                s.add_assertion(assumption)
            for idx,v in enumerate(relu_v_list):
                print (f'state #{idx}')
                stable_information = stable_list[idx]
                res = False
                
                if stable_information == '0':
                    s.push()
                    formula = Not( LE ( v , self._encode_const(epsilon)))
                    s.add_assertion(formula)
                    res = s.solve()
                    s.pop()
                elif stable_information == '+':
                    s.push()
                    formula = Not( GE ( v , self._encode_const(-epsilon)))
                    s.add_assertion(formula)
                    res = s.solve()
                    s.pop()
                if res: # if sat
                    m = s.get_model()
                    print ('--------- assertions ---------')
                    print (serialize(Not( LE ( v , self._encode_const(0)))))
                    for assumption in io_assumptions:
                        print (assumption)
                    print ('--------- model ---------')
                    print (m)
                    print ('--------- stable info ---------')
                    print (f'idx:{idx} , {stable_information}')
                    print (stable_list)
                    print ('--------- state ---------')
                    print (m.get_py_value(v))
                    for v in relu_v_list:
                        print (m.get_py_value(v), end = ' , ')
                    return False,m
            return True, None

        assert (n_stimulus_step <= len(relu_stable_list))
        # first layer
        self._add_init_frame(init_v = self.init_vector)
        res, model = _check_frame( self.frames[-1] , relu_stable_list[0], self.input_constraints)
        if res == False:
            print ('RELU stable info error at init layer!')
            return

        # the later layers
        for nlayer in range(1,n_stimulus_step):
            self._add_transition_frame(nlayer, relu_stable_list[nlayer-1], True )
            if self.iv_cnstr_func:
                self.input_constraints.append( self.iv_cnstr_func(nlayer, self.input_vars[-1]) )
            else:
                print ("Warning: input constraint generator function not registered.")
            if True: #nlayer >= 7:
                res , model = _check_frame( self.frames[-1] , relu_stable_list[nlayer], self.input_constraints)
                if res == False: # dump layer by layer
                    print (f'RELU stable info error at layer {nlayer}')
                    self.DumpLayerValue(model)
                    return
                else:
                    print (f'layer #{nlayer} checked.')
        
    def CheckNaiveIA(self, n_response_step, lbs, ubs):
        """lbs and ubs are provided as the last layers lb/ub"""
        def _add_init_IA_bound(lbs, ubs):
            pass
        # first layer
        self._add_init_frame(init_v = None)
        pass

        
    
    # for debugging
    def DumpLayerValue(self, model):
        print ('dumping layer by layer')
        for idx in range(len(self.frames)):
            print (f'layer {idx} : ', end = '')
            for v in self.frames[idx]:
                raw_result = model.get_py_value(v)
                result = eval(str(raw_result)+".0")
                print (result, end = ' , ')
            print (' ') # end-of-line


