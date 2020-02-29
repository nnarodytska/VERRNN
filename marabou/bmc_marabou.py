import numpy as np
import dataload
from maraboupy import MarabouCore

# 1. We want to confirm our bound estimation result. This can be checked with bmc. And it is done
# 2. We want to check if the naive IA is able to prove?
#
#

large = 500.0

class BMCMarabou(object):
    def __init__(self, dataholder, precision, input_idx = 0, another_idx = 1):
        self.num_state = dataholder.num_state
        self.num_input = 1 # dataholder.num_input
        self.dataholder= dataholder # the true dataholder (w.o. change on num_input)
        self.W_rec = dataholder.W_rec
        self.W_in = dataholder.W_in[input_idx] # for the first input
        print (dataholder.W_in)
        assert (dataholder.W_in.shape[0] == 2)
        assert (dataholder.W_in.shape[1] == self.num_state)
        self.W_in2 = dataholder.W_in[another_idx]
        self.b_rec = dataholder.b_rec
        self.init_state = dataholder.init_state
        self.precision = precision
        self.W_out = dataholder.W_out
        self.b_out = dataholder.b_out
        assert (self.W_out.shape[1] == 1)
        assert (self.W_out.shape[0] == self.num_state)
        self.num_output = self.b_out.shape[0]
        
    def _round(self,v):
        if self.precision == 0:
            return float(v)
        return (((int(v * self.precision) / float(self.precision) )))
    
    
    def createProblem(self, n_stimulus_layer, n_settle_layer , ilb, iub, expect_sign, settle_input_val = 1.0):
        def get_ivar(layer,idx):
            assert (idx >= 0)
            if layer < n_stimulus_layer: # 0..49
                assert (idx < self.num_state)
                return (self.num_state+self.num_input)*layer + idx
            assert (0 <= layer and layer < n_stimulus_layer+n_settle_layer) # 0..99
            assert (idx < self.num_state)
            return (self.num_state+self.num_input)*n_stimulus_layer + self.num_state*(layer - n_stimulus_layer) + idx

        def get_inp(layer,inp):
            assert (layer >=0 and layer < (n_stimulus_layer))
            assert (inp >= 0 and inp < self.num_input)
            return (self.num_state+self.num_input)*layer + self.num_state + inp

        def get_relu_fvar(layer, idx):
            assert (layer >=0 and layer < (n_stimulus_layer + n_settle_layer-1))
            base = (self.num_state+self.num_input)*n_stimulus_layer + self.num_state*n_settle_layer
            return base + self.num_state*layer + idx

        def get_outvar(oidx):
            return (self.num_state+self.num_input)*n_stimulus_layer + \
                self.num_state*n_settle_layer + \
                self.num_state*(n_stimulus_layer+n_settle_layer-1) + oidx 

        self.out_var_idx = get_outvar(0)
        
        # construct solver
        self.inputQuery = MarabouCore.InputQuery()
        self.inputQuery.setNumberOfVariables( 
            (self.num_state+self.num_input)*n_stimulus_layer + \
            self.num_state*n_settle_layer + \
            self.num_state*(n_stimulus_layer+n_settle_layer-1) + self.num_output ) # before relus + 
                
        # add init state bound
        print ('init-state = ')
        print (self.init_state)
        for idx in range(self.num_state):
            self.inputQuery.setLowerBound(get_ivar(0,idx), abs(self.init_state[idx])-0.00001)
            self.inputQuery.setUpperBound(get_ivar(0,idx), abs(self.init_state[idx])+0.00001)

        # add input var bound
        for layer in range(n_stimulus_layer):
            for inp in range(self.num_input):
                self.inputQuery.setLowerBound(get_inp(layer,inp), ilb)
                self.inputQuery.setUpperBound(get_inp(layer,inp), iub)

        # add output range
        if expect_sign > 0:
            for idx in range(self.num_output):
                self.inputQuery.setLowerBound(get_outvar(idx), 0.5)
                self.inputQuery.setUpperBound(get_outvar(idx), large)
        else:
            for idx in range(self.num_output):
                self.inputQuery.setLowerBound(get_outvar(idx), -large)
                self.inputQuery.setUpperBound(get_outvar(idx), -0.5)

        # add inner state bound
        for layer in range(1,n_stimulus_layer+n_settle_layer):
            for idx in range(self.num_state):
                self.inputQuery.setLowerBound(get_ivar(layer,idx), -large)
                self.inputQuery.setUpperBound(get_ivar(layer,idx), large)
        # add fb bound
        for layer in range(0,n_stimulus_layer+n_settle_layer-1):
            for idx in range(self.num_state):
                self.inputQuery.setLowerBound(get_relu_fvar(layer,idx), -large)
                self.inputQuery.setUpperBound(get_relu_fvar(layer,idx), large)


        # construct stimulus layers
        print self.W_rec
        print self.W_in
        print self.b_rec
        # let's encode the first layer
        """
        layer = 0
        for idx_prime in range(self.num_state):
            equation = MarabouCore.Equation()
            fix_state = 0
            for idx in range(self.num_state):
                fix_state += self.W_rec[idx, idx_prime] * abs(self.init_state[idx])
            for idx in range(self.num_input):
                assert (idx == 0)
                equation.addAddend(self._round(self.W_in[idx_prime]), get_inp(layer, idx))
            equation.addAddend(-1, get_relu_fvar(layer, idx_prime) )
            equation.setScalar(-self._round(self.b_rec[idx_prime] + fix_state))
            self.inputQuery.addEquation(equation)
        """


        for layer in range(0,n_stimulus_layer):
            for idx_prime in range(self.num_state):
                equation = MarabouCore.Equation()
                for idx in range(self.num_state):
                    equation.addAddend(self._round(self.W_rec[idx, idx_prime]), get_ivar(layer, idx))
                for idx in range(self.num_input):
                    assert (idx == 0)
                    equation.addAddend(self._round(self.W_in[idx_prime]), get_inp(layer, idx))
                equation.addAddend(-1, get_relu_fvar(layer, idx_prime) )
                equation.setScalar(-self._round(self.b_rec[idx_prime]))
                self.inputQuery.addEquation(equation)

        # construct settle layers
        for layer in range(n_stimulus_layer, n_stimulus_layer+n_settle_layer-1):
            for idx_prime in range(self.num_state):
                equation = MarabouCore.Equation()
                for idx in range(self.num_state):
                    equation.addAddend(self._round(self.W_rec[idx, idx_prime]), get_ivar(layer, idx))
                equation.addAddend(-1, get_relu_fvar(layer, idx_prime) )
                equation.setScalar(self._round(-self.b_rec[idx_prime]-self.W_in2[idx_prime]*settle_input_val))
                self.inputQuery.addEquation(equation)

        # construct relu relations
        for layer in range(n_stimulus_layer+n_settle_layer-1):
            for idx in range(self.num_state):
                MarabouCore.addReluConstraint(self.inputQuery,get_relu_fvar(layer,idx),get_ivar(layer+1,idx))


        # construct output variables
        last_layer = n_stimulus_layer+n_settle_layer-1
        for out_idx in range(self.num_output):
            out_equation = MarabouCore.Equation()
            for idx in range(self.num_state):
                out_equation.addAddend(self._round(self.W_out[idx, out_idx]), get_ivar(last_layer, idx))
            out_equation.addAddend(-1, get_outvar(out_idx))
            out_equation.setScalar(-self._round(self.b_out))
            self.inputQuery.addEquation(out_equation)

        self.inputQuery.dump()

    def solve(self):
        self.vars, self.stat = MarabouCore.solve(self.inputQuery, MarabouCore.Options(), "")
        if len(self.vars)>0:
            otp = self.vars[self.out_var_idx]
            return ("SAT" + str(otp))
        else:
            return ("UNSAT")



# -----------------------------------------------
# configurations

modelpath = "../N7_L1_r11/"


testranges = [\
(0.000087 ,0.000115),
(0.027858 ,0.036758),
(0.111430 ,0.147033),
(0.445722 ,0.588134),
(0.891444 ,1.176267),
(0.000071 ,0.000142),
(0.022627 ,0.045255),
(0.090510 ,0.181019),
(0.362039 ,0.724077),
(0.724077 ,1.448155),
(0.000062 ,0.000163),
(0.019698 ,0.051984),
(0.078793 ,0.207937),
(0.315173 ,0.831746),
(0.630346 ,1.663493),
(-0.000115,-0.000087),
(-0.036758,-0.027858),
(-0.147033,-0.111430),
(-0.588134,-0.445722),
(-1.176267,-0.891444),
(-0.000142,-0.000071),
(-0.045255,-0.022627),
(-0.181019,-0.090510),
(-0.724077,-0.362039),
(-1.448155,-0.724077),
(-0.000163,-0.000062),
(-0.051984,-0.019698),
(-0.207937,-0.078793),
(-0.831746,-0.315173),
(-1.663493,-0.630346)]

import signal
import time

res = None

class TimeoutExp(Exception):
  pass

class TimeoutException():
    """Timeout class using ALARM signal and raise exception."""
    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)    # disable alarm

    def raise_timeout(self, *args):
        raise TimeoutExp

def test():
    global res
    with open('result.log', 'w') as fout:
        for ilb, iub in testranges[8:]:
            weightsObj = dataload.LOADER(modelpath, 10)
            bmc = BMCMarabou(weightsObj, precision = 0 )
            bmc.createProblem(50,50, ilb = ilb, iub = iub, expect_sign = 1 if ilb < 0 else -1)
            res = None
            
            t0 = time.time()
            with TimeoutException(6000):
                try:
                    res = bmc.solve()
                except TimeoutExp:
                    res = 'TIMEOUT'
            t1 = time.time()
            print >> fout, 'ilb = ', ilb, 'iub = ', iub, 'res = ' , res , 'time = ', t1 - t0
            fout.flush()
        
test()
