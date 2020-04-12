import numpy as np
import scipy.io
from os import path

class LOADER(object):
    def __init__(self, wp, zero_steps, clipping = 0, verbose = False):
        weight_path = path.join(wp, "models/weights.npz")
        weights = np.load(weight_path)
        self.init_val = weights["/init_state"][0]
        self.W_in = weights["/W_in"]
        self.W_out = weights["/W_out"]
        self.W_rec = weights["/W_rec"]
        self.b_rec = weights["/b_rec"]
        self.b_out = weights["/b_out"]
        self.rec_connectivity = weights["/rec_connectivity"]

        if (self.rec_connectivity == 0.0).any():
            print ('apply rec_connectivity mask on W_rec')
            self.W_rec = self.W_rec * self.rec_connectivity
        
        # make it the way we want
        self.W_rec = np.transpose(self.W_rec)
        self.W_in  = np.transpose(self.W_in)
        self.W_out = np.transpose(self.W_out)
        self.init_val = self.compute_new_init_state(self.init_val, zero_steps, self.W_rec, self.b_rec, clipping)
        self.init_state = self.init_val # alias
        
        self.num_state = self.W_rec.shape[0]
        self.num_input = self.W_in.shape[0]
        
        assert (len(self.W_rec.shape) == 2 and self.W_rec.shape[0] == self.W_rec.shape[1] and self.W_rec.shape[0] == self.num_state )
        assert (len(self.W_in.shape) == 2 and self.W_in.shape[0] == self.num_input and self.W_in.shape[1] == self.num_state)
        assert (len(self.W_out.shape) == 2 and self.W_out.shape[0] == self.num_state and self.W_out.shape[1] == 1)
        assert (len(self.b_rec.shape) == 1 and self.b_rec.shape[0] == self.num_state)
        assert (len(self.b_out.shape) == 1 and self.b_out.shape[0] == 1)
        
        if verbose:
            print ('loaded %d states, %d inputs' % (self.num_state,self.num_input))
    
    def toMatlab(self, filename):
        scipy.io.savemat(filename, {'W_in':self.W_in, 'W_out':self.W_out, 'W_rec':self.W_rec, 'b_rec':self.b_rec, 'b_out':self.b_out, 'init_state':self.init_state})
    
    
    def compute_new_init_state(self, init_vector, zero_steps, sim_W_rec, sim_b_rec, clipping):
        ## sub-functions
        def simulate_recurrent_timestep(state):
            if clipping:
                return np.matmul(np.minimum(np.maximum(state, 0), clipping), sim_W_rec) + sim_b_rec
            else:
                return np.matmul(np.maximum(state, 0), sim_W_rec) + sim_b_rec
        
        s = init_vector
        for i in range(zero_steps):
            s = simulate_recurrent_timestep(s)
        # now we have s as the new state
        return s
        
