import numpy as np

class RangeSampling(object):
    def __init__(self, dataholder):
        """ zero_steps are the steps of fixation """
        self.num_state = dataholder.num_state
        self.num_input = dataholder.num_input
        self.dataholder = dataholder
        self.sample_ubs = []
        self.sample_lbs = []
        self.MAX = 1000000
        
        
    def sample_a_point(self, num_stimulus_step, stimulus, num_response_step,  clipping = 0, verbose = False): # stimulus_step: int , stimulus: f(int)->value
        self.state_list = [self.dataholder.init_state]
        #print (self.dataholder.init_state)
        sim_W_rec_reg = self.dataholder.W_rec
        sim_W_in_reg  = self.dataholder.W_in
        for idx in range(num_stimulus_step+num_response_step):
            prev_state = self.state_list[-1] # get the previous state
            if idx < num_stimulus_step:
                ix = np.array([[ stimulus(idx), 0.0]]) # get the stimulus
            else:
                ix = np.array([[ 0.0, 1.0]]) # get the stimulus
            if clipping:
                new_state = np.matmul( np.minimum( np.maximum(prev_state, 0.0) , clipping),  sim_W_rec_reg) + \
                    np.matmul( ix , sim_W_in_reg ) + self.dataholder.b_rec
            else:
                new_state = np.matmul( np.maximum(prev_state, 0.0),  sim_W_rec_reg) + \
                    np.matmul( ix , sim_W_in_reg ) + self.dataholder.b_rec
            if verbose:
                print ('======== sample @{idx} ========='.format(idx = idx))
                print ('-------- prev state ----------')
                print (prev_state)
                print ('-------- input ----------')
                print (ix)
                print ('-------- new state ----------')
                print (new_state)
                print ('======== END sample @{idx} ========='.format(idx = idx))

            self.state_list.append(new_state)

    def update_range(self): # return void
        for idx in range(len(self.state_list)):
            if len(self.state_list[idx].shape) == 1:
                slist = self.state_list[idx]
            else:
                slist = self.state_list[idx][0]

            if idx >= len(self.sample_lbs):
                self.sample_lbs.append([self.MAX]*self.num_state)
                self.sample_ubs.append([-self.MAX]*self.num_state)

            for sidx in range(self.num_state):
                v = slist[sidx]
                if self.sample_lbs[idx][sidx] > v:
                    self.sample_lbs[idx][sidx] = v
                if self.sample_ubs[idx][sidx] < v:
                    self.sample_ubs[idx][sidx] = v
    
    def get_node_output_range(self):
        rnn_node_output_ubs = [-self.MAX] *self.num_state
        rnn_node_output_lbs = [self.MAX] * self.num_state
        for idx in range(len(self.state_list)):
            for sidx in range(self.num_state):
                ub = self.sample_ubs[idx][sidx]
                lb = self.sample_lbs[idx][sidx]
                rnn_node_output_ubs[sidx] = max(rnn_node_output_ubs[sidx], ub)
                rnn_node_output_lbs[sidx] = min(rnn_node_output_lbs[sidx], lb)
        return rnn_node_output_lbs, rnn_node_output_ubs
                

    def print_signs(self):
        self.sign_lists = []
        for idx in range(len(self.sample_ubs)):
            self.sign_lists.append([]) # make a new frame
            for sidx in range(len(self.sample_ubs[idx])):
                ub = self.sample_ubs[idx][sidx]
                lb = self.sample_lbs[idx][sidx]
                ch = ' '
                assert (ub >= lb)
                if ub < 0:
                    ch = '0'
                elif lb >= 0:
                    ch = '+'
                self.sign_lists[-1].append(ch)
                print ( ch , end = '')
                #if idx == 0:
                #    print (ub,lb)
            print (' ')
        return self.sign_lists



    def check_bounds_network(self, lbs, ubs, epsilon = 0.001):
        #print (lbs.shape)
        #print (len(self.state_list))
        flag = True
        for idx in range(1,len(self.state_list)-1):
            lb = lbs[idx][0]
            ub = ubs[idx][0]
            slist = self.state_list[idx][0]
            for sidx in range(self.num_state):
                if not ((lb[sidx] - epsilon <= slist[sidx]) and (slist[sidx]  <= ub[sidx] + epsilon)):
                    print ('layer {idx}.{sidx} : NOT {l} <= {v} <= {u}'.format(idx = idx, sidx = sidx , l = lb[sidx], v = slist[sidx], u = ub[sidx]))
                    flag = False
        return flag

    def check_bounds_static_estimator(self, lbs, ubs, epsilon = 0.001):
        print ('checking bounds for first # layers: ',len(self.state_list))
        flag = True
        for idx in range(1,len(self.state_list)-1):
            lb = lbs[idx]
            ub = ubs[idx]
            #print (lb.shape)
            slist = self.state_list[idx][0]
            for sidx in range(self.num_state):
                if not ((lb[sidx] - epsilon <= slist[sidx]) and (slist[sidx] <= ub[sidx] + epsilon)):
                    print ('layer {idx}.{sidx} : NOT {l} <= {v} <= {u}'.format(idx=idx, sidx = sidx, l = lb[sidx], v = slist[sidx], u = ub[sidx] ))
                    flag = False
        return flag
