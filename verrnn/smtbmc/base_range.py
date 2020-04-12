import numpy as np
import matplotlib.pyplot as plt

class RangeEstimation(object):
    def __init__(self, dataholder, verbose = False):
        self.num_state = dataholder.num_state
        self.num_input = dataholder.num_input
        self.dataholder = dataholder

    def get_previous_populate_output_status(self, last_n:int = 1):
        """ return (stable?, pos? )"""
        olb = self.output_l_lists[-last_n:]
        oub = self.output_u_lists[-last_n:]
        return oub, olb

    def populate_polytope_state_range(self, stimulus_step, state_ub_list, state_lb_list, \
        stimulus_upper = 2, stimulus_lower = -2, response_step = 5, verbose = False  ):

        assert (stimulus_upper >= stimulus_lower)
        u_lists = []
        l_lists = []
        u_lists.append(state_ub_list)
        l_lists.append(state_lb_list) # the initial upper/lower bounds

        output_u_lists = []
        output_l_lists = []
        
        sim_W_rec_reg = self.dataholder.W_rec
        sim_W_in_reg  = self.dataholder.W_in
        sim_W_out_reg = self.dataholder.W_out
        b_out = self.dataholder.b_out
        
        input_max = np.array([stimulus_upper, 0])
        input_min = np.array([stimulus_lower, 0])

        range2_input_max = np.array([0.0, 1.0])
        range2_input_min = np.array([0.0, 1.0])

        Ws = [sim_W_rec_reg]
        biases = [self.dataholder.b_rec]
        W_ins = [sim_W_in_reg]

        ilbs = [input_min]
        ulbs = [input_max]

        for idx in range(stimulus_step+response_step): # stimulus stage   
            if verbose:
                print (f'>>>>>>>>>>>>>> calling to estimatiate layer: {idx+1}')
            lb, ub = self._compute_bounds_n_layers_out(idx+1, l_lists, u_lists, Ws , biases, W_ins, ilbs, ulbs, verbose = (verbose if idx >= 1 else False))
            #print (idx)
            #print (ub.shape)
            #print (lb.shape)
            #if idx >= stimulus_step:
            #    Ws_output = [sim_W_out_reg] + Ws[1:]
            #    biases_output = [b_out] + biases[1:]
            #    olb, oub = self._compute_bounds_n_layers_out(idx+1, l_lists, u_lists, Ws_output , biases_output, W_ins, ilbs, ulbs, verbose = (verbose if idx >= 1 else False))
            #    print (idx)
            #    print (olb.shape)
            #    print (oub.shape)
            #    output_u_lists.append(oub)
            #    output_l_lists.append(olb)

            # deal with the next level
            u_lists.insert(0, ub) #u_lists.append(ub)
            l_lists.insert(0, lb) #l_lists.append(lb)
            Ws.append(sim_W_rec_reg)
            biases.append(self.dataholder.b_rec)
            W_ins.append(sim_W_in_reg)
            ilbs.insert(0, input_min if idx < stimulus_step-1 else range2_input_min)
            ulbs.insert(0, input_max if idx < stimulus_step-1 else range2_input_max)

        u_lists.reverse()
        l_lists.reverse()
        self.u_lists = u_lists
        self.l_lists = l_lists
        self.stimulus_step = stimulus_step

        # compute the output range
        for idx in range(stimulus_step + response_step):
            state_lb = l_lists[idx]
            state_ub = u_lists[idx]
            olb, oub = self._output_interval_arithmetic(state_lb, state_ub, sim_W_out_reg, b_out)
            #print (olb.shape)
            output_u_lists.append(oub)
            output_l_lists.append(olb)
            # do a simple IA : W+*u + W-*l + b
        self.output_u_lists = output_u_lists
        self.output_l_lists = output_l_lists

    def populate_ranges_tight_output(self, stimulus_step, stimulus_upper = 2, stimulus_lower = -2, response_step = 5, verbose = False):
        assert (stimulus_upper >= stimulus_lower)
        u_lists = []
        l_lists = []
        u_lists.append(self.dataholder.init_state)
        l_lists.append(self.dataholder.init_state) # the initial upper/lower bounds

        output_u_lists = []
        output_l_lists = []
        
        sim_W_rec_reg = self.dataholder.W_rec
        sim_W_in_reg  = self.dataholder.W_in
        sim_W_out_reg = self.dataholder.W_out
        b_out = self.dataholder.b_out
        
        input_max = np.array([stimulus_upper, 0])
        input_min = np.array([stimulus_lower, 0])

        range2_input_max = np.array([0.0, 1.0])
        range2_input_min = np.array([0.0, 1.0])

        Ws = [sim_W_rec_reg]
        biases = [self.dataholder.b_rec]
        W_ins = [sim_W_in_reg]

        ilbs = [input_min]
        ulbs = [input_max]

        for idx in range(stimulus_step+response_step): # stimulus stage   
            if verbose:
                print (f'>>>>>>>>>>>>>> calling to estimatiate layer: {idx+1}')
            lb, ub = self._compute_bounds_n_layers_out(idx+1, l_lists, u_lists, Ws , biases, W_ins, ilbs, ulbs, verbose = (verbose if idx >= 1 else False))
            #print (idx)
            #print (ub.shape)
            #print (lb.shape)
            #if idx >= stimulus_step:
            #    Ws_output = [sim_W_out_reg] + Ws[1:]
            #    biases_output = [b_out] + biases[1:]
            #    olb, oub = self._compute_bounds_n_layers_out(idx+1, l_lists, u_lists, Ws_output , biases_output, W_ins, ilbs, ulbs, verbose = (verbose if idx >= 1 else False))
            #    print (idx)
            #    print (olb.shape)
            #    print (oub.shape)
            #    output_u_lists.append(oub)
            #    output_l_lists.append(olb)

            # deal with the next level
            u_lists.insert(0, ub) #u_lists.append(ub)
            l_lists.insert(0, lb) #l_lists.append(lb)
            Ws.append(sim_W_rec_reg)
            biases.append(self.dataholder.b_rec)
            W_ins.append(sim_W_in_reg)
            ilbs.insert(0, input_min if idx < stimulus_step-1 else range2_input_min)
            ulbs.insert(0, input_max if idx < stimulus_step-1 else range2_input_max)

        u_lists.reverse()
        l_lists.reverse()
        self.u_lists = u_lists
        self.l_lists = l_lists
        self.stimulus_step = stimulus_step

        # compute the output range
        for idx in range(stimulus_step + response_step):
            state_lb = l_lists[idx]
            state_ub = u_lists[idx]
            olb, oub = self._output_interval_arithmetic(state_lb, state_ub, sim_W_out_reg, b_out)
            #print (olb.shape)
            output_u_lists.append(oub)
            output_l_lists.append(olb)
            # do a simple IA : W+*u + W-*l + b
        self.output_u_lists = output_u_lists
        self.output_l_lists = output_l_lists

    def _compute_bounds_n_layers_out(self, n, lbs, ubs, Ws, biases, W_ins, ilbs, iubs, verbose):
        assert (n == len(lbs))
        assert (n == len(ubs)) # W is just a matrix, bias is just a vector
        assert (n == len(Ws))
        assert (n == len(biases))
        assert (n == len(W_ins))
        assert (n == len(ilbs))
        assert (n == len(iubs))
        lb = lbs[0]
        ub = ubs[0]
        W = Ws[0]
        b = biases[0]
        W_in = W_ins[0]
        ilb = ilbs[0]
        iub = iubs[0]
        #print (W.shape)
        # base case
        #print (f'<<<< n:{n} , verbose:{verbose} ')
        if n == 1:
            naive_ia_bounds = self._interval_arithmetic(lb, ub, W, b, W_in, ilb, iub)
            if verbose:
                print ('--------- IA naive call --------')
                print ('--------- lb --------')
                print (lb)
                print ('--------- ub --------')
                print (ub)
                print ('--------- W --------')
                print (W)
                print ('--------- b --------')
                print (b)
                print ('--------- W_in --------')
                print (W_in)
                print ('--------- ilb --------')
                print (ilb)
                print ('--------- iub --------')
                print (iub)
                print ('---------- result bounds ------')
                print (naive_ia_bounds)
                print ('--------- END of IA naive call --------')

            return naive_ia_bounds
        # recursive case
        W_prev = Ws[1]
        b_prev = biases[1]
        W_in_prev = W_ins[1]
        W_A = np.transpose(np.transpose(W) * (lb > 0))
        W_NA = np.transpose(np.transpose(W) * (lb <= 0))
        prev_layer_bounds = self._compute_bounds_n_layers_out(1, [lb], [ub], [W_NA], [b], [W_in], [ilb], [iub], verbose)
        if verbose:
            print ('--------- lb --------')
            print (lb)
            print ('--------- WA --------')
            print (W_A)
            print ('--------- W_NA --------')
            print (W_NA)
            print ('--------- item1 --------')
            print (prev_layer_bounds)
            print ('------------------------')
        W_prod = np.matmul(W_prev , W_A)
        b_prod = np.matmul(b_prev, W_A)
        W_in_prod = np.matmul(W_in_prev, W_A)
        lbs_new = lbs[1:]
        ubs_new = ubs[1:]
        Ws_new = [W_prod] + Ws[2:]
        biases_new = [b_prod] + biases[2:]
        W_ins_new = [W_in_prod] + W_ins[2:]
        #print (type(ilbs[1:]))
        deeper_bounds = self._compute_bounds_n_layers_out(n-1, lbs_new, ubs_new, Ws_new, biases_new , W_ins_new, ilbs[1:], iubs[1:], verbose)
        #W_in_max = np.maximum(W_in, 0.0)
        #W_in_min = np.minimum(W_in, 0.0)
        #return (prev_layer_bounds[0] + deeper_bounds[0] + np.matmul(ilb, W_in_max) + np.matmul(iub, W_in_min) \
        #    , prev_layer_bounds[1] + deeper_bounds[1] + np.matmul(iub, W_in_max) + np.matmul(ilb, W_in_min) )
        return (prev_layer_bounds[0] + deeper_bounds[0] \
            , prev_layer_bounds[1] + deeper_bounds[1])
        
    def populate_ranges_tight(self, stimulus_step, stimulus_upper = 2, stimulus_lower = -2, verbose = False):
        assert (stimulus_upper >= stimulus_lower)
        u_lists = []
        l_lists = []
        u_lists.append(self.dataholder.init_state)
        l_lists.append(self.dataholder.init_state) # the initial upper/lower bounds
        
        sim_W_rec_reg = self.dataholder.W_rec
        sim_W_in_reg  = self.dataholder.W_in
        
        input_max = np.array([stimulus_upper, 0])
        input_min = np.array([stimulus_lower, 0])

        Ws = [sim_W_rec_reg]
        biases = [self.dataholder.b_rec]
        W_ins = [sim_W_in_reg]

        for idx in range(stimulus_step):            
            if verbose:
                print (f'>>>>>>>>>>>>>> calling to estimatiate layer: {idx+1}')
            lb, ub = self._compute_bounds_n_layers(idx+1, l_lists, u_lists, Ws , biases, W_ins, input_min, input_max, verbose = (verbose if idx >= 1 else False))
            u_lists.insert(0, ub) #u_lists.append(ub)
            l_lists.insert(0, lb) #l_lists.append(lb)
            Ws.append(sim_W_rec_reg)
            biases.append(self.dataholder.b_rec)
            W_ins.append(sim_W_in_reg)
        
        u_lists.reverse()
        l_lists.reverse()
        self.u_lists = u_lists
        self.l_lists = l_lists
        self.stimulus_step = stimulus_step

    def _compute_bounds_n_layers(self, n, lbs, ubs, Ws, biases, W_ins, ilb, iub, verbose):
        assert (n == len(lbs))
        assert (n == len(ubs)) # W is just a matrix, bias is just a vector
        assert (n == len(Ws))
        assert (n == len(biases))
        assert (n == len(W_ins))
        lb = lbs[0]
        ub = ubs[0]
        W = Ws[0]
        b = biases[0]
        W_in = W_ins[0]
        #print (W.shape)
        # base case
        #print (f'<<<< n:{n} , verbose:{verbose} ')
        if n == 1:
            naive_ia_bounds = self._interval_arithmetic(lb, ub, W, b, W_in, ilb, iub)
            if verbose:
                print ('--------- IA naive call --------')
                print ('--------- lb --------')
                print (lb)
                print ('--------- ub --------')
                print (ub)
                print ('--------- W --------')
                print (W)
                print ('--------- b --------')
                print (b)
                print ('--------- W_in --------')
                print (W_in)
                print ('--------- ilb --------')
                print (ilb)
                print ('--------- iub --------')
                print (iub)
                print ('---------- result bounds ------')
                print (naive_ia_bounds)
                print ('--------- END of IA naive call --------')

            return naive_ia_bounds
        # recursive case
        W_prev = Ws[1]
        b_prev = biases[1]
        W_in_prev = W_ins[1]
        W_A = np.transpose(np.transpose(W) * (lb > 0))
        W_NA = np.transpose(np.transpose(W) * (lb <= 0))
        prev_layer_bounds = self._compute_bounds_n_layers(1, [lb], [ub], [W_NA], [b], [W_in], ilb, iub, verbose)
        if verbose:
            print ('--------- lb --------')
            print (lb)
            print ('--------- WA --------')
            print (W_A)
            print ('--------- W_NA --------')
            print (W_NA)
            print ('--------- item1 --------')
            print (prev_layer_bounds)
            print ('------------------------')
        W_prod = np.matmul(W_prev , W_A)
        b_prod = np.matmul(b_prev, W_A)
        W_in_prod = np.matmul(W_in_prev, W_A)
        lbs_new = lbs[1:]
        ubs_new = ubs[1:]
        Ws_new = [W_prod] + Ws[2:]
        biases_new = [b_prod] + biases[2:]
        W_ins_new = [W_in_prod] + W_ins[2:]
        deeper_bounds = self._compute_bounds_n_layers(n-1, lbs_new, ubs_new, Ws_new, biases_new , W_ins_new, ilb, iub, verbose)
        #W_in_max = np.maximum(W_in, 0.0)
        #W_in_min = np.minimum(W_in, 0.0)
        #return (prev_layer_bounds[0] + deeper_bounds[0] + np.matmul(ilb, W_in_max) + np.matmul(iub, W_in_min) \
        #    , prev_layer_bounds[1] + deeper_bounds[1] + np.matmul(iub, W_in_max) + np.matmul(ilb, W_in_min) )
        return (prev_layer_bounds[0] + deeper_bounds[0] \
            , prev_layer_bounds[1] + deeper_bounds[1])
    
    @staticmethod
    def _output_interval_arithmetic(lb, ub, W, b):
        W_max = np.maximum(W, 0.0)
        W_min = np.minimum(W, 0.0)
        lb = np.maximum(lb,0.0)
        ub = np.maximum(ub,0.0)
        new_lb = np.matmul(lb, W_max) + np.matmul(ub, W_min) + b
        new_ub = np.matmul(ub, W_max) + np.matmul(lb, W_min) + b
        return new_lb, new_ub


    @staticmethod
    def _interval_arithmetic(lb, ub, W, b, W_in, ilb, iub):
        W_max = np.maximum(W, 0.0)
        W_min = np.minimum(W, 0.0)
        W_in_max = np.maximum(W_in, 0.0)
        W_in_min = np.minimum(W_in, 0.0)
        lb = np.maximum(lb,0.0)
        ub = np.maximum(ub,0.0)
        #print (f'lb : {lb}')
        #print (f'W_max : {W_max}')
        new_lb = np.matmul(lb, W_max) + np.matmul(ub, W_min) + \
                 np.matmul(ilb, W_in_max) + np.matmul(iub, W_in_min) + b
        new_ub = np.matmul(ub, W_max) + np.matmul(lb, W_min) + \
                 np.matmul(iub, W_in_max) + np.matmul(ilb, W_in_min) + b
        return new_lb, new_ub
            

    def compute_stable_relus(self):
        stables = 0
        for idx in range(self.stimulus_step):
            st = np.sum(self.u_lists[idx] * self.l_lists[idx] > 0)
            stables += st
            print ('layer {idx} : #{st} stables '.format(idx = idx, st = st))
        return stables
        
    def print_signs(self):
        self.sign_lists = []
        for idx in range(self.stimulus_step):
            self.sign_lists.append([]) # make a new frame
            for sidx in range(self.u_lists[idx].shape[0]):
                ub = self.u_lists[idx][sidx]
                lb = self.l_lists[idx][sidx]
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

    def show_range(self, nidx, depth):
        for idx in range(depth):
            print ('{lb} --> {ub}'.format(lb =  self.l_lists[idx][nidx], ub = self.u_lists[idx][nidx]))

    def plot_range(self, nidx):
        ubs = []
        lbs = []
        for idx in range(0, len(self.u_lists)):
            ubs.append(self.u_lists[idx][nidx])
            lbs.append(self.l_lists[idx][nidx])
        xlabel = range(len(self.u_lists))
        plt.plot(xlabel, lbs, 'r', xlabel, ubs, 'b')
        plt.title(f'S{nidx}')
        plt.show()

    def plot_output_range(self):
        xlabel = range(len(self.output_u_lists))
        plt.plot(xlabel, self.output_l_lists, 'r--', xlabel, self.output_u_lists, 'b--')
        plt.title('Output')
        plt.show()

        

    def check_range_converge(self):
        for idx in range(1, len(self.u_lists)):
            ubs = self.u_lists[idx]
            ubs_prev = self.u_lists[idx-1]
            lbs = self.l_lists[idx]
            lbs_prev = self.l_lists[idx-1]
            print (f'ubs:{ubs}')
            #print (f'lbs:{lbs}')
            if (ubs < ubs_prev).all() and (lbs > lbs_prev).all():
                print (f"Converged at #{idx}")
                return idx
        print ("Not converging")
        return 0
        
    def get_pos_vec(self):
        if len(self.l_lists) < 50:
            print ('Warning: this is not the final output stage')
        lb = self.l_lists[-1]
        ub = self.u_lists[-1]
        same_sign = lb * ub >= 0
        converge = ub-lb<0.1
        if not same_sign.all() or not converge.all():
            print ('Warning: seems not converged!')
        return (lb > 0).astype(float)
        
    
            
    

class RangeEstimationClipping(RangeEstimation):
    def __init__(self, dataholder, clipping):
        super(RangeEstimationClipping,self).__init__(dataholder)
        self.clipping = clipping
    
    
    def populate_ranges_tight_output(self, stimulus_step, stimulus_upper = 2, stimulus_lower = -2, response_step = 5, verbose = False):
        assert (stimulus_upper >= stimulus_lower)
        u_lists = []
        l_lists = []
        u_lists.append(self.dataholder.init_state)
        l_lists.append(self.dataholder.init_state) # the initial upper/lower bounds

        output_u_lists = []
        output_l_lists = []
        
        sim_W_rec_reg = self.dataholder.W_rec
        sim_W_in_reg  = self.dataholder.W_in
        sim_W_out_reg = self.dataholder.W_out
        b_out = self.dataholder.b_out
        
        input_max = np.array([stimulus_upper, 0.0])
        input_min = np.array([stimulus_lower, 0.0])

        range2_input_max = np.array([0.0, 1.0])
        range2_input_min = np.array([0.0, 1.0])

        Ws = [sim_W_rec_reg]
        biases = [self.dataholder.b_rec]
        W_ins = [sim_W_in_reg]

        ilbs = [input_min]
        ulbs = [input_max]

        for idx in range(stimulus_step+response_step): # stimulus stage   
            if verbose:
                print (f'>>>>>>>>>>>>>> calling to estimatiate layer: {idx+1}')
            lb, ub = self._compute_bounds_n_layers_out(idx+1, l_lists, u_lists, Ws , biases, W_ins, ilbs, ulbs, verbose = (verbose if idx >= 1 else False))
            #print (idx)
            #print (ub.shape)
            #print (lb.shape)
            #if idx >= stimulus_step:
            #    Ws_output = [sim_W_out_reg] + Ws[1:]
            #    biases_output = [b_out] + biases[1:]
            #    olb, oub = self._compute_bounds_n_layers_out(idx+1, l_lists, u_lists, Ws_output , biases_output, W_ins, ilbs, ulbs, verbose = (verbose if idx >= 1 else False))
            #    print (idx)
            #    print (olb.shape)
            #    print (oub.shape)
            #    output_u_lists.append(oub)
            #    output_l_lists.append(olb)

            # deal with the next level
            u_lists.insert(0, ub) #u_lists.append(ub)
            l_lists.insert(0, lb) #l_lists.append(lb)
            Ws.append(sim_W_rec_reg)
            biases.append(self.dataholder.b_rec)
            W_ins.append(sim_W_in_reg)
            ilbs.insert(0, input_min if idx < stimulus_step-1 else range2_input_min)
            ulbs.insert(0, input_max if idx < stimulus_step-1 else range2_input_max)

        u_lists.reverse()
        l_lists.reverse()
        self.u_lists = u_lists
        self.l_lists = l_lists
        self.stimulus_step = stimulus_step

        # compute the output range
        for idx in range(stimulus_step + response_step):
            state_lb = l_lists[idx]
            state_ub = u_lists[idx]
            olb, oub = self._output_interval_arithmetic(state_lb, state_ub, sim_W_out_reg, b_out, self.clipping)
            #print (olb.shape)
            output_u_lists.append(oub)
            output_l_lists.append(olb)
            # do a simple IA : W+*u + W-*l + b
        self.output_u_lists = output_u_lists
        self.output_l_lists = output_l_lists

    def _compute_bounds_n_layers_out(self, n, lbs, ubs, Ws, biases, W_ins, ilbs, iubs, verbose):
        assert (n == len(lbs))
        assert (n == len(ubs)) # W is just a matrix, bias is just a vector
        assert (n == len(Ws))
        assert (n == len(biases))
        assert (n == len(W_ins))
        assert (n == len(ilbs))
        assert (n == len(iubs))
        lb = lbs[0]
        ub = ubs[0]
        W = Ws[0]
        b = biases[0]
        W_in = W_ins[0]
        ilb = ilbs[0]
        iub = iubs[0]
        #print (W.shape)
        # base case
        #print (f'<<<< n:{n} , verbose:{verbose} ')
        if n == 1:
            naive_ia_bounds = self._interval_arithmetic(lb, ub, W, b, W_in, ilb, iub, self.clipping)
            if verbose:
                print ('--------- IA naive call --------')
                print ('--------- lb --------')
                print (lb)
                print ('--------- ub --------')
                print (ub)
                print ('--------- W --------')
                print (W)
                print ('--------- b --------')
                print (b)
                print ('--------- W_in --------')
                print (W_in)
                print ('--------- ilb --------')
                print (ilb)
                print ('--------- iub --------')
                print (iub)
                print ('---------- result bounds ------')
                print (naive_ia_bounds)
                print ('--------- END of IA naive call --------')

            return naive_ia_bounds
        # recursive case
        W_prev = Ws[1]
        b_prev = biases[1]
        W_in_prev = W_ins[1]
        W_A = np.transpose(np.transpose(W) * (lb > 0) * (ub < self.clipping))
        W_NA = W - W_A # np.transpose(np.transpose(W) * (lb <= 0))
        prev_layer_bounds = self._compute_bounds_n_layers_out(1, [lb], [ub], [W_NA], [b], [W_in], [ilb], [iub], verbose)
        if verbose:
            print ('--------- lb --------')
            print (lb)
            print ('--------- WA --------')
            print (W_A)
            print ('--------- W_NA --------')
            print (W_NA)
            print ('--------- item1 --------')
            print (prev_layer_bounds)
            print ('------------------------')
        W_prod = np.matmul(W_prev , W_A)
        b_prod = np.matmul(b_prev, W_A)
        W_in_prod = np.matmul(W_in_prev, W_A)
        lbs_new = lbs[1:]
        ubs_new = ubs[1:]
        Ws_new = [W_prod] + Ws[2:]
        biases_new = [b_prod] + biases[2:]
        W_ins_new = [W_in_prod] + W_ins[2:]
        #print (type(ilbs[1:]))
        deeper_bounds = self._compute_bounds_n_layers_out(n-1, lbs_new, ubs_new, Ws_new, biases_new , W_ins_new, ilbs[1:], iubs[1:], verbose)
        #W_in_max = np.maximum(W_in, 0.0)
        #W_in_min = np.minimum(W_in, 0.0)
        #return (prev_layer_bounds[0] + deeper_bounds[0] + np.matmul(ilb, W_in_max) + np.matmul(iub, W_in_min) \
        #    , prev_layer_bounds[1] + deeper_bounds[1] + np.matmul(iub, W_in_max) + np.matmul(ilb, W_in_min) )
        return (prev_layer_bounds[0] + deeper_bounds[0] \
            , prev_layer_bounds[1] + deeper_bounds[1])
        
    def populate_ranges_tight(self, stimulus_step, stimulus_upper = 2, stimulus_lower = -2, verbose = False):
        assert (stimulus_upper >= stimulus_lower)
        u_lists = []
        l_lists = []
        u_lists.append(self.dataholder.init_state)
        l_lists.append(self.dataholder.init_state) # the initial upper/lower bounds
        
        sim_W_rec_reg = self.dataholder.W_rec
        sim_W_in_reg  = self.dataholder.W_in
        
        input_max = np.array([stimulus_upper, 0])
        input_min = np.array([stimulus_lower, 0])

        Ws = [sim_W_rec_reg]
        biases = [self.dataholder.b_rec]
        W_ins = [sim_W_in_reg]

        for idx in range(stimulus_step):            
            if verbose:
                print (f'>>>>>>>>>>>>>> calling to estimatiate layer: {idx+1}')
            lb, ub = self._compute_bounds_n_layers(idx+1, l_lists, u_lists, Ws , biases, W_ins, input_min, input_max, verbose = (verbose if idx >= 1 else False))
            u_lists.insert(0, ub) #u_lists.append(ub)
            l_lists.insert(0, lb) #l_lists.append(lb)
            Ws.append(sim_W_rec_reg)
            biases.append(self.dataholder.b_rec)
            W_ins.append(sim_W_in_reg)
        
        u_lists.reverse()
        l_lists.reverse()
        self.u_lists = u_lists
        self.l_lists = l_lists
        self.stimulus_step = stimulus_step

    def _compute_bounds_n_layers(self, n, lbs, ubs, Ws, biases, W_ins, ilb, iub, verbose):
        assert (n == len(lbs))
        assert (n == len(ubs)) # W is just a matrix, bias is just a vector
        assert (n == len(Ws))
        assert (n == len(biases))
        assert (n == len(W_ins))
        lb = lbs[0]
        ub = ubs[0]
        W = Ws[0]
        b = biases[0]
        W_in = W_ins[0]
        #print (W.shape)
        # base case
        #print (f'<<<< n:{n} , verbose:{verbose} ')
        if n == 1:
            naive_ia_bounds = self._interval_arithmetic(lb, ub, W, b, W_in, ilb, iub, self.clipping)
            if verbose:
                print ('--------- IA naive call --------')
                print ('--------- lb --------')
                print (lb)
                print ('--------- ub --------')
                print (ub)
                print ('--------- W --------')
                print (W)
                print ('--------- b --------')
                print (b)
                print ('--------- W_in --------')
                print (W_in)
                print ('--------- ilb --------')
                print (ilb)
                print ('--------- iub --------')
                print (iub)
                print ('---------- result bounds ------')
                print (naive_ia_bounds)
                print ('--------- END of IA naive call --------')

            return naive_ia_bounds
        # recursive case
        W_prev = Ws[1]
        b_prev = biases[1]
        W_in_prev = W_ins[1]
        W_A = np.transpose(np.transpose(W) * (lb > 0) * (ub < self.clipping))
        W_NA = W - W_A # np.transpose(np.transpose(W) * (lb <= 0))
        prev_layer_bounds = self._compute_bounds_n_layers(1, [lb], [ub], [W_NA], [b], [W_in], ilb, iub, verbose)
        if verbose:
            print ('--------- lb --------')
            print (lb)
            print ('--------- WA --------')
            print (W_A)
            print ('--------- W_NA --------')
            print (W_NA)
            print ('--------- item1 --------')
            print (prev_layer_bounds)
            print ('------------------------')
        W_prod = np.matmul(W_prev , W_A)
        b_prod = np.matmul(b_prev, W_A)
        W_in_prod = np.matmul(W_in_prev, W_A)
        lbs_new = lbs[1:]
        ubs_new = ubs[1:]
        Ws_new = [W_prod] + Ws[2:]
        biases_new = [b_prod] + biases[2:]
        W_ins_new = [W_in_prod] + W_ins[2:]
        deeper_bounds = self._compute_bounds_n_layers(n-1, lbs_new, ubs_new, Ws_new, biases_new , W_ins_new, ilb, iub, verbose)
        #W_in_max = np.maximum(W_in, 0.0)
        #W_in_min = np.minimum(W_in, 0.0)
        #return (prev_layer_bounds[0] + deeper_bounds[0] + np.matmul(ilb, W_in_max) + np.matmul(iub, W_in_min) \
        #    , prev_layer_bounds[1] + deeper_bounds[1] + np.matmul(iub, W_in_max) + np.matmul(ilb, W_in_min) )
        return (prev_layer_bounds[0] + deeper_bounds[0] \
            , prev_layer_bounds[1] + deeper_bounds[1])
    
    @staticmethod
    def _output_interval_arithmetic(lb, ub, W, b, clipping):
        W_max = np.maximum(W, 0.0)
        W_min = np.minimum(W, 0.0)
        lb = np.minimum(np.maximum(lb,0.0), clipping)
        ub = np.minimum(np.maximum(ub,0.0), clipping)
        new_lb = np.matmul(lb, W_max) + np.matmul(ub, W_min) + b
        new_ub = np.matmul(ub, W_max) + np.matmul(lb, W_min) + b
        return new_lb, new_ub


    @staticmethod
    def _interval_arithmetic(lb, ub, W, b, W_in, ilb, iub, clipping):
        W_max = np.maximum(W, 0.0)
        W_min = np.minimum(W, 0.0)
        W_in_max = np.maximum(W_in, 0.0)
        W_in_min = np.minimum(W_in, 0.0)
        lb = np.minimum(np.maximum(lb,0.0), clipping)
        ub = np.minimum(np.maximum(ub,0.0), clipping)
        #print (f'lb : {lb}')
        #print (f'W_max : {W_max}')
        new_lb = np.matmul(lb, W_max) + np.matmul(ub, W_min) + \
                 np.matmul(ilb, W_in_max) + np.matmul(iub, W_in_min) + b
        new_ub = np.matmul(ub, W_max) + np.matmul(lb, W_min) + \
                 np.matmul(iub, W_in_max) + np.matmul(ilb, W_in_min) + b
        return new_lb, new_ub
            
             
            
# let's make a new (tighter estimator)

class RangeEstimationTriangleRegion(object):
    def __init__(self, dataholder, verbose = False):
        self.num_state = dataholder.num_state
        self.num_input = dataholder.num_input
        self.dataholder = dataholder
    
            
    def get_previous_populate_output_status(self):
        """ return (stable?, pos? )"""
        olb = self.output_l_lists[-1]
        oub = self.output_u_lists[-1]
        if oub - olb < 1.0:
            # this is stable
            return True, oub, olb
        return False, oub, olb

    def populate_polytope_state_range(self, stimulus_step, state_ub_list, state_lb_list, \
        stimulus_upper = 2, stimulus_lower = -2, response_step = 5, verbose = False  ):

        assert (stimulus_upper >= stimulus_lower)
        u_lists = []
        l_lists = []
        u_lists.append(state_ub_list)
        l_lists.append(state_lb_list) # the initial upper/lower bounds

        output_u_lists = []
        output_l_lists = []
        
        sim_W_rec_reg = self.dataholder.W_rec
        sim_W_in_reg  = self.dataholder.W_in
        sim_W_out_reg = self.dataholder.W_out
        b_out = self.dataholder.b_out
        
        input_max = np.array([stimulus_upper, 0])
        input_min = np.array([stimulus_lower, 0])

        range2_input_max = np.array([0.0, 1.0])
        range2_input_min = np.array([0.0, 1.0])

        Ws = [sim_W_rec_reg]
        biases = [self.dataholder.b_rec]
        W_ins = [sim_W_in_reg]

        ilbs = [input_min]
        ulbs = [input_max]

        for idx in range(stimulus_step+response_step): # stimulus stage   
            if verbose:
                print (f'>>>>>>>>>>>>>> calling to estimatiate layer: {idx+1}')
            lb, ub = self._compute_bounds_n_layers_out(idx+1, l_lists, u_lists, Ws , biases, W_ins, ilbs, ulbs, verbose = (verbose if idx >= 1 else False))
            #print (idx)
            #print (ub.shape)
            #print (lb.shape)
            #if idx >= stimulus_step:
            #    Ws_output = [sim_W_out_reg] + Ws[1:]
            #    biases_output = [b_out] + biases[1:]
            #    olb, oub = self._compute_bounds_n_layers_out(idx+1, l_lists, u_lists, Ws_output , biases_output, W_ins, ilbs, ulbs, verbose = (verbose if idx >= 1 else False))
            #    print (idx)
            #    print (olb.shape)
            #    print (oub.shape)
            #    output_u_lists.append(oub)
            #    output_l_lists.append(olb)

            # deal with the next level
            u_lists.insert(0, ub) #u_lists.append(ub)
            l_lists.insert(0, lb) #l_lists.append(lb)
            Ws.append(sim_W_rec_reg)
            biases.append(self.dataholder.b_rec)
            W_ins.append(sim_W_in_reg)
            ilbs.insert(0, input_min if idx < stimulus_step-1 else range2_input_min)
            ulbs.insert(0, input_max if idx < stimulus_step-1 else range2_input_max)

        u_lists.reverse()
        l_lists.reverse()
        self.u_lists = u_lists
        self.l_lists = l_lists
        self.stimulus_step = stimulus_step

        # compute the output range
        for idx in range(stimulus_step + response_step):
            state_lb = l_lists[idx]
            state_ub = u_lists[idx]
            olb, oub = self._output_interval_arithmetic(state_lb, state_ub, sim_W_out_reg, b_out)
            #print (olb.shape)
            output_u_lists.append(oub)
            output_l_lists.append(olb)
            # do a simple IA : W+*u + W-*l + b
        self.output_u_lists = output_u_lists
        self.output_l_lists = output_l_lists
