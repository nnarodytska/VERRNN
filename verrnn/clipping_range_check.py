import dataload
import smtbmc
import random

# -----------------------------------------------
# configurations

clipping3_modelpath = "/home/hongce/deep_monkey/results/2019-06-24_0_neurons_10_steps_110.0_noise_0.3"
clipping4_modelpath = "/home/hongce/deep_monkey/results/2019-06-25_0_neurons_10_steps_110.0_noise_0.3"

# -----------------------------------------------
# main function

def main(mp, clip, ub, lb):
    def gen_stimulus(idx): # [0,1] -> [lb,ub]
        n = random.random()*(ub-lb)+lb
        assert (lb <= n <= ub)
        return n
    
    weightsObj = dataload.LOADER(mp, 10)
    estimator = smtbmc.RangeEstimationClipping(weightsObj, clip)
    estimator.populate_ranges_tight_output(50, ub, lb, response_step = 30)
    estimator.compute_stable_relus()
    estimator.print_signs()
    for idx in range(10):
        estimator.plot_range(idx)
    estimator.plot_output_range()
    
    # validation using sampling
    rsObj = smtbmc.RangeSampling(weightsObj)
    N = 1000
    for i in range(N):
        if i % (N/100) == 0 and i > 0:            
            print (i/N*100, '%', end = '')
            lbs, ubs = rsObj.get_node_output_range()
            print (min(lbs), max(ubs))
        rsObj.sample_a_point(50, gen_stimulus, 30, clipping=clip)
        rsObj.update_range()
    
    lbs, ubs = rsObj.get_node_output_range()
    print ('llb',min(lbs))
    print ('uub',max(ubs))   
    
    if rsObj.check_bounds_static_estimator(estimator.l_lists, estimator.u_lists):
        print ('Bounds checked!' )
    
    
    
if __name__ == "__main__":
    main(clipping4_modelpath, clip = 4.0, ub = 2.0, lb = -2.0)

