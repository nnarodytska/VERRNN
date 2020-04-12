import dataload
import smtbmc
import time


# -----------------------------------------------
# configurations
modelpath = "../deep_monkey/results/N7_L1_r11/"
testranges = [\
(0.000087 ,0.000115),
(0.027858 ,0.036758),
(0.111430 ,0.147033),
(0.891444 ,1.176267),
(0.000071 ,0.000142),
(0.022627 ,0.045255),
(0.090510 ,0.181019),
(0.724077 ,1.448155),
(0.000062 ,0.000163),
(0.019698 ,0.051984),
(0.078793 ,0.207937),
(0.630346 ,1.663493),
(-0.000115,-0.000087),
(-0.036758,-0.027858),
(-0.147033,-0.111430),
(-1.176267,-0.891444),
(-0.000142,-0.000071),
(-0.045255,-0.022627),
(-0.181019,-0.090510),
(-1.448155,-0.724077),
(-0.000163,-0.000062),
(-0.051984,-0.019698),
(-0.207937,-0.078793),
(-1.663493,-0.630346)]


#modelpath = "/home/hongce/deep_monkey/results/2019-06-17_0_neurons_10_steps_110.0_noise_0.3"
#originalPath = "/home/hongce/deep_monkey/results/2019-06-04_0_neurons_10_steps_110.0_noise_0.3"

# -----------------------------------------------
# main function

def main():
    weightsObj = dataload.LOADER(modelpath, 10)
    estimator = smtbmc.RangeEstimation(weightsObj)
    estimator.populate_ranges_tight(50, 1.15, 0.9)
    estimator.compute_stable_relus()
    estimator.print_signs()
    estimator.check_range_converge()
    #for idx in range(10):
    #    estimator.plot_range(idx)
    # check if we use this range, can we prove?
    
    for (ilb, iub) in testranges:
        t0 = time.time()
        estimator2 = smtbmc.RangeEstimation(weightsObj)
        estimator2.populate_ranges_tight_output(50, iub, ilb, response_step = 50)
        oub, olb = estimator2.get_previous_populate_output_status()
        t1 = time.time()
        print ((ilb, iub),' ---> ', end = '')
        print (olb, oub, end = '  T:')
        print (t1-t0)
    
    #estimator2.compute_stable_relus()
    #estimator2.print_signs()
    #estimator2.check_range_converge()
    
    #for idx in range(10):
    #    estimator2.plot_range(idx)
    #estimator2.plot_output_range()
    
if __name__ == "__main__":
    main()
