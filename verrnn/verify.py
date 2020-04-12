import dataload
import smtbmc

# -----------------------------------------------
# configurations
modelpath = "../deep_monkey/results/N7_L1_r11/"

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
    
    estimator2 = smtbmc.RangeEstimation(weightsObj)
    estimator2.populate_ranges_tight_output(50, 0.000115, 0.000087, response_step = 50)
    estimator2.compute_stable_relus()
    estimator2.print_signs()
    estimator2.check_range_converge()
    #for idx in range(10):
    #    estimator2.plot_range(idx)
    #estimator2.plot_output_range()
    
if __name__ == "__main__":
    main()
