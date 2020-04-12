import dataload
import smtbmc
import fpspacer

relu_stable_modelpath = "/home/hongce/deep_monkey/results/2019-06-17_0_neurons_10_steps_110.0_noise_0.3"
originalPath = "/home/hongce/deep_monkey/results/2019-06-04_0_neurons_10_steps_110.0_noise_0.3"


def main(mp, ub, lb):
    weightsObj = dataload.LOADER(mp, 10)
    estimator = smtbmc.RangeEstimation(weightsObj)
    estimator.populate_ranges_tight(50, ub, lb)
    estimator.compute_stable_relus()
    stable_signs = estimator.print_signs()
    pos_list = []
    for idx,sign in enumerate(stable_signs[-1]):
        if sign == '0':
            pos_list.append(idx)
    print ('Bound estimation conjecture: negative stable idx : ', pos_list)
    
    # start spacer
    spacer_solver = fpspacer.ChcEncoder(weightsObj, precision = 2**16, relu_extra_var = True)
    spacer_solver.add_bounded_relu_stable_info(stable_signs, 10)
    #spacer_solver.EncodeReluStableProperty(pos_list, ilb = lb, iub = ub , start = 8)
    spacer_solver.EncodeReluStableProperty(range(10), ilb = lb, iub = ub , start = 8)
    
        
if __name__ == "__main__":
    main(mp = relu_stable_modelpath, ub = 1.15, lb = 0.9)

