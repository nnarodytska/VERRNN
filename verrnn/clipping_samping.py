import dataload
import smtbmc
import random

# -----------------------------------------------
# configurations

clipping3_modelpath = "/home/hongce/deep_monkey/results/2019-06-24_0_neurons_10_steps_110.0_noise_0.3"
clipping4_modelpath = "/home/hongce/deep_monkey/results/2019-06-25_0_neurons_10_steps_110.0_noise_0.3"
relu_stable_modelpath = "/home/hongce/deep_monkey/results/2019-06-17_0_neurons_10_steps_110.0_noise_0.3"
originalPath = "/home/hongce/deep_monkey/results/2019-06-04_0_neurons_10_steps_110.0_noise_0.3"

# -----------------------------------------------
# main function

def gen_stimulus(idx): # [0,1] -> [-2,2]
    n = random.random()*4-2
    assert (-2 <= n <= 2)
    return n

def main():
    N = 5000
    random.seed(2000)
    weightsObj = dataload.LOADER(clipping4_modelpath, 10, clipping = 4.0)
    #weightsObj = dataload.LOADER(relu_stable_modelpath, 10)
    #weightsObj = dataload.LOADER(originalPath, 10)
    rsObj = smtbmc.RangeSampling(weightsObj)
    for i in range(N):
        if i % (N/100) == 0 and i > 0:            
            print (i/N*100, '%', end = '')
            lbs, ubs = rsObj.get_node_output_range()
            print (min(lbs), max(ubs))
        rsObj.sample_a_point(50, gen_stimulus, 30, clipping=4.0)
        rsObj.update_range()

    rsObj.print_signs()    
    lbs, ubs = rsObj.get_node_output_range()
    print ('lbs:', lbs)
    print ('ubs:', ubs)
    print (min(lbs))
    print (max(ubs))   
     
    
if __name__ == "__main__":
    main()
