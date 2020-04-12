import matplotlib.pyplot as plt
import dataload
import smtbmc
import numpy as np
import pickle

# -----------------------------------------------
# configurations

modelpath = "/home/hongce/deep_monkey/results/2019-06-17_0_neurons_10_steps_110.0_noise_0.3"
originalPath = "/home/hongce/deep_monkey/results/2019-06-04_0_neurons_10_steps_110.0_noise_0.3"
N8Path = "/home/hongce/deep_monkey/results/N8_relu_stable_0.000004"
N7Path = "/home/hongce/deep_monkey/results/N7_relu_stable_0.0000009"
N7NostablePath = "/home/hongce/deep_monkey/results/N7_no_stable"

datapath_relu_stable = "data/stable_range_2019-06-17.data"
datapath_unstable = "data/stable_range_2019-06-04.data"
datapath_N8 = "data/stable_n8_2019-07-01.data"
datapath_N7 = "data/stable_n7_2019-07-01.data"
datapath_N7stable = "data/stable_n7_relu_stable_2019-07-01.data"

def testRange(upper, lower, weights):
    estimator = smtbmc.RangeEstimation(weights)
    estimator.populate_ranges_tight_output(50, upper, lower, response_step = 30)
    #estimator2.compute_stable_relus()
    #estimator2.print_signs()
    olb = estimator.output_l_lists[-1]
    oub = estimator.output_u_lists[-1]
    if oub - olb < 1.0 and oub * olb >= 0:
        # this is stable
        return True, olb > 0
    return False, None

def testAllRanges(step, weights, llb, uub, breakWhenNotConverging = True):
    # pos ranges
    bounds_range = {}
    for ub in np.arange(uub,0.0,-step):
        bounds_range[ub] = (ub, None)
        print ('ub:', ub, end = ' ')
        for lb in np.arange(ub, llb, -step):
            res, pos = testRange(ub, lb, weights)
            if res:
                print ('*', end = '', flush=True)
                if lb <= bounds_range[ub][0]:
                    bounds_range[ub] = (lb, pos)
            else:
                if (breakWhenNotConverging):
                    break
                print (' ', end = '', flush=True)
                
        print ()
    
    for lb in np.arange(llb,0.0,step):
        bounds_range[lb] = (lb, None)
        print ('lb:', lb, end = ' ')
        for ub in np.arange(lb, uub, step):
            res, pos = testRange(ub, lb, weights)
            if res:
                print ('*', end = '', flush=True)
                if ub >= bounds_range[lb][0]:
                    bounds_range[lb] = (ub, pos)
            else:
                if (breakWhenNotConverging):
                    break
                print (' ', end = '', flush=True)
        print ()
    return bounds_range


def prepare_data():
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
    (-0.000163,-0.000062),
    (-0.051984,-0.019698),
    (-0.207937,-0.078793),
    (0.362039 , 0.724077),
    (0.315173 ,	0.831746),
    (-0.724077,	-0.362039),
    (-0.831746,	-0.315173),
    (0.500000,	2.000000),
    (-1.448155,-0.724077),
    (-1.663493,-0.630346)
    ]
    testranges.sort(key=lambda p:p[0])
    bounds_range = {}
    for idx,(lb,ub) in enumerate(testranges):
        bounds_range[lb] = (ub, 1)
    plot_stable_range(bounds_range)
        


def plot_stable_range(bounds_range):
    bars = []
    for b1, (b2, pos) in bounds_range.items():
        lb = min(b1, b2)
        ub = max(b1, b2)
        bars.append((lb, ub, pos))
    
        
    bars.sort(key=lambda p:p[0]*1000+p[1])

    single_point = []
    for idx, (lb, ub, pos) in enumerate(bars):
        print (lb,ub, pos)
        if lb == ub:
            single_point.append((idx,lb, pos))

    bottom0 = [b[0] for b in bars if not b[2] and b[0] < 1 ]
    top0 = [b[1]-b[0] for b in bars if not b[2] and b[0] < 1]
    bottom0_mis1 = [b[0] for b in bars if not b[2] and b[0] >= 1 ]
    top0_mis1 = [b[1]-b[0] for b in bars if not b[2] and b[0] >= 1]



    bottom1 = [b[0] for b in bars if b[2] ]
    top1 = [b[1]-b[0] for b in bars if b[2]]
    xlabels0 = range(len(bottom0))
    xlabels1 = range(len(bottom0), len(bottom0) + len(bottom1))
    xlabels2 = range(len(bottom0) + len(bottom1), len(bottom0) + len(bottom1) + len(bottom0_mis1))

    plt.bar(xlabels0, top0, width=0.8, label='out -> -1', color='red', bottom=bottom0)
    plt.bar(xlabels1, top1, width=0.8, label='out -> 1', color='blue', bottom=bottom1)
    plt.bar(xlabels2, top0_mis1, width=0.8, label='out -> -1', color='red', bottom=bottom0_mis1)

    plt.plot([x for x,y,pos in single_point if not pos], [y for x,y,pos in single_point if not pos],  'ro' )
    plt.plot([x for x,y,pos in single_point if pos], [y for x,y,pos in single_point if pos],  'bo' )

    plt.show()


if __name__ == "__main__":
    prepare_data()
