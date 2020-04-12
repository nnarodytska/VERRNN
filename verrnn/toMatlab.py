import dataload
import polytope
import copy
import time

# okay we don't import again
np = polytope.np
List = polytope.List
Sequence = polytope.Sequence

import scipy.io

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

if __name__ == "__main__":
    testranges = np.array(testranges)
    weightsObj = dataload.LOADER(modelpath, 10)
    weightsObj.toMatlab('rnn.mat')
    scipy.io.savemat('ranges.mat', {'testranges':testranges})

