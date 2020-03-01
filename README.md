# VERRNN
Verification of Recurrent Neural Networks for Cognitive Tasks via Reachability Analysis

https://github.com/nnarodytska/VERRNN/blob/master/rnn_veri.pdf

# Code Structure

  * verrnn : Our methods presented in the paper
  * marabou: Running marabou on the unrolled RNN (need to have marabou and its python API compiled and installed)
  * nnv    : NNV (need to have MATLAB and the nnv toolbox installed)
  * N7_L1_r11 : the model of 7-neuron RNN under verification

# To Run the Experiments

   Work in progress...

## Our Tool

## NNV for comparison

First, please install [NNV toolbox](https://github.com/verivital/nnv) for Matlab.

In `nnv` subfolder, those scripts starting with `p1` are for property 1. Those starting with `p2` are for property 2.
Unlike the other experiments, we don't have the outer loop to test all test ranges in one run (because usually we need to interrupt the execution manually as we don't have a time-out mechanism built into the scripts). So for each run you need to change the test range selection in the line like `range_select_idx = ?;`, or you can manually specify the ranges in the lines
```
ilb = ???; % input lower bound
iub = ???; % input upper bound
```
in the beginning of the scripts.

## Marabou for comparison

Please first have [Marabou](https://github.com/NeuralNetworkVerification/Marabou/tree/master) and its python API installed. And then go to `marabou`, run `python bmc_marabou.py`. Results will be stored at `result.log`
We only experimented Marabou with Property 1.


