import dataload
import polytope
import copy
import time

# okay we don't import again
np = polytope.np
List = polytope.List
Sequence = polytope.Sequence

# -----------------------------------------------
# configurations
modelpath = "../N7_L1_r11/"


testranges = [\
(0.000087 ,0.000115),
(0.027858 ,0.036758),
(0.111430 ,0.147033),
(0.019698 ,0.051984),
(0.078793 ,0.207937),
(0.315173 ,0.831746),
(-0.000115,-0.000087),
(-0.036758,-0.027858),
(-0.147033,-0.111430),
(-0.051984,-0.019698),
(-0.207937,-0.078793),
(-0.831746,-0.315173)]

pulseindices = [1,2,3,4]
pulseabsrange = (0.5,1.0)



# -----------------------------------------------
# Options
class cegar_option(polytope.DefaultOptions):
    def __init__(self):
        super().__init__()
        self.NumPolyToStartAbs = 500 # change it to 10 for cegar ?
        self.StartFixedpointIter = 0 # never start # change to 1000 for cegar ?
        self.PercentageOfFixedpointToContinue = 0.05
        self.AbsoluteFixedpointToContinue = 10
        self.FixedpointIternDelta1 = 2
        self.FixedpointIternDelta2 = 2
        self.RangeEstStart = 40
        self.externQhull = True
        self.FacetApproxBound = 4e4 # if the number of facet is this much (after reduce, if reduce enabled)
        self.ReduceDupFacet = True # whether to reduce the H-resp
        self.RdFacetBound = 1e5 # if the number of facet is > than this, will not reduce H-resp
        # volume based fixedpoint heuristics
        # layer based propagation computation


# -----------------------------------------------
# helpers
def get_init_polytope(weightsObj : dataload.LOADER, ilb : float, iub : float) -> List[polytope.convex_polytope] :
    vars = ['s%d' % s for s in range(weightsObj.num_state)] + ['x%d' % x for x in range(1)]
    initial_polytope = polytope.convex_polytope(vars)
    for idx, v in enumerate(weightsObj.init_state):
        v = 0.0 if v <= 0 else v
        initial_polytope.add_vi_range(idx,v, v)
    initial_polytope.add_vi_range(weightsObj.num_state, ilb, iub) # input range
    return initial_polytope

#    pltp_mgr = polytope.polytopes(weightsObj, input_idx=0)
# init polytope = get_init_polytope(weightsObj, ilb = ilb, iub = iub )

maxlayers_gone = 0
maxNeural = 0

def poly_prop_from_layer(timeoutFile:str,pltp_mgr : polytope.polytopes, start_layer : int, init_polys : List[polytope.convex_polytope] , \
    stimulus_total : int, response_total : int, ilb : float, iub : float, pulseidx:int, pulse_ilb:float, pulse_iub:float,  expect_output: int, options : cegar_option, \
    cegar_stack : List[int]) -> bool :
    """ expect_output: +1 -1, returns True if holds, otherwise False """

    global maxlayers_gone
    global maxNeural

    # print the cegar stack
    # HERE
    print ('BT stack: ', cegar_stack)
    
    weightsObj = pltp_mgr.dataholder

    # Configurations
    fixedpoint_layer = options.StartFixedpointIter

    # layer 0 : init 
    layer_polytope_map = [ init_polys ]
    # for each layer:
    for layer_idx in range(start_layer,stimulus_total+1):
        # ------------- PRINT --------------
        prev_layer_polys = layer_polytope_map[-1]
        print ('start on layer no.%d, #polytope = %d' % (layer_idx, len(prev_layer_polys)))
        

        # ------------- PROPAGATE ----------
        per_layer_option = copy.copy(options)
        per_layer_option.DoJoin = len(prev_layer_polys) > options.NumPolyToStartAbs
        per_layer_option.Invtest = layer_idx >= fixedpoint_layer and layer_idx > pulseidx
        per_layer_option.PropPreCheck = layer_idx >= options.RangeEstStart and layer_idx > pulseidx

        next_layer_polytopes, prev_polyResult, misc = \
            polytope.PolytopePropagateByLayer(timeoutFile, pltp_mgr, layer_idx, stimulus_total, response_total, \
            prev_layer_polytopes = prev_layer_polys, \
            ilb = ilb if layer_idx != pulseidx else pulse_ilb, \
            iub = iub if layer_idx != pulseidx else pulse_iub, \
            options=per_layer_option)
        
        # ------------- PRINT -------------
        print ("layer %d summary: " % layer_idx, end = '')
        npos = 0; nneg = 0; nunknown = 0; nnone = 0
        for poly_res in prev_polyResult:
            res = poly_res.prev
            if res == polytope.PolyPrev.POS:
                print ('+', end = '') ; npos += 1
            elif res == polytope.PolyPrev.NEG:
                print ('-', end = '') ; nneg += 1
            elif res == polytope.PolyPrev.UNKNOWN:
                print ('?', end = '') ; nunknown += 1
            else:
                print ('.', end = '') ; nnone += 1
        print ('\n---- pos / neg / unknown / numerical = %d / %d / %d / %d' % (npos, nneg, nunknown, nnone))
        
        # ------------- STATISTICS --------------
        maxlayers_gone = layer_idx
        if len(next_layer_polytopes) > maxNeural:
            maxNeural = len(next_layer_polytopes)
        # ------------- EOL LOGIC --------------
        if (expect_output > 0 and nneg > 0) or (expect_output < 0 and npos > 0):
            print ('CEX found!'); return False
        if len(next_layer_polytopes) == 0:
            print ('Procedure complete!'); return True
        
        # ------------- ADD INPUT RANGE -------------
        for poly in next_layer_polytopes:
            poly.add_vi_range(weightsObj.num_state, ilb if layer_idx+1 != pulseidx else pulse_ilb , iub if layer_idx+1 != pulseidx else pulse_iub) # add input range
        layer_polytope_map.append(next_layer_polytopes)


        # ------------- ADJUST Parameters -------------
        fp_ratio = misc.num_fixedpoint /  float(len(prev_layer_polys))
        if per_layer_option.Invtest and (misc.num_fixedpoint < options.NumPolyToStartAbs or \
            fp_ratio < options.PercentageOfFixedpointToContinue):
            if misc.num_fixedpoint >= options.NumPolyToStartAbs or fp_ratio >= options.PercentageOfFixedpointToContinue:
                fixedpoint_layer += options.FixedpointIternDelta1
            else:
                fixedpoint_layer += options.FixedpointIternDelta2
            print ('Defer computing fixedpoint to iter @ %d' % fixedpoint_layer)
    
    finalcheck = copy.copy(options)
    finalcheck.DoProp = False
    _ , last_layer_status, _ = polytope.PolytopePropagateByLayer(timeoutFile, pltp_mgr, layer_idx, stimulus_total, response_total, \
            next_layer_polytopes, ilb = ilb, iub = iub, options=finalcheck)

    # print summary of the final abstraction
    # HERE
    print ('Cegar id :',cegar_stack,"reaches end: status : ", end = '')
    npos = 0 ; nneg = 0; nunknown = 0
    for res in last_layer_status:
        if res == polytope.PolyPrev.POS: npos += 1
        elif res == polytope.PolyPrev.NEG: nneg += 1
        elif res == polytope.PolyPrev.UNKNOWN: nunknown += 1
    print ('pos / neg / unknown = %d / %d / %d' % (npos, nneg, nunknown))

    #CEGAR HERE:
    """
    for idx,status in enumerate(last_layer_status):
        poly = next_layer_polytopes[idx]
        if status == polytope.PolyPrev.UNKNOWN:
            assert (poly.attribute.is_abstract)
            print ('CEGAR : to poly @ %d')
            new_start_layer = poly.attribute.abs_poly_layer
            new_start_poly = poly.attribute.abs_poly_ref
            res = \
                poly_prop_from_layer(pltp_mgr, start_layer = new_start_layer, init_polys=[new_start_poly],\
                stimulus_total = stimulus_total, response_total = response_total, ilb = ilb, iub = iub,
                expect_output = expect_output, options = options, cegar_stack = cegar_stack+[new_start_layer])
            if not res:
                return False
    """
    for idx,status in enumerate(last_layer_status):
        poly = next_layer_polytopes[idx]
        if status == polytope.PolyPrev.UNKNOWN:
            return "(UNKNOWN)"
    return True



def test(ilb, iub, pulseidx, pulse_ilb, pulse_iub, cegarOptions, expect_output, timeoutFile):
    weightsObj = dataload.LOADER(modelpath, 10)
    #cegarOptions = cegar_option()

    pltp_mgr = polytope.polytopes(weightsObj, input_idx=0)
    init_polytope = get_init_polytope(weightsObj, ilb = ilb if pulseidx != 1 else pulse_ilb,
        iub = iub if pulseidx != 1 else pulse_iub )
    # FIXME: 
    # You need more than : ilb, iub things
    # init is this , but the second should be after it
    # propagate polytopes in the first several layers (and after pulse, use ...)

    check_result = \
        poly_prop_from_layer(timeoutFile, pltp_mgr, 1, init_polys = [init_polytope], stimulus_total=50,\
        response_total=50, ilb=ilb, iub=iub, pulseidx = pulseidx, pulse_ilb = pulse_ilb, pulse_iub = pulse_iub , expect_output=expect_output,\
        options=cegarOptions, cegar_stack=[1])
    
    print ('Checking results:', check_result)
    # return check results, # polytopes, 
    return check_result, 


def run_all_tests(tests,  pulseidxs, pulse_abs_range ,  timeout, timeoutFile, cegarOptions , fileoption = 'w'):
    global maxlayers_gone
    global maxNeural
    
    with open('pulse_test.log', fileoption) as fp:
        for ilb, iub in tests:
            for pulse_idx in pulseidxs:
                pulse_ilb = np.min(np.abs(np.array(pulse_abs_range)) * (-1 if ilb > 0 else 1))
                pulse_iub = np.max(np.abs(np.array(pulse_abs_range)) * (-1 if ilb > 0 else 1))
                maxNeural = 0
                maxlayers_gone = 0
                with open(timeoutFile,'w') as fout:
                    fout.write('continue')
                t0 = time.time()
                try:
                    with polytope.Timeout(timeout, timeoutFile):
                        result = test(ilb,iub, pulse_idx, pulse_ilb, pulse_iub, cegarOptions, 1 if ilb > 0 else -1,  timeoutFile = timeoutFile)
                except KeyboardInterrupt:
                    print ("SKIP")
                    result = "(SKIPPED)"
                except polytope.FileTimeOut:
                    print ('TIMEOUT:', timeout)
                    result = "(TIMEOUT)"
                t1 = time.time()
                print ((ilb, iub),'pulse@idx:', pulse_idx,(pulse_ilb,pulse_iub) ,' ---> ', end = '', file=fp)
                print (result, end = '', file=fp)
                print ('N=',maxNeural, 'L=',maxlayers_gone, end = '  T:', file=fp)
                print (t1-t0, file=fp)
                fp.flush()
    # collect run time

if __name__ == "__main__":
    cegarOptions = cegar_option()
    cegarOptions.NumPolyToStartAbs = 300
    run_all_tests(tests = testranges,cegarOptions=cegarOptions, timeout=100*60, timeoutFile='pulse_test.lock',fileoption='w', \
        pulseidxs = pulseindices, pulse_abs_range = pulseabsrange)
    
    
    
    
    
