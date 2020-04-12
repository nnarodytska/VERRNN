import dataload
import polytope
import copy

# okay we don't import again
np = polytope.np
List = polytope.List
Sequence = polytope.Sequence

# -----------------------------------------------
# configurations
modelpath = "../deep_monkey/results/N7_L1_r11/"


# -----------------------------------------------
# Options
class cegar_option(polytope.DefaultOptions):
    def __init__(self):
        super().__init__()
        self.NumPolyToStartAbs = 300
        self.StartFixedpointIter = 4
        self.PercentageOfFixedpointToContinue = 0.05
        self.AbsoluteFixedpointToContinue = 10
        self.FixedpointIternDelta1 = 1
        self.FixedpointIternDelta2 = 10
        self.RangeEstStart = 45
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

def poly_prop_from_layer(pltp_mgr : polytope.polytopes, start_layer : int, init_polys : List[polytope.convex_polytope] , \
    stimulus_total : int, response_total : int, ilb : float, iub : float, expect_output: int, options : cegar_option, \
    cegar_stack : List[int]) -> bool :
    """ expect_output: +1 -1, returns True if holds, otherwise False """

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
        per_layer_option.Invtest = layer_idx >= fixedpoint_layer
        per_layer_option.PropPreCheck = layer_idx >= options.RangeEstStart

        next_layer_polytopes, prev_polyResult, misc = \
            polytope.PolytopePropagateByLayer('fp_test.lock',pltp_mgr, layer_idx, stimulus_total, response_total, \
            prev_layer_polytopes = prev_layer_polys, ilb = ilb, iub = iub, options=per_layer_option)
        
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
        print ('\n---- pos / neg / unknown / ? = %d / %d / %d / %d' % (npos, nneg, nunknown, nnone))
        if (expect_output > 0 and nneg > 0) or (expect_output < 0 and npos > 0):
            print ('CEX found!'); return False
        if len(next_layer_polytopes) == 0:
            print ('Procedure complete!'); return True
        
        # ------------- ADD INPUT RANGE -------------
        for poly in next_layer_polytopes:
            poly.add_vi_range(weightsObj.num_state, ilb, iub) # add input range
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
    _ , last_layer_status, _ = polytope.PolytopePropagateByLayer('fp_test.lock',pltp_mgr, layer_idx, stimulus_total, response_total, \
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
    return True



def test():
    weightsObj = dataload.LOADER(modelpath, 10)
    ilb = 0.3
    iub = 0.7
    expect_output = 1
    cegarOptions = cegar_option()

    pltp_mgr = polytope.polytopes(weightsObj, input_idx=0)
    init_polytope = get_init_polytope(weightsObj, ilb = ilb, iub = iub )
    poly_prop_from_layer(pltp_mgr, 1, init_polys = [init_polytope], stimulus_total=50,\
        response_total=50, ilb=ilb, iub=iub,expect_output=expect_output,\
        options=cegarOptions, cegar_stack=[1])


if __name__ == "__main__":
    test()
