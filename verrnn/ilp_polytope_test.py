import dataload
import polytope
import smtbmc
import numpy as np

# -----------------------------------------------
# configurations

modelpath = "../deep_monkey/results/N7_L1_r11/"

# a known data
class dh(object):
    def __init__(self):
        self.num_input = 1
        self.num_state = 2
        self.W_rec = np.array([[0.1,0.2],[0.3,-0.4]]).T
        self.W_in = np.array([[0.2],[-0.1]]).T
        self.b_rec = (np.array([[-0.2],[0.5]]).T)[0]
        self.init_state = np.array([0.1,1.2])

# a known data
class dh2(object):
    def __init__(self):
        self.num_input = 1
        self.num_state = 3
        self.W_rec = np.array([[0.1,0.2,-0.1],[0.3,-0.4,0.1],[-0.2,0.1,0.3]]).T
        self.W_in = np.array([[0.2],[-0.1],[0.1]]).T
        self.b_rec = (np.array([[-0.2],[0.5],[0.1]]).T)[0]
        self.init_state = np.array([0.1,1.2,0.0])
    
# -----------------------------------------------
# test function
def test1():

    dataholder = dh()

    vars = ['s%d' % s for s in range(dataholder.num_state)] + ['x%d' % x for x in range(dataholder.num_input)]

    initial_polytope = polytope.convex_polytope(vars)
    for idx, v in enumerate(dataholder.init_state):
        initial_polytope.add_vi_range(idx,v, v)
    initial_polytope.add_vi_range(dataholder.num_state, -1.0, 1.0) # input range

    #initial_polytope.add_vi_range(0, -1.0, 1.0)
    #initial_polytope.add_vi_range(1, -1.0, 1.0)
    #initial_polytope.add_vi_range(2, -1.0, 1.0)


    print (initial_polytope.get_vertices())

    pltp_mgr = polytope.polytopes(dataholder)
    new_polys, _ = pltp_mgr.propagate_split_polytope(initial_polytope,-1.0,1.0, 0,False,0)
    for p in new_polys:
        print ('-------------------------')
        p.add_vi_range(dataholder.num_state, -1.0, 1.0) # input range
        print ('ineqs:')
        p.print_eqs()
        print ('points:')
        print (p.get_vertices())


# -----------------------------------------------
# main function
def test2():
    
    weightsObj = dh2()

    ilb = -1.0
    iub = 1.0

    vars = ['s%d' % s for s in range(weightsObj.num_state)] + ['x%d' % x for x in range(1)]

    initial_polytope = polytope.convex_polytope(vars)
    for idx, v in enumerate(weightsObj.init_state):
        v = 0.0 if v <= 0 else v
        initial_polytope.add_vi_range(idx,v, v)
    initial_polytope.add_vi_range(weightsObj.num_state, ilb, iub) # input range

    layer_polytope_map = [[initial_polytope]] # int->list of polytopes

    #print (initial_polytope.get_vertices())

    pltp_mgr = polytope.polytopes(weightsObj)

    Nlayer = 10
    for idx in range(Nlayer):
        prev_layer_polytopes = layer_polytope_map[idx]
        next_layer_polytopes = []
        for p in prev_layer_polytopes:
            new_polys, _ = pltp_mgr.propagate_split_polytope(p,ilb, iub,0,False,0)
            # add input ranges
            for newpoly in new_polys:
                newpoly.add_vi_range(weightsObj.num_state, ilb, iub) # input ranges
            next_layer_polytopes += new_polys
        layer_polytope_map.append(next_layer_polytopes)
        print ('layer %d : # polytopes = %d' % (idx+1, len(next_layer_polytopes) ) )
    


# -----------------------------------------------
# test function
def range_popagate():
    weightsObj = dataload.LOADER(modelpath, 10)

    ilb = 0.5
    iub = 2.0

    vars = ['s%d' % s for s in range(weightsObj.num_state)] + ['x%d' % x for x in range(1)]

    initial_polytope = polytope.convex_polytope(vars)
    for idx, v in enumerate(weightsObj.init_state):
        v = 0.0 if v <= 0 else v
        initial_polytope.add_vi_range(idx,v, v)
    initial_polytope.add_vi_range(weightsObj.num_state, ilb, iub) # input range

    layer_polytope_map = [[initial_polytope]] # int->list of polytopes

    #print (initial_polytope.get_vertices())

    pltp_mgr = polytope.polytopes(weightsObj)

    Nlayer = 10
    for idx in range(Nlayer):
        prev_layer_polytopes = layer_polytope_map[idx]
        next_layer_polytopes = []
        for p in prev_layer_polytopes:
            new_polys, _ = pltp_mgr.propagate_split_polytope(p,ilb, iub,0,False,0)
            # okay, let's test it here:
            new_polys_left = []
            for newpoly in new_polys:
                slb,sub = newpoly.get_range_box()
                assert (slb.shape[0] == weightsObj.num_state)
                if idx == 3:
                    range_estimator = smtbmc.RangeEstimation(weightsObj)
                    #range_estimator.populate_polytope_state_range(50-Nlayer,slb, sub, stimulus_lower = ilb, stimulus_upper = iub, response_step = 30)
                    range_estimator.populate_polytope_state_range(10,slb, sub, stimulus_lower = ilb, stimulus_upper = iub, response_step = 30)
                    stable, olb, oub = range_estimator.get_previous_populate_output_status()
                    if stable:
                        print ('stable: ', olb, oub)
                    else:
                        print ('instable: ', olb, oub)
                        new_polys_left.append(newpoly)
                else:
                    new_polys_left.append(newpoly)
            # add input ranges
            for newpoly in new_polys_left:
                newpoly.add_vi_range(weightsObj.num_state, ilb, iub) # input ranges
            next_layer_polytopes += new_polys_left
        layer_polytope_map.append(next_layer_polytopes)
        print ('Start on layer %d : # polytopes = %d' % (idx+1, len(next_layer_polytopes) ) )



# -----------------------------------------------
# test function

def merge_polytope():
    weightsObj = dataload.LOADER(modelpath, 10)

    ilb = 0.3
    iub = 0.7

    vars = ['s%d' % s for s in range(weightsObj.num_state)] + ['x%d' % x for x in range(1)]

    initial_polytope = polytope.convex_polytope(vars)
    for idx, v in enumerate(weightsObj.init_state):
        v = 0.0 if v <= 0 else v
        initial_polytope.add_vi_range(idx,v, v)
    initial_polytope.add_vi_range(weightsObj.num_state, ilb, iub) # input range

    layer_polytope_map = [[initial_polytope]] # int->list of polytopes

    #print (initial_polytope.get_vertices())

    pltp_mgr = polytope.polytopes(weightsObj)

    Nlayer = 30
    for idx in range(Nlayer):
        prev_layer_polytopes = layer_polytope_map[idx]
        next_layer_polytopes = []
        num_fixedpoint = 0
        for p in prev_layer_polytopes:
            if p.is_null():
                continue
            oldlb, oldub = p.get_range_box()
            oldlb = oldlb[:-1]
            oldub = oldub[:-1]

            new_polys, _ = pltp_mgr.propagate_split_polytope(p,ilb, iub,0,False,0)
            if idx >= 4 and len(new_polys) != 0: # here we start to approximate:
                approx, _ = pltp_mgr.merge_polytopes_by_convex_hull(new_polys, vars,0,False,0)
                vertices = approx.get_vertices()
                
                approx.original_pts = vertices

                new_polys = [approx]
            if idx >= 5 and len(new_polys) != 0: # here we start to check for inductiveness
                # check for inclusion : 
                newlb, newub = new_polys[0].get_range_box()
                if (newub < oldub + 0.0001).all() and (newlb > oldlb - 0.0001).all():
                    print ('!!! --->  possibly fixedpoint')
                    if (polytope.check_safety(new_polys[0], weightsObj, ilb, iub, +1)):
                        print ('@#@#@ safe!')
                        if ( polytope.check_inductiveness_qhull( p.original_pts , new_polys[0].original_pts ) ):
                            print ('******* confirmed fixedpoint')
                            new_polys = []
                            num_fixedpoint += 1
                        else:
                            print ('------- not fixedpoint')
                else:
                    pass
                    #print ('not fixedpoint ,' , end = '' )
                    #if not (newub < oldub + 0.0001).all():
                    #    print (' ub ', oldub, ' --> ', newub, end = '')
                    #if not (newlb > oldlb - 0.0001).all():
                    #    print (' lb ', oldlb, ' --> ', newlb, end = '')
                    #print () # make a new line
            # add input ranges
            for newpoly in new_polys:
                old_pts = newpoly.get_vertices()
                
                newpoly.original_pts = old_pts # our own buffer of points

                newpoly.add_vi_range(weightsObj.num_state, ilb, iub) # input ranges
                try : 
                    newpoly.is_null()
                except :
                    print ('constructed from points : ' )
                    print (old_pts)
                    np.savez('original_pts', old_pts)
                    exit(1)

            next_layer_polytopes += new_polys
        layer_polytope_map.append(next_layer_polytopes)
        print ('Current layer fp:', num_fixedpoint, '/', len(layer_polytope_map[idx]))
        print ('Start on layer %d : # polytopes = %d' % (idx+1, len(next_layer_polytopes) ) )
    



# -----------------------------------------------
# test function

def propagate_polytope():
    weightsObj = dataload.LOADER(modelpath, 10)

    #ilb = 0.3
    #iub = 0.7
    #Approximate_layer = 4
    ilb = -0.7
    iub = -0.3
    Approximate_layer = 20

    vars = ['s%d' % s for s in range(weightsObj.num_state)] + ['x%d' % x for x in range(1)]

    initial_polytope = polytope.convex_polytope(vars)
    for idx, v in enumerate(weightsObj.init_state):
        v = 0.0 if v <= 0 else v
        initial_polytope.add_vi_range(idx,v, v)
    initial_polytope.add_vi_range(weightsObj.num_state, ilb, iub) # input range

    layer_polytope_map = [[initial_polytope]] # int->list of polytopes

    #print (initial_polytope.get_vertices())

    pltp_mgr = polytope.polytopes(weightsObj)
    
    npos = 0
    nneg = 0
    nconflict = 0
    nunknown = 0

    Nlayer = 48
    for idx in range(Nlayer):
        prev_layer_polytopes = layer_polytope_map[idx]
        next_layer_polytopes = []
        num_fixedpoint = 0
        for p in prev_layer_polytopes:
            if p.is_null():
                continue
            oldlb, oldub = p.get_range_box()
            oldlb = oldlb[:-1]
            oldub = oldub[:-1]

            new_polys, _ = pltp_mgr.propagate_split_polytope(p,ilb, iub,0,False,0)
            if idx >= Approximate_layer and len(new_polys) != 0: # here we start to approximate (Join):
                approx, _ = pltp_mgr.merge_polytopes_by_convex_hull(new_polys, vars)
                #print ( '---- Coef size: ', np.array(approx.coefs).shape)
                #vertices = approx.get_vertices()
                #print ('#v =', len(vertices))
                #print ('#coef = ',len(approx.coefs))
                #approx.original_pts = vertices
                new_polys = [approx]
            
            if idx == 47 and len(new_polys) != 0:
                res = polytope.check_safety(new_polys[0], weightsObj, ilb, iub, +1)
                print ( '----------------' if not res else '' , res)
                res_neg = polytope.check_safety(new_polys[0], weightsObj, ilb, iub, -1)
                print ( '>>>>>>> -1 <<<<<' if res_neg else '' )
                
                if (res and not res_neg):
                    npos += 1
                elif (res_neg and not res):
                    nneg += 1
                elif (res and res_neg):
                    nconflict += 1
                else:
                    nunknown += 1
                

            # add input ranges
            for newpoly in new_polys:
                old_pts = newpoly.get_vertices()
                
                newpoly.original_pts = old_pts # our own buffer of points
                print ('#v =', len(old_pts))
                print ('#coef = ',len(newpoly.coefs))

                newpoly.add_vi_range(weightsObj.num_state, ilb, iub) # input ranges
                try : 
                    newpoly.is_null()
                except :
                    print ('constructed from points : ' )
                    print (old_pts)
                    np.savez('original_pts', old_pts)
                    exit(1)

            next_layer_polytopes += new_polys
        layer_polytope_map.append(next_layer_polytopes)
        print ('Current layer fp:', num_fixedpoint, '/', len(layer_polytope_map[idx]))
        print ('Start on layer %d : # polytopes = %d' % (idx+1, len(next_layer_polytopes) ) )

    print('pos/neg/conflict/unknown = ', npos, nneg, nconflict, nunknown)
    #all_approx_poly = layer_polytope_map[-1]
    #for poly in all_approx_poly:
    #    res = polytope.check_safety(poly, weightsObj, ilb, iub, +1)
    #    print (res)




def relaxing_polytope():
    weightsObj = dataload.LOADER(modelpath, 10)

    #ilb = 0.3
    #iub = 0.7
    #Approximate_layer = 4
    ilb = -0.7
    iub = -0.3
    Approximate_layer = 1000 # a large number that you should not reach unless changed if too many polytopes
    ConvexHullApproxThreshold = 200 # if > 200 polytope turn on convex hull

    vars = ['s%d' % s for s in range(weightsObj.num_state)] + ['x%d' % x for x in range(1)]

    initial_polytope = polytope.convex_polytope(vars)
    for idx, v in enumerate(weightsObj.init_state):
        v = 0.0 if v <= 0 else v
        initial_polytope.add_vi_range(idx,v, v)
    initial_polytope.add_vi_range(weightsObj.num_state, ilb, iub) # input range

    layer_polytope_map = [[initial_polytope]] # int->list of polytopes

    #print (initial_polytope.get_vertices())

    pltp_mgr = polytope.polytopes(weightsObj)
    
    npos = 0
    nneg = 0
    nconflict = 0
    nunknown = 0

    Nlayer = 48
    for idx in range(Nlayer):
        prev_layer_polytopes = layer_polytope_map[idx]
        next_layer_polytopes = []
        num_fixedpoint = 0
        for p in prev_layer_polytopes:
            if p.is_null():
                continue
            oldlb, oldub = p.get_range_box()
            oldlb = oldlb[:-1]
            oldub = oldub[:-1]

            
            new_polys, new_vertices = pltp_mgr.propagate_split_polytope(p,ilb, iub,0,False,0)

            if (len(new_vertices) > 0):
                # cannot use ConvexHull to create polytope
                # you can use loosen technique for it
                print ('failed to construct : ', len(new_vertices))
                if len(new_vertices) == 1:
                    new_vertices = new_vertices[0]
                else:
                    new_vertices = np.vstack(new_vertices)
                new_poly = pltp_mgr.loosen_poly_based_on_vertices(p, new_vertices)
                new_polys += [new_poly]
                print ('relaxing constraints:')
                print ('#coef = ', len(p.coefs) , ' --> ',len(new_poly.coefs))


            if idx >= Approximate_layer and len(new_polys) > 1: # here we start to approximate (Join):
                approx, _ = pltp_mgr.merge_polytopes_by_convex_hull(new_polys, vars)
                #print ( '---- Coef size: ', np.array(approx.coefs).shape)
                #vertices = approx.get_vertices()
                #print ('#v =', len(vertices))
                #print ('#coef = ',len(approx.coefs))
                #approx.original_pts = vertices
                new_polys = [approx]
            
            if idx == 47 and len(new_polys) != 0:
                res = polytope.check_safety(new_polys[0], weightsObj, ilb, iub, +1)
                print ( '----------------' if not res else '' , res)
                res_neg = polytope.check_safety(new_polys[0], weightsObj, ilb, iub, -1)
                print ( '>>>>>>> -1 <<<<<' if res_neg else '' )
                
                if (res and not res_neg):
                    npos += 1
                elif (res_neg and not res):
                    nneg += 1
                elif (res and res_neg):
                    nconflict += 1
                else:
                    nunknown += 1
                

            # add input ranges
            for newpoly in new_polys:
                old_pts = newpoly.get_vertices()
                
                newpoly.original_pts = old_pts # our own buffer of points
                print ('#v =', len(old_pts))
                print ('#coef = ',len(newpoly.coefs))

                newpoly.add_vi_range(weightsObj.num_state, ilb, iub) # input ranges
                try : 
                    newpoly.is_null()
                except :
                    print ('constructed from points : ' )
                    print (old_pts)
                    np.savez('original_pts', old_pts)
                    exit(1)

            next_layer_polytopes += new_polys
        layer_polytope_map.append(next_layer_polytopes)
        print ('Current layer fp:', num_fixedpoint, '/', len(layer_polytope_map[idx]))
        print ('Start on layer %d : # polytopes = %d' % (idx+1, len(next_layer_polytopes) ) )
        if (len(next_layer_polytopes) > ConvexHullApproxThreshold):
            print ('# Polytope > %d  ==>  turn on convex hull approximation' % ConvexHullApproxThreshold)
            Approximate_layer = idx+1

    print('pos/neg/conflict/unknown = ', npos, nneg, nconflict, nunknown)
    #all_approx_poly = layer_polytope_map[-1]
    #for poly in all_approx_poly:
    #    res = polytope.check_safety(poly, weightsObj, ilb, iub, +1)
    #    print (res)


# TODO: Better bound propogate
# TODO: volume and inductiveness...

def relaxing_compute_fixedpoint():
    weightsObj = dataload.LOADER(modelpath, 10)

    ilb = 0.3
    iub = 0.7
    expect_output = 1
    #Approximate_layer = 4
    #expect_output = -1
    #ilb = -2.0
    #iub = -0.3
    Approximate_layer = 1000 # a large number that you should not reach unless changed if too many polytopes
    Fixedpoint_compute_layer = 4
    ConvexHullApproxThreshold = 1e6 # if > 200 polytope turn on convex hull

    vars = ['s%d' % s for s in range(weightsObj.num_state)] + ['x%d' % x for x in range(1)]

    initial_polytope = polytope.convex_polytope(vars)
    for idx, v in enumerate(weightsObj.init_state):
        v = 0.0 if v <= 0 else v
        initial_polytope.add_vi_range(idx,v, v)
    initial_polytope.add_vi_range(weightsObj.num_state, ilb, iub) # input range

    layer_polytope_map = [[initial_polytope]] # int->list of polytopes

    #print (initial_polytope.get_vertices())

    pltp_mgr = polytope.polytopes(weightsObj)
    
    npos = 0
    nneg = 0
    nconflict = 0
    nunknown = 0

    Nlayer = 50
    NstartBoundProp = 47
    for idx in range(Nlayer):
        prev_layer_polytopes = layer_polytope_map[idx]
        next_layer_polytopes = []
        num_fixedpoint = 0
        num_safe = 0
        num_unsafe = 0
        for p in prev_layer_polytopes:
            if p.is_null():
                continue
            oldlb, oldub = p.get_range_box()
            oldlb = oldlb[:-1]
            oldub = oldub[:-1]

            
            new_polys, new_bad_vertices = pltp_mgr.propagate_split_polytope(p,ilb, iub,0,False,0)

            if (len(new_bad_vertices) > 0):
                # cannot use ConvexHull to create polytope
                # you can use loosen technique for it
                print ('failed to construct : ', len(new_bad_vertices))
                if len(new_bad_vertices) == 1:
                    new_bad_vertices = new_bad_vertices[0]
                else:
                    new_bad_vertices = np.vstack(new_bad_vertices)
                new_poly = pltp_mgr.loosen_poly_based_on_vertices(p, new_bad_vertices, include_old_points=False)
                new_poly.attribute.abs_poly_layer = idx
                new_poly.attribute.abs_poly_ref = p
                
                new_polys += [new_poly]
                print ('relaxing constraints:')
                print ('#coef = ', len(p.coefs) , ' --> ',len(new_poly.coefs))

            if idx >= Fixedpoint_compute_layer and len(new_polys) >= 1:
                collection_of_points = []
                for poly in new_polys:
                    collection_of_points.append(np.array(poly.get_vertices()))
                
                collection_of_points = np.vstack(collection_of_points)
                
                not_safe_flag = False
                relaxed_poly = p
                print ('Relaxing to inductive fp', end = '', flush=True)
                num_of_loosen_layer = idx
                while not relaxed_poly.check_points_in(collection_of_points, num_input=1, epsilon = 1e-6) and \
                    num_of_loosen_layer <= Nlayer: # if we loosen enough time, we are still good, then we are fine
                    relaxed_poly = pltp_mgr.loosen_poly_based_on_vertices(relaxed_poly, collection_of_points, include_old_points=True)
                    num_of_loosen_layer += 1
                    print ('.', end = '', flush=True)
                    if polytope.check_safety(relaxed_poly, weightsObj, ilb, iub, -expect_output, nsteps=0):
                        print ('CEX!')
                        exit (1)
                    if not polytope.check_safety(relaxed_poly, weightsObj, ilb, iub, expect_output, nsteps=0):
                        not_safe_flag = True
                        break
                    relaxed_poly.add_vi_range(weightsObj.num_state, ilb, iub) # input ranges
                    collection_of_points = []
                    nps_relaxed, nbv = pltp_mgr.propagate_split_polytope(relaxed_poly,ilb, iub,0,False,0,toPointsOnly=True)
                    for poly in nps_relaxed:
                        collection_of_points.append(np.array(poly.get_vertices()))
                    if len(nbv) == 0:
                        break # no points
                    for v in nbv:
                        collection_of_points.append(v)
                    collection_of_points = np.vstack(collection_of_points)
                        # map to 0 points

                if not not_safe_flag:
                    # we get a safe invariant:
                    new_polys = [] # we can avoid considering this case
                    num_fixedpoint += 1
                    print ('safe')
                    num_safe += 1
                else:
                    print ('not safe')
                    num_unsafe += 1

            if idx >= Approximate_layer and len(new_polys) > 1: # here we start to approximate (Join):
                approx, _ = pltp_mgr.merge_polytopes_by_convex_hull(new_polys, vars)
                approx.attribute.abs_poly_layer = idx
                approx.attribute.abs_poly_ref = p
                #print ( '---- Coef size: ', np.array(approx.coefs).shape)
                #vertices = approx.get_vertices()
                #print ('#v =', len(vertices))
                #print ('#coef = ',len(approx.coefs))
                #approx.original_pts = vertices
                new_polys = [approx]
            
            if idx >= NstartBoundProp and len(new_polys) != 0:
                new_poly_after_direct_bound_prop = []
                for new_poly in new_polys:
                    res = polytope.check_safety(new_poly, weightsObj, ilb, iub, +1, nsteps=Nlayer-idx)
                    print ( '----------------' if not res else '' , res)
                    res_neg = polytope.check_safety(new_polys[0], weightsObj, ilb, iub, -1, nsteps=Nlayer-idx)
                    print ( '>>>>>>> -1 <<<<<' if res_neg else '' )
                    
                    if (res and not res_neg):
                        npos += 1
                    elif (res_neg and not res):
                        nneg += 1
                    elif (res and res_neg):
                        nconflict += 1
                    else:
                        nunknown += 1
                        new_poly_after_direct_bound_prop.append(new_poly)
                

            # add input ranges
            for newpoly in new_polys:
                old_pts = newpoly.get_vertices()
                
                newpoly.original_pts = old_pts # our own buffer of points
                #print ('#v =', len(old_pts))
                #print ('#coef = ',len(newpoly.coefs))

                newpoly.add_vi_range(weightsObj.num_state, ilb, iub) # input ranges
                try : 
                    newpoly.is_null()
                except :
                    print ('constructed from points : ' )
                    print (old_pts)
                    np.savez('original_pts', old_pts)
                    exit(1)

            next_layer_polytopes += new_polys
        layer_polytope_map.append(next_layer_polytopes)
        print ('Current layer fp:', num_fixedpoint, '/', len(layer_polytope_map[idx]))
        print ('Safe : Unsafe = %d : %d' % (num_safe, num_unsafe))
        print ('Start on layer %d : # polytopes = %d' % (idx+1, len(next_layer_polytopes) ) )
        if (len(next_layer_polytopes) > ConvexHullApproxThreshold):
            print ('# Polytope > %d  ==>  turn on convex hull approximation' % ConvexHullApproxThreshold)
            Approximate_layer = idx+1
        else:
            Approximate_layer = 1000 # turn it off
        if (len(next_layer_polytopes) == 0):
            print ('complete!')
            break

    # CEGAR HERE:
    

    # print('pos/neg/conflict/unknown = ', npos, nneg, nconflict, nunknown)
    #all_approx_poly = layer_polytope_map[-1]
    #for poly in all_approx_poly:
    #    res = polytope.check_safety(poly, weightsObj, ilb, iub, +1)
    #    print (res)


if __name__ == "__main__":
    #merge_polytope() #propagate_polytope()
    relaxing_compute_fixedpoint()
