from .ilp_polytope import polytopes, convex_polytope, np, cplex


class ConvexRangeEstimateResponse(object):
    def __init__(self, dataholder, verbose = False):
        self.num_state = dataholder.num_state
        self.num_input = dataholder.num_input
        self.dataholder = dataholder
        self.pltp_mgr = polytopes(dataholder, input_idx=1)
    
    def propagate(self, total_response_step, initpoly, options): # you don't need ilb/iub either?
        # by default we use the last response , ilb = 1.0, iub = 1.0
        prev_layer_poly = initpoly
        for idx in range(total_response_step):
            _, vertices = self.pltp_mgr.propagate_split_polytope(prev_layer_poly, 1.0, 1.0, \
                FacetApproxBound=options.FacetApproxBound,ReduceDupFacet=options.ReduceDupFacet,\
                RdFacetBound=options.RdFacetBound,toPointsOnly=True, qhullTimeout=options.qhullTimeout)
            
            try:
                approx, _ = self.pltp_mgr.merge_polytopes_by_convex_hull(vertices, \
                    FacetApproxBound=options.FacetApproxBound,ReduceDupFacet=options.ReduceDupFacet,RdFacetBound=options.RdFacetBound)
                vertices = []
                prev_layer_poly = approx
            except RuntimeError:
                vertices = [np.vstack(vertices)]
            
            if len(vertices) != 0:
                assert (len(vertices) == 1)
                bvs = vertices[0]
                approx_loosen = self.pltp_mgr.loosen_poly_based_on_vertices(prev_layer_poly, bvs, include_old_points = False)
                prev_layer_poly = approx_loosen
            prev_layer_poly.add_vi_range(self.num_state, 1.0, 1.0 )
        return prev_layer_poly
    
    def get_last_1_output_ub_lb(self, ply):
        # here you need to run cplex 
        # TODO: 
        pass
            

