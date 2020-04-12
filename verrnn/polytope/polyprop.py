from .ilp_polytope import polytopes, convex_polytope, np
import smtbmc
from enum import IntEnum
from typing import List, Dict, Tuple, Sequence, Callable
import time



class DefaultOptions(object):
  def __init__(self):
    self.PropPreCheck = False # whether to do a pre-bound-propagation
    self.PropPreMethod = 'BP' # 'Triangle/CvxHull/StarSet', what method to use for precheck
    self.DoProp = True        # whether to do propagation
    self.DoJoin = True        # whether to do a join
    self.FacetApproxBound = 2e3 # if the number of facet is this much (after reduce, if reduce enabled)
    self.ReduceDupFacet = True # whether to reduce the H-resp
    self.RdFacetBound = 1e4 # if the number of facet is > than this, will not reduce H-resp
    self.Invtest = True # Test and found invariant
    self.externQhull = False # whetehr to use external qhull to compute
    self.qhullTimeout = 30 # if it is longer than this, we will give up and loosen instead

  def check(self):
    if not self.PropPreCheck and not self.DoProp:
      print ('Will do nothing!')


class FileTimeOut(Exception):
  pass


class PolyPrev(IntEnum):
  POS = 1
  UNKNOWN = 0
  NEG = -1
  NumericalNone = 2

class PolyPreStatus(object):
  def __init__(self, prev, next_num):
    self.prev : PolyPrev = prev
    self.next_num : int = next_num

class PerLayerInfo(object):
  def __init__(self):
    self.num_fixedpoint = 0

def PropagateCheckSafe(full_data_holder, ilb : float, iub : float,  stimulus_step : int, response_step : int, pltp : convex_polytope, method : str):
  """returns (stable, result_sign)"""
  if method == 'BP':
    if pltp.no_input_vs is None and pltp.is_null():
      return True, PolyPrev.NumericalNone
    slb,sub = pltp.get_range_box()
    if slb.shape[0] > full_data_holder.num_state:
      slb = slb[0:full_data_holder.num_state]
    if sub.shape[0] > full_data_holder.num_state:
      sub = sub[0:full_data_holder.num_state]
    range_estimator = smtbmc.RangeEstimation(full_data_holder)
    range_estimator.populate_polytope_state_range(stimulus_step, state_ub_list = sub,state_lb_list = slb , stimulus_lower = ilb, stimulus_upper = iub, response_step = response_step)
    oub , olb = range_estimator.get_previous_populate_output_status(last_n = 10)
    oub = np.array(oub);olb = np.array(olb)
    if (olb >= 0).all():
      return True, PolyPrev.POS
    if (oub <= 0).all():
      return True, PolyPrev.NEG
    return False, PolyPrev.UNKNOWN
  elif method == 'Triangle':
    assert False # Not implemented
  elif method == 'CvxHull':
    assert False # Not implemented
  elif method == 'StarSet':
    assert False # Not implemented
  assert False # Not implemented

def _collect_points(polytopes : Sequence[convex_polytope] , points : Sequence[np.ndarray]) -> np.ndarray:
  collection_of_points = []
  for poly in polytopes:
    collection_of_points.append(np.array(poly.get_vertices()))
  collection_of_points += points
  if len(collection_of_points) == 0:
    assert (len(polytopes) == 0)
    raise RuntimeError('no points available!')
  collection_of_points = np.vstack(collection_of_points)
  return collection_of_points

def TryRelaxToInvarint(pltp_mgr:polytopes, prev_poly : convex_polytope, collection_of_points, no_loosen:bool):
  if isinstance(collection_of_points, list):
    collection_of_points = np.array(collection_of_points)
  print (collection_of_points.shape)
  #if collection_of_points.shape[0] > 50000:
  no_loosen = True

  if prev_poly.no_input_vs is not None:
    old_vl = prev_poly.no_input_vs
  else:
    old_vl = prev_poly.get_vertices()
  old_v = np.array(old_vl)
  if len(old_vl) > 0 and old_v.shape[1] > pltp_mgr.num_state:
    old_v = old_v[:,:pltp_mgr.num_state]
  if collection_of_points.shape[1] > pltp_mgr.num_state:
    collection_of_points = collection_of_points[:,:pltp_mgr.num_state]

  if len(old_vl) == 0:
    vs = collection_of_points
  else:
    vs = np.vstack([old_v,collection_of_points])
  # here we create convex hull of them
  try:
    if no_loosen: # once we are out of convex, we will keep loosen instead of convex
      raise Exception
    new_pltp = convex_polytope(pltp_mgr.vars)
    new_pltp.from_vertices(vs, pltp_mgr.num_input, FacetApproxBound=1000, ReduceDupFacet=True, RdFacetBound=1000, externQhull = True, timeout=10)
  except:
    # construct by loosening
    new_pltp = pltp_mgr.loosen_poly_based_on_vertices(prev_poly, collection_of_points, include_old_points=True)
    no_loosen = True

  return new_pltp, no_loosen

def ConstructInvariant(pltp_mgr: polytopes , ilb:float, iub:float, layer_idx : int, stimulus_total : int, response_total : int, initial_polytope : convex_polytope, \
    initial_next_level_points : np.ndarray, SafeCheckMethod : str , qhullTimeout:int ) -> PolyPrev :
  
  relaxed_poly = initial_polytope
  collection_of_points = initial_next_level_points

  loosen_time = 0.0
  pp_time = 0.0
  prop_split = 0.0
  
  print ('Relaxing to inductive fp', end = '', flush=True)
  num_of_loosen_layer = layer_idx
  no_loosen = False
  while not relaxed_poly.check_points_in(collection_of_points, num_input=1, epsilon = 1e-6) and \
      num_of_loosen_layer <= stimulus_total: # if we loosen enough time, we are still good, then we are fine
    t0 = time.time()
    #relaxed_poly = pltp_mgr.loosen_poly_based_on_vertices(relaxed_poly, collection_of_points, include_old_points=True)
    relaxed_poly, no_loosen = TryRelaxToInvarint(pltp_mgr,relaxed_poly,collection_of_points , no_loosen)
    t1 = time.time(); loosen_time += t1-t0
    
    print ('*' if no_loosen else '.' , end = '', flush=True)
    
    t0 = time.time()
    response_step = response_total if num_of_loosen_layer <= stimulus_total else response_total - (num_of_loosen_layer-stimulus_total)
    stable , res = PropagateCheckSafe(full_data_holder=pltp_mgr.dataholder, ilb=ilb, iub=iub, \
        stimulus_step=0,response_step=response_step,pltp=relaxed_poly,method=SafeCheckMethod)
    t1 = time.time(); pp_time += t1-t0
    
    if res == PolyPrev.NumericalNone:
      print ('(N)')
      # start the debug here
      print ('Points should be inside! Check points in:', end = '')
      print (relaxed_poly.check_points_in(collection_of_points, num_input=1, epsilon = 1e-6))
      exit(1)

      print ('time:',loosen_time, pp_time, prop_split)
      return PolyPrev.NumericalNone

    num_of_loosen_layer += 1
    if not stable:
      print ('(UNKNOWN)')
      print ('time:',loosen_time, pp_time, prop_split)
      return PolyPrev.UNKNOWN

    t0 = time.time()
    relaxed_poly.add_vi_range(pltp_mgr.num_state, ilb, iub) # add input range
    nps_relaxed, nbv = pltp_mgr.propagate_split_polytope( \
      relaxed_poly,ilb, iub, 0, False, 0,externQhull=True, toPointsOnly=True,\
      qhullTimeout=qhullTimeout)
    t1 = time.time(); prop_split += t1-t0
    # "0, False, 0" actually does not matter
    try:
      collection_of_points = _collect_points(nps_relaxed, nbv)
    except RuntimeError:
      print ('(N)')
      print (len(nbv),len(nps_relaxed))
      assert (len(nbv) == 0)
      assert (len(nps_relaxed) == 0)
      print ('time:',loosen_time, pp_time, prop_split)
      return PolyPrev.NumericalNone
    print ('[%d]%d' %( len(nbv), collection_of_points.shape[0]),end='')

  #end of while loop
  _ , sign = PropagateCheckSafe(full_data_holder=pltp_mgr.dataholder,ilb=ilb, iub=iub, \
      stimulus_step=0,response_step=response_total,pltp=relaxed_poly,method=SafeCheckMethod)
  relaxed_poly.dump_to_file('inv_poly')
  print ('('+str(int(sign))+')')
  print ('time:',loosen_time, pp_time, prop_split)
  if int(sign) == 2:
    print ('Check point in :', end = '')
    print (relaxed_poly.check_points_in(collection_of_points, num_input=1,epsilon=1e-6))
  return sign
  
# after propagation by layer, you also need 

def PolytopePropagateByLayer(timeoutFile:str,pltp_mgr: polytopes , layer_idx : int, stimulus_total : int, response_total : int, \
    prev_layer_polytopes : Sequence[convex_polytope], ilb : float, iub : float, options : DefaultOptions) -> \
    Tuple[List[convex_polytope],List[PolyPreStatus], PerLayerInfo]:
  """returns (next_layer_polytopes, prev_polyResult),
  layer_idx is the layer id of the current layer!
  """
  with open(timeoutFile) as fin:
    wd = fin.read()
    if 'stopset' in wd:
      raise FileTimeOut
  
  prev_polyResult : List[PolyPreStatus] = [] # list of PolyPreStatus
  next_layer_polytopes : List[convex_polytope] = []
  num_fixedpoint = 0
  layer_stat = PerLayerInfo()

  for p in prev_layer_polytopes:
    next_layer_poly_for_current_poly : List[PolyPreStatus] = []
    if p.is_null():
      prev_polyResult.append(PolyPreStatus( PolyPrev.NumericalNone , 0))
      continue

    if options.PropPreCheck: # do pre-check
      stimulus_step = stimulus_total-layer_idx if stimulus_total-layer_idx >= 0 else 0
      response_step = response_total if stimulus_total-layer_idx >= 0 else response_step-(layer_idx-stimulus_total)
      stable, res = PropagateCheckSafe(full_data_holder=pltp_mgr.dataholder, \
          ilb=ilb, iub=iub, \
          stimulus_step = stimulus_step, response_step = response_step, \
          pltp = p, method = options.PropPreMethod )
      if stable:
        prev_polyResult.append(PolyPreStatus( res , -1))
        continue

    if options.DoProp:
      new_polys, new_bad_vertices = pltp_mgr.propagate_split_polytope(p,ilb, iub, \
        FacetApproxBound=options.FacetApproxBound,ReduceDupFacet=options.ReduceDupFacet,\
        RdFacetBound=options.RdFacetBound,externQhull=options.externQhull, toPointsOnly=options.DoJoin,
        qhullTimeout=options.qhullTimeout)
      n_poly = len(new_polys) + len(new_bad_vertices) # before approximate
      if n_poly == 0:
        prev_polyResult.append(PolyPreStatus( PolyPrev.NumericalNone , n_poly))
        continue

      if options.Invtest:
        # we will first test if we can get a good fixpoint from it.
        poly_status = ConstructInvariant(pltp_mgr=pltp_mgr, layer_idx=layer_idx, stimulus_total=stimulus_total, response_total=response_total, \
            initial_polytope=p,initial_next_level_points=_collect_points(new_polys,new_bad_vertices), SafeCheckMethod=options.PropPreMethod, \
            ilb = ilb, iub =iub, qhullTimeout=options.qhullTimeout )
        if poly_status == PolyPrev.POS or poly_status == PolyPrev.NEG:
          prev_polyResult.append(PolyPreStatus( poly_status , -1))
          num_fixedpoint += 1
          continue

      next_layer_poly_for_current_poly = new_polys # use this as current polys

      if options.DoJoin: # if we do join, we would expect no polys returned from ...
        new_bad_vertices = _collect_points(new_polys,new_bad_vertices)
        next_layer_poly_for_current_poly = []
        try:
          approx, _ = pltp_mgr.merge_polytopes_by_convex_hull(new_bad_vertices, \
            FacetApproxBound=options.FacetApproxBound,ReduceDupFacet=options.ReduceDupFacet,RdFacetBound=options.RdFacetBound, externQhull=options.externQhull,timeout=options.qhullTimeout)
          if p.attribute.is_abstract: # inherit the starting poly
            assert (p.attribute.abs_poly_ref is not None)
            approx.attribute.abs_poly_ref = p.attribute.abs_poly_ref
            approx.attribute.abs_poly_layer = p.attribute.abs_poly_layer
          else:
            approx.attribute.abs_poly_ref = p
            approx.attribute.abs_poly_layer = layer_idx - 1
          new_bad_vertices = []
          next_layer_poly_for_current_poly = [approx]
        except RuntimeError:
          # new_bad_vertices = [np.vstack(new_bad_vertices)]
          new_bad_vertices = [new_bad_vertices]
          pass

      for bvs in new_bad_vertices: # approximate those that has too many facets
        approx_loosen = pltp_mgr.loosen_poly_based_on_vertices(p, bvs, include_old_points = False)
        if p.attribute.is_abstract: # inherit the starting poly
          assert (p.attribute.abs_poly_ref is not None)
          approx_loosen.attribute.abs_poly_ref = p.attribute.abs_poly_ref
          approx_loosen.attribute.abs_poly_layer = p.attribute.abs_poly_layer
        else: 
          approx_loosen.attribute.abs_poly_ref = p
          approx_loosen.attribute.abs_poly_layer = layer_idx - 1
        next_layer_poly_for_current_poly.append(approx_loosen)

      prev_polyResult.append(PolyPreStatus( PolyPrev.UNKNOWN , n_poly))
      next_layer_polytopes += next_layer_poly_for_current_poly

    else:
      prev_polyResult.append(PolyPreStatus( PolyPrev.UNKNOWN , -1))
  # add vi range is outside this function!
  layer_stat.num_fixedpoint = num_fixedpoint

  if p.attribute.is_abstract:
    assert (p.attribute.abs_poly_ref is not None)
    for poly in next_layer_polytopes:
      poly.attribute.is_abstract = True
      poly.attribute.abs_poly_layer = p.attribute.abs_poly_layer
      poly.attribute.abs_poly_ref = p.attribute.abs_poly_ref

  return (next_layer_polytopes, prev_polyResult, layer_stat)

