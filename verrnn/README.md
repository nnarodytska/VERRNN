#### Requirements

* IBM iLOG CPLEX: Due to the problem size limit, promotional version will not work.
* Qhull: Although Python also provides qhull library, we use the latest version from https://github.com/qhull/qhull. After succesful build from source, please update the `External_Qhull_Path` in `polytope/pnt2hresp.py` to point to the qhull executable.
* Python library: numpy, scipy, cdd, pysmt, 
matplotlib

### Instructions

For Property 1: to run experiment using polytope propagation, invariant, and CEGAR method, use:
`python3 pp_test.py`,
`python3 fp_test.py`,
`python3 cegar_test.py`


For Property 1: to run experiment using invariant method, use:
`python3 pulse_test.py`

The results are stored in `*_test.log` files.
