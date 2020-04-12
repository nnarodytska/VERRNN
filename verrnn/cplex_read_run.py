import cplex
from cplex.exceptions import CplexError

problem = cplex.Cplex("./ilptest-t2.lp")
try:
    problem.write("test_check", "lp")
    problem.solve()
    solution = problem.solution             
        #exit()
    print(solution.get_status())
except CplexError:
    print ("checker: Exception raised during checking")
if not (problem.solution.is_primal_feasible()):  
    print("checker {}: no solutions".format(0))

problem.populate_solution_pool()
print ('# solutions = ', problem.solution.pool.get_num())
