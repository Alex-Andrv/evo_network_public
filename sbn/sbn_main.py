from sbn_generator import *
from sbn_io import *
from sbn_problems import solve_problem_1

if __name__ == '__main__':

    n = 15
    m = 50
    sbn = build_conformist_sbn(random_graph__erdos_renyi_m(n, m), 0.8)
    write_sbn(sbn, "samples/sbn_test_rnd.txt")

    for instigators in range(1, n):
        solution = solve_problem_1(sbn, instigators, n - instigators)
        print("{} instigators: {}".format(instigators, solution))

        if solution:
            break
