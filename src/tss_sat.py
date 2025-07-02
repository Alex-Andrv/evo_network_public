from copy import copy

from pysat.formula import IDPool, CNF
from pysat.pb import EncType, PBEnc
from pysat.solvers import Glucose3


def __atleast_equals(lits, weights, eq_lit, first_lit, bound, vpool=IDPool(), encoding=EncType.seqcounter):
    """
        Builds CNF encoding for condition  ((sum(lit*weight) >= bound) == eq_lit) OR first_lit
    """
    if bound == 0:
        return [[eq_lit]]

    atleast_bd = PBEnc.atleast(lits=lits, weights=weights, bound=int(bound), vpool=vpool, top_id=None, encoding=encoding)
    for clause in atleast_bd:
        clause.extend([-eq_lit, first_lit])

    atmost_bd = PBEnc.atmost(lits=lits, weights=weights, bound=int(bound - 1), vpool=vpool, top_id=None, encoding=encoding)
    for clause in atmost_bd:
        clause.append(eq_lit)

    formula = CNF()
    formula.extend(atleast_bd)
    formula.extend(atmost_bd)
    formula.append([eq_lit, -first_lit])
    return formula


def build_cnf(tss, instigators, pb_encoding=EncType.seqcounter):
    dltm = tss.dltm
    n = dltm.nodes_count()
    need_to_activate = tss.threshold - instigators

    if instigators < 0 or instigators > n:
        raise ValueError("Invalid instigators number ({})".format(instigators))
    if need_to_activate < 0:
        raise ValueError("instigators number ({}) is bigger than need-to-activate number ({}))"
                         .format(instigators, tss.threshold))

    curr_row = list(range(1, n + 1))         # initial vars in circuit will have IDs from 1 to 'agents'
    vpool = IDPool()
    vpool.occupy(1, n)
    vpool.top = n+1

    formula = CNF()
    formula.extend(PBEnc.equals(
        lits=curr_row,
        weights=None,
        bound=int(instigators),
        vpool=vpool,
        top_id=None,
        encoding=pb_encoding)
    )

    influencers_of = {}
    for i in range(n):
        agent = dltm.ord_to_agent[i]
        influencers = dltm.graph.get(agent, [])
        influencers_of[i] = list(dltm.agent_to_ord[agent_id] for agent_id in influencers)

    for _ in range(int(need_to_activate)):
        new_row = list(vpool.id() for _ in range(n))
        for i in range(n):
            influencers_in_curr_row = list(curr_row[j] for j in influencers_of[i])
            weights_of_influencers = list(dltm.infl[(dltm.ord_to_agent[j], dltm.ord_to_agent[i])] for j in influencers_of[i])
            formula.extend(__atleast_equals(
                influencers_in_curr_row,
                weights_of_influencers,
                new_row[i], i + 1, dltm.agents[dltm.ord_to_agent[i]], vpool, pb_encoding))
        curr_row = copy(new_row)

    formula.extend(PBEnc.atleast(
        lits=curr_row,
        weights=None,
        bound=int(tss.threshold),
        vpool=vpool,
        top_id=None,
        encoding=pb_encoding)
    )

    return formula


def extract_solution(dltm, cnf_solution):
    """
        Extracts instigators disposition from a CNF solution of Problem 1 instance.
    """
    sliced_vars = cnf_solution[:dltm.nodes_count()]
    true_vars = list(filter(lambda cnf_var: cnf_var > 0, sliced_vars))
    return [dltm.ord_to_agent[i - 1] for i in true_vars]


def solve_tss(
        tss, instigators, pb_encoding=EncType.seqcounter,
        sat_solver=lambda cnf: Glucose3(bootstrap_with=cnf)
):
    """
        Returns either an instigators disposition for a TSS problem instance or
        ``None`` if there's no such disposition.
        Pseudo-boolean constraints encoding and solver initializer are optional.
    """
    cnf = build_cnf(tss, instigators, pb_encoding)
    with sat_solver(cnf) as solver:
        if solver.solve():
            cnf_solution = solver.get_model()
            return extract_solution(tss.dltm, cnf_solution)
        else:
            return None
