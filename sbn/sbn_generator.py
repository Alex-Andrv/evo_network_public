import random

from sbn import *


def __add_edge(graph, e_from, e_to):
    """
        Adds an edge e_from -> e_to into directed graph
    """
    if not (e_from in graph):
        graph[e_from] = set()
    graph[e_from].add(e_to)


def __invert_graph(graph):
    """
        Returns new graph where all edges are inverted
    """
    inv_graph = {}
    for e_from, es_to in graph.items():
        for e_to in es_to:
            __add_edge(inv_graph, e_to, e_from)
    return inv_graph


def __random_sample(n, k, seed=None):
    """
        Returns uniformly distributed random k-sample of set {0,1,...,n-1}.
        Note: return type is list! (not set)
    """

    random.seed(seed)

    s = list(range(0, n))
    r = list(range(0, k))
    for i in range(k, n):
        j = random.randint(0, i)
        if j < k:
            r[j] = s[i]
    return r


def random_graph__erdos_renyi_p(n, p, seed=None):
    """
        Generates random graph using Erdős–Rényi G(n,p) model
    """

    random.seed(seed)

    graph = {i: set() for i in range(1, n + 1)}
    for e_from in range(1, n + 1):
        for e_to in range(1, n + 1):
            if e_from == e_to:
                continue
            if random.random() < p:
                __add_edge(graph, e_from, e_to)

    return graph


def random_graph__erdos_renyi_m(n, m, seed=None):
    """
        Generates random graph using Erdős–Rényi G(n,m) model
    """

    random.seed(seed)

    edges_count = n * (n - 1)
    graph = {i: set() for i in range(1, n + 1)}
    edge_ids = __random_sample(edges_count, m, seed)
    for edge_id in edge_ids:
        e_from = edge_id // (n - 1)
        e_to = edge_id % (n - 1)
        if e_to == e_from:
            e_to += 1
        __add_edge(graph, e_from + 1, e_to + 1)

    return graph


def build_conformist_sbn(graph, theta):
    """
        Creates new SBN with a given graph.
        All agents will be conformist ones and will have the same theta parameter
    """

    sbn = SBN()
    sbn.graph = graph
    sbn.inv_graph = __invert_graph(graph)

    agent_ids = sbn.graph.keys() | (sbn.inv_graph.keys() | set())
    [sbn.put_agent(agent_id, Agent(AgentType.conformist, theta)) for agent_id in agent_ids]

    return sbn
