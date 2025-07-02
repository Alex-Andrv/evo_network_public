from collections import deque


def solve(tss, d1=None, d2=None):
    threshold = tss.dltm.agents.copy()

    s = set()
    r = set()
    infl = {agent_id: 0 for agent_id in tss.dltm.agents}

    for agent_id in tss.dltm.agents:
        infl[agent_id] = compute_influence(tss.dltm, agent_id, threshold, r, d1)

    while len(r) < tss.threshold:
        best_agent_id = None
        best_agent_infl = None
        for agent_id, agent_infl in infl.items():
            if agent_id not in r and (best_agent_infl is None or best_agent_infl < agent_infl):
                best_agent_id = agent_id
                best_agent_infl = agent_infl
        s.add(best_agent_id)
        r.add(best_agent_id)
        update_for_new_seed(tss.dltm, best_agent_id, threshold, r, d1, d2)

    return list(s)


# def compute_influence(dltm, agent_id, threshold, r, d1):
#     queue = deque()
#     queue.append((agent_id, 0))
#     visited = set()
#     visited.add(agent_id)
#     new_infl = 0
#     threshold_tmp = threshold.copy()
#     while queue:
#         u, lvl = queue.popleft()
#         for w in dltm.graph[u]:
#             if w in r or w in visited:
#                 continue
#
#             if dltm.infl[(u, w)] >= threshold_tmp[w]:
#                 new_infl += 1
#                 threshold_tmp[w] = 0
#                 if d1 is None or lvl < d1:
#                     queue.append((w, lvl + 1))
#                 visited.add(w)
#             else:
#                 new_infl += dltm.infl[(u, w)] / threshold_tmp[w]
#                 threshold_tmp[w] -= dltm.infl[(u, w)]
#     return new_infl



def compute_influence(dltm, agent_id, threshold, r, d1):
    queue = deque()
    queue.append((agent_id, 0))
    visited = set()
    visited.add(agent_id)
    new_infl = 0
    threshold_tmp = threshold.copy()

    ya_new_infl = 0 #a slightly different heuristic to measure influence
    newly_active = 0
    while queue:
        u, lvl = queue.popleft()
        for w in dltm.graph[u]:
            if w in r or w in visited:
                continue

            if dltm.infl[(u, w)] >= threshold_tmp[w]:
                new_infl += 1
                newly_active += 1

                threshold_tmp[w] = 0
                if (d1 is None) or (lvl < d1):
                    queue.append((w, lvl + 1))
                visited.add(w)
            else:
                new_infl += dltm.infl[(u, w)] / threshold_tmp[w]
                threshold_tmp[w] -= dltm.infl[(u, w)]


    ya_new_infl += newly_active
    for u in threshold_tmp:
        if threshold_tmp[u] > 0:
            ya_new_infl += (threshold[u] - threshold_tmp[u])/ threshold[u]

    return ya_new_infl





# def compute_influence(dltm, agent_id, threshold, r, d1):
#     queue = deque()
#     queue.append((agent_id, 0))
#     visited = set()
#     visited.add(agent_id)
#     new_infl = 0
#     threshold_tmp = threshold.copy()
#
#     ya_new_infl = 0 #a slightly different heuristic to measure influence
#     newly_active = 0
#     while queue:
#         u, lvl = queue.popleft()
#         for w in dltm.graph[u]:
#             if w in r or w in visited:
#                 continue
#
#             if dltm.infl[(u, w)] >= threshold_tmp[w]:
#                 new_infl += 1
#                 newly_active += 1
#
#                 threshold_tmp[w] = 0
#                 if lvl < d1:
#                     queue.append((w, lvl + 1))
#                 visited.add(w)
#             else:
#                 new_infl += dltm.infl[(u, w)] / threshold_tmp[w]
#                 threshold_tmp[w] -= dltm.infl[(u, w)]
#
#
#     ya_new_infl += newly_active
#     for u in threshold_tmp:
#         if threshold_tmp[u] > 0:
#             ya_new_infl += (threshold[u] - threshold_tmp[u])/ threshold[u]
#
#     return ya_new_infl



def update_for_new_seed(dltm, agent_id, threshold, r, d1, d2):
    queue = deque()
    queue.append((agent_id, 0))
    visited = set()
    visited.add(agent_id)

    uini_set = set()
    ci_set = set()
    while queue:
        u, lvl = queue.popleft()
        for w in dltm.graph[u]:
            if w in r or w in visited:
                continue

            if dltm.infl[(u, w)] >= threshold[w]:
                r.add(w)
                visited.add(w)
                threshold[w] = 0
                if d2 is None or lvl < d2:
                    queue.append((w, lvl + 1))
            else:
                threshold[w] -= dltm.infl[(u, w)]
            uini_set.add(w)

    for w in uini_set:
        update_incoming_neighbour_influence(dltm, w, threshold, r, d1, ci_set)
    for ww in ci_set:
        compute_influence(dltm, ww, threshold, r, d1)


def update_incoming_neighbour_influence(dltm, agent_id, threshold, r, d1, ci_set):
    queue = deque()
    queue.append((agent_id, 0))
    visited = set()
    visited.add(agent_id)
    while queue:
        u, lvl = queue.popleft()
        for w in dltm.graph_inv[u]:
            if w in r or w in visited:
                continue

            ci_set.add(w)
            if (w, u) in dltm.infl and dltm.infl[(w, u)] >= threshold[w]:
                visited.add(w)
                if (d1 is None) or (lvl < d1):
                    queue.append((w, lvl + 1))
