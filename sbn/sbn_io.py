from sbn import *


def read_sbn(path):
    """
        Reads SBN from a given path and returns it.
        See /samples directory for SBN samples.
    """
    with open(path) as f:
        sbn = SBN()

        for line in f.readlines():
            sep_pos = line.rfind("#")
            line_content = line if sep_pos == -1 else line[:sep_pos]
            line_args = line_content.split()

            if len(line_args) == 0:
                continue
            if len(line_args) == 1:
                raise ValueError("Too few arguments, expected either agent declaration or edge(s), got {}".format(line_content))
            else:
                if line_args[1] == '=':
                    if len(line_args) < 4:
                        raise ValueError("Too few arguments at agent declaration. Expected '<agent_id> = <conf|anticonf> <theta>', got {}".format(line_content))
                    sbn.put_agent(line_args[0], Agent(agent_type_str_to_id[line_args[2]], float(line_args[3])))
                else:
                    if line_args[1] == '<-':
                        [sbn.add_edge(line_args[0], to) for to in line_args[2:]]
                    else:
                        raise ValueError("Expected either agent declaration ('a = ...') or edge(s) ('a <- ...'), got {}".format(line_content))

        return sbn


def write_sbn(sbn, path):
    """
        Writes given SBN into a given path.
    """
    file = open(path, "w")

    for agent_id in sbn.agents.keys():
        agent = sbn.agents[agent_id]
        file.write("{} = {} {}\n".format(agent_id, agent_type_id_to_str[agent.agent_type], agent.theta))
    file.write("\n")

    for v in sbn.graph.keys():
        file.write("{} <-".format(v))
        for u in sbn.graph[v]:
            file.write(" {}".format(u))
        file.write("\n")
