class AgentType:
    """
        Type of agent. It could be either conformist or anticoformist
    """

    conformist = 0
    anticonformist = 1


agent_type_id_to_str = {
    AgentType.conformist: 'conf',
    AgentType.anticonformist: 'anticonf'
}

agent_type_str_to_id = {v: k for k, v in agent_type_id_to_str.items()}


class Agent:
    """
        Agent data structure.
    """

    def __init__(self, agent_type, theta):
        if theta < 0 or theta > 1:
            raise ValueError('Theta argument must be in [0;1] range')
        if agent_type is None:
            raise ValueError('Invalid agent type')
        self.agent_type = agent_type
        self.theta = theta

    def __str__(self):
        return '{} theta={}'.format(agent_type_id_to_str[self.agent_type], self.theta)


class SBN:
    """
        Synchronised boolean network structure.
        It's defined as a set of agents (agents are associated with their IDs using ``self.agents``)
        and a agent dependency directed graph ``self.graph``
        Agents IDs class must be hashable.
    """

    def __init__(self):
        self.agents = {}
        self.graph = {}
        self.inv_graph = {}

    def agents_count(self):
        return len(self.agents)

    def put_agent(self, agent_id, agent):
        self.agents[agent_id] = agent

    def put_agents(self, agent_pairs):
        for a_id, a in agent_pairs:
            self.put_agent(a_id, a)

    def add_edge(self, from_id, to_id):
        if not (from_id in self.graph):
            self.graph[from_id] = []
        self.graph[from_id].append(to_id)

        if not (to_id in self.inv_graph):
            self.inv_graph[to_id] = []
        self.inv_graph[to_id].append(from_id)

    def add_edges(self, from_to_pairs):
        for from_id, to_id in from_to_pairs:
            self.add_edge(from_id, to_id)

    def is_conforming(self):
        """
            Returns whether SBN is conforming (== every agent of this SBN is conforming)
        """
        return all(agent.agent_type == AgentType.conformist for agent in self.agents.values())
