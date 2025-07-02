iteration_count_type = 0
stagnation_count_type = 1


class TSSStopCriteria:

    def __init__(self, criteria_type, criteria_value):
        self.criteria_type = criteria_type
        self.criteria_value = criteria_value

    def is_iteration_count(self):
        return self.criteria_type == iteration_count_type

    def get_iteration_count(self):
        return self.criteria_value

    def is_stagnation_count(self):
        return self.criteria_type == stagnation_count_type

    def get_stagnation_count(self):
        return self.criteria_value


def by_iteration_count(iteration_count):
    return TSSStopCriteria(iteration_count_type, iteration_count)


def by_stagnation_count(stagnation_count):
    return TSSStopCriteria(stagnation_count_type, stagnation_count)
