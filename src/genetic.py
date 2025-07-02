import functools
import random
from copy import copy

import numpy as np


def genetics_algorithm(init_pop, fit_function, mutation, crossover, stop_criteria):
    population = init_pop
    fitness_scores = [fit_function(ind) for ind in population]
    iterations = 0
    stagnations = 0
    best_solution = None
    best_fitness = float('inf')

    while not ((stop_criteria.is_iteration_count() and iterations >= stop_criteria.get_iteration_count()) or
               (stop_criteria.is_stagnation_count() and stagnations >= stop_criteria.get_stagnation_count())):
        # Выбор пар для кроссовера
        parents = select_parents(population, fitness_scores)
        # Создание нового поколения
        new_generation = []
        for p1, p2 in parents:
            offspring1, offspring2 = crossover(p1, p2)
            new_generation.extend([mutation(offspring1), mutation(offspring2)])

        # Оценка нового поколения
        new_fitness_scores = [fit_function(ind) for ind in new_generation]

        # Выбор наилучших особей для следующего поколения
        population, fitness_scores = select_survivors(new_generation, new_fitness_scores, len(init_pop))

        # Обновление лучшего решения
        current_best = min(fitness_scores)
        if current_best < best_fitness:
            best_fitness = current_best
            best_solution = population[fitness_scores.index(best_fitness)]
            stagnations = 0
        else:
            stagnations += 1

        iterations += 1

    return best_solution, {'iterations': iterations, 'best_fitness': best_fitness}


def select_parents(population, fitness_scores):
    # Простой способ выбора родителей через турнирный отбор или случайный выбор
    return [(population[i], population[i + 1]) for i in range(0, len(population), 2)]


def select_survivors(new_generation, new_fitness_scores, num_survivors):
    # Селекция лучших особей для следующего поколения
    sorted_indices = sorted(range(len(new_fitness_scores)), key=lambda x: new_fitness_scores[x])
    return [new_generation[i] for i in sorted_indices[:num_survivors]], [new_fitness_scores[i] for i in
                                                                         sorted_indices[:num_survivors]]

def using_1p1(init_vec, fit_function, mutation, stop_criteria):
    cur_vec = init_vec
    cur_fit, active  = fit_function(cur_vec)
    cache = set([tuple(init_vec)])
    iterations = 0
    stagnations = 0
    t_size = []
    while not ((stop_criteria.is_iteration_count() and iterations >= stop_criteria.get_iteration_count())
               or (stop_criteria.is_stagnation_count() and stagnations >= stop_criteria.get_stagnation_count())):
        
        new_vec, cnt_mutation = mutation(cur_vec)
        iterations += cnt_mutation
        # if tuple(new_vec) in cache:
        #     continue
        
        cache.add(tuple(new_vec))

        new_fit, new_active = fit_function(new_vec)

        t_size.append(cur_fit)
        if new_fit < cur_fit:
            for _ in range(cnt_mutation):
                print(f"best_fit={new_active}, k={sum(new_vec)}")
            # print(f"Best k = {cur_fit}")
            stagnations = 0
        else:
            for _ in range(cnt_mutation):
                print(f"best_fit={active}, k={sum(cur_vec)}")
            stagnations += 1

        if new_fit <= cur_fit:
            cur_vec = new_vec
            cur_fit = new_fit
            active = new_active

    print(f"best_fit={active}, k={sum(cur_vec)}")
    
    return cur_vec, {'iterations': iterations, 't_size': t_size}


def using_1cl(init_vec, fit_function, mutation, lmbd, stop_criteria):
    if lmbd < 1:
        raise ValueError('Lambda parameter in (1,lambda) algorithm must be >= 1')

    cur_vec = copy(init_vec)
    best_vec = copy(init_vec)
    best_fit = None
    iterations = 0
    stagnations = 0
    while not ((stop_criteria.is_iteration_count() and iterations >= stop_criteria.get_iteration_count())
               or (stop_criteria.is_stagnation_count() and stagnations >= stop_criteria.get_stagnation_count())):
        new_opt_vec = None
        new_opt_fit = None
        for _ in range(lmbd):
            new_vec = mutation(cur_vec)
            new_fit, active = fit_function(new_vec)
            if not new_opt_fit or new_fit < new_opt_fit:
                new_opt_vec = new_vec
                new_opt_fit = new_fit

        iterations += 1
        if best_fit is None or new_opt_fit < best_fit:
            best_vec = new_opt_vec
            best_fit = new_opt_fit
            stagnations = 0
        else:
            stagnations += 1

        cur_vec = new_opt_vec

    return best_vec, {'iterations': iterations}


def using_custom_ga(init_vec, fit_function, mutation, crossover, l, h, g, stop_criteria):
    if l < 0 or h < 0 or g < 0 or l + h + g < 1:
        raise ValueError('l, h and g parameters must be non-negative integers; also l + h + g must be >= 1')
    if g % 2 == 1:
        raise ValueError('g must be even number')

    def cmp_by_2nd_asc(idx1, idx2):
        return 0 if idx1[1] == idx2[1] else (-1 if idx1[1] < idx2[1] else 1)

    sz = l + h + g  # population size
    population_with_fit = [with_fit(item, fit_function) for item in [init_vec] * sz]
    cache = set(tuple(init_vec))
    iterations = 0
    stagnations = 0

    best_fit = min(population_with_fit, key=functools.cmp_to_key(cmp_by_2nd_asc))[1]

    while not ((stop_criteria.is_iteration_count() and iterations >= stop_criteria.get_iteration_count())
               or (stop_criteria.is_stagnation_count() and stagnations >= stop_criteria.get_stagnation_count())):
        new_population_with_fit = []
        u = [(i, population_with_fit[i][1]) for i in range(sz)]

        u.sort(key=functools.cmp_to_key(cmp_by_2nd_asc))

        [new_population_with_fit.append(population_with_fit[u[i][0]]) for i in range(l)]  # elitism

        for _ in range(h):
            i = weighted_random_index(u)
            mutated, cnt = mutation(population_with_fit[i][0])
            iterations += cnt
            for _ in range(cnt):
                print(f"best_fit={best_fit}, k={best_fit}")
            new_population_with_fit.append(with_fit(mutated, fit_function))  # mutation

        for _ in range(g // 2):
            i1, i2 = weighted_random_index(u), weighted_random_index(u)
            iterations += 1
            print(f"best_fit={best_fit}, k={best_fit}")
            crossed_with_fit = [with_fit(crossed, fit_function) for crossed in
                                crossover(population_with_fit[i1][0], population_with_fit[i2][0])]
            new_population_with_fit.extend(crossed_with_fit)  # crossover

        
        new_best_fit = min(new_population_with_fit, key=functools.cmp_to_key(cmp_by_2nd_asc))[1]
        if best_fit > new_best_fit:
            best_fit = new_best_fit
        # if new_best_fit  < best_fit:
        #     stagnations = 0
        # else:
        #     stagnations += 1

        population_with_fit = new_population_with_fit

    best_in_population = min(population_with_fit, key=functools.cmp_to_key(cmp_by_2nd_asc))[0]
    return best_in_population, {'iterations': iterations}


def weighted_random_index(u):
    r = random.uniform(0, sum([item[1] for item in u]))
    for i in range(len(u)):
        if r <= u[i][1]:
            return i
        else:
            r -= u[i][1]
    raise AssertionError('Unreachable state')


def with_fit(vec, fit_function):
    return vec, fit_function(vec)[0]


def non_increasing_mutation_of(base_mutation, vec, env):
    cnt_mutations = 0
    vec_wt = sum(vec)
    new_vec = None
    new_vec_wt = None
    while new_vec_wt is None or new_vec_wt > vec_wt:
        new_vec = base_mutation(vec, env)
        new_vec_wt = sum(new_vec)
        cnt_mutations += 1
    return new_vec, cnt_mutations


def default_mutation(vec, env=None):
    n = len(vec)
    return [1 - vec[i] if random.randrange(n) == 0 else vec[i] for i in range(n)]


def non_increasing_default_mutation(vec):
    return non_increasing_mutation_of(default_mutation, vec, None)


def crossover(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length")

        # Выбор случайной точки для кроссовера
    point = random.randint(1, len(vec1) - 1)

    # Создание новых векторов путем обмена частями
    new_vec1 = vec1[:point] + vec2[point:]
    new_vec2 = vec2[:point] + vec1[point:]

    return new_vec1, new_vec2


def init_doerr_env(beta, ndiv2):
    env = []
    for i in range(ndiv2):
        env.append((i + 1) ** -beta)
    for i in range(ndiv2 - 1):
        env[i + 1] += env[i]
    last = env[len(env) - 1]
    for i in range(ndiv2):
        env[i] /= last
    return env


def generate_alpha_for_doerr_mutation(env):
    x = random.random()
    for i in range(len(env)):
        if x <= env[i]:
            return i + 1
    raise AssertionError('Unreachable state')


def doerr_mutation(vec, env):
    n = len(vec)
    alpha = generate_alpha_for_doerr_mutation(env)
    return [1 - vec[i] if random.random() < alpha / n else vec[i] for i in range(n)]


def non_increasing_doerr_mutation(vec, env):
    return non_increasing_mutation_of(doerr_mutation, vec, env)


def two_point_crossover(vec1, vec2):
    n = len(vec1)
    i1, i2 = random.randrange(n), random.randrange(n - 1)
    if i2 >= i1:
        i2 += 1

    new_vec1, new_vec2 = [], []
    for i in range(n):
        if i < i1 == i < i2:
            new_vec1.append(vec1[i])
            new_vec2.append(vec2[i])
        else:
            new_vec1.append(vec2[i])
            new_vec2.append(vec1[i])

    return new_vec1, new_vec2
