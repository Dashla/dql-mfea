#from MFEA.individual import Individual
from individual import Individual
from operators import crossover, mutate, RouletteWheelSelection

import numpy as np
import matplotlib.pyplot as plt


def mfea(tasks,
         pop=100,
         gen=1000,
         selection_process='elitist',
         rmp=0.3,
         p_il=0,
         reps=20,
         method='L-BFGS-B',
         plot=False,
         codification_type=0):

    '''
    :param tasks: List of Task type, can not be empty
    :param pop: Integer, population size
    :param gen: Integer, generation
    :param selection_process: String, only can be 'elitist' or 'roulette wheel', or can be customized
    :param rmp: Float, between 0 and 1
    :param p_il: Float, between 0 and 1
    :param reps: Integer, Repetition times
    :param method: String, details can be seen in document of scipy.optimize.minimize
    :param plot: Boolean, True or false
    :return: TotalEvaluations, ndarray, shape = (reps, gen)
              bestobj, ndarray, shape = (reps, gen, no_of_tasks))
              bestind, ndarray, shape = (shape=(reps, no_of_tasks, D_multitask))
    '''
    assert len(tasks) >= 1 and pop % 2 == 0
    if (pop % 2 != 0): pop += 1

    no_of_tasks = len(tasks)
    D = np.zeros(shape=no_of_tasks)
    for i in range(no_of_tasks):
        tasks[i].dqltask.model.summary()
        D[i] = tasks[i].dim
    D_multitask = int(np.max(D))

    # Wheter the last layer is combined or not
    if codification_type == 1:
        for task in tasks:
            #print(task.dqltask.layer_dims)
            D_multitask += task.dqltask.layer_dims[-1]

    fnceval_calls = np.zeros(shape=reps)
    TotalEvaluations = np.zeros(shape=(reps, gen))
    bestobj = np.empty(shape=(reps, gen, no_of_tasks))
    bestind = np.empty(shape=(reps, no_of_tasks, D_multitask))

    for rep in range(reps):
        print('Repetition: '+str(rep)+' :')
        population = np.asarray([Individual(D_multitask, tasks) for _ in range(2*pop)])
        factorial_costs = np.full(shape=(2 * pop, no_of_tasks), fill_value=np.inf)
        best_tmp = np.full(shape=no_of_tasks, fill_value=np.Inf)
        calls_per_individual = np.zeros(shape=pop)

        factorial_ranks = np.empty(shape=(2 * pop, no_of_tasks))

        for i, individual in enumerate(population[:pop]):
            individual.skill_factor = i % no_of_tasks
            j, factorial_cost, calls_per_individual[i] = individual.evaluate(p_il, method)
            factorial_costs[i, j] = factorial_cost

        fnceval_calls[rep] = fnceval_calls[rep] + np.sum(calls_per_individual)

        mu = 2
        mum = 5
        # generation = 0
        population_on_generation = []
        crossover_matrix = [[0 for x in range(len(tasks))] for _ in range(len(tasks))]
        effective_crossovers = [[0 for x in range(len(tasks))] for _ in range(len(tasks))]
        best_of_gen = []

        for generation in range(gen):
            inorder = np.random.permutation(pop)
            count = pop
            factorial_costs[pop:,:] = np.inf
            for i in range(int(pop/2)):
                p1 = population[inorder[i]]
                p2 = population[inorder[i + int(pop/2)]]
                c1 = population[count]
                c2 = population[count+1]

                count += 2
                if(p1.skill_factor == p2.skill_factor or np.random.uniform()<rmp):

                    u = np.random.uniform(size=D_multitask)
                    cf = np.empty(shape=D_multitask)
                    cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (mu + 1)))
                    cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (mu + 1)))

                    c1.rnvec = crossover(p1.rnvec, p2.rnvec, cf)
                    c2.rnvec = crossover(p2.rnvec, p1.rnvec, cf)

                    c1.rnvec = mutate(c1.rnvec, D_multitask, mum)
                    c2.rnvec = mutate(c2.rnvec, D_multitask, mum)


                    sf1 = 1 + np.round(np.random.uniform())
                    sf2 = 1 + np.round(np.random.uniform())
                    if sf1 == 1:
                        c1.skill_factor = p1.skill_factor
                        c1.parents_skfactor = (p2.skill_factor, p1.skill_factor)
                        c1.parent_fitness = p1.objective
                    else:
                        c1.skill_factor = p2.skill_factor
                        c1.parents_skfactor = (p1.skill_factor, p2.skill_factor)
                        c1.parent_fitness = p2.objective

                    if sf2 == 1:
                        c2.skill_factor = p1.skill_factor
                        c2.parents_skfactor = (p2.skill_factor, p1.skill_factor)
                        c2.parent_fitness = p1.objective
                    else:
                        c2.skill_factor = p2.skill_factor
                        c2.parents_skfactor = (p1.skill_factor, p2.skill_factor)
                        c2.parent_fitness = p2.objective

                else:
                    c1.rnvec = mutate(p1.rnvec, D_multitask, mum)
                    c1.skill_factor = p1.skill_factor
                    c2.rnvec = mutate(p2.rnvec, D_multitask, mum)
                    c2.skill_factor = p2.skill_factor

            for i, individual in enumerate(population[pop:]):
                j, factorial_cost, calls_per_individual[i] = individual.evaluate(p_il, method)
                factorial_costs[pop+i, j] = factorial_cost

                # + Update crossover count
                if individual.parent_fitness is not None:
                    a,b = individual.parents_skfactor
                    crossover_matrix[a][b] += 1
                    if individual.objective < individual.parent_fitness:
                        effective_crossovers[a][b] += 1
            #print(crossover_matrix)
            #print(effective_crossovers)

            fnceval_calls[rep] = fnceval_calls[rep] + np.sum(calls_per_individual)
            TotalEvaluations[rep, generation] = fnceval_calls[rep]

            for j in range(no_of_tasks):
                factorial_cost_j = factorial_costs[:, j]
                indices = list(range(len(factorial_cost_j)))
                indices.sort(key=lambda x: factorial_cost_j[x])
                ranks = np.empty(shape=2 * pop)
                for i, x in enumerate(indices):
                    ranks[x] = i + 1
                factorial_ranks[:, j] = ranks

            for i in range(2*pop):
                population[i].scalar_fitness = 1/np.min(factorial_ranks[i])

            if selection_process == 'elitist':
                scalar_fitnesses = np.array([individual.scalar_fitness for individual in population])
                y = np.argsort(scalar_fitnesses)[::-1]
                population = population[y]
                factorial_costs = factorial_costs[y]
                factorial_ranks = factorial_ranks[y]

            elif selection_process == 'roulette wheel':
                # skill_groups = np.array([Group() for _ in range(no_of_tasks)])
                # for j in range(no_of_tasks):
                #     skill_groups[j].individuals = population[np.where(population.skill_factor==j)]
                # skill_factors = np.array([individual.skill_factor for individual in population])
                # for i in range(pop):
                #     skill = i % no_of_tasks
                #     skill_individuals = population[np.where(skill_factors==skill)]
                #     skill_fitnesses = np.array([individual.scalar_fitness for individual in skill_individuals])
                #     population[i] = skill_individuals[RouletteWheelSelection(skill_fitnesses)]
                scalar_fitnesses = np.array([individual.scalar_fitness for individual in population])
                for i in range(pop):
                    population[i] = population[RouletteWheelSelection(scalar_fitnesses)]

            for j in range(no_of_tasks):
                xxx = np.argmin(factorial_costs[:, j])
                if(best_tmp[j] > factorial_costs[xxx, j]):
                    bestobj[rep, generation, j] = factorial_costs[xxx, j]
                    best_tmp[j] = factorial_costs[xxx, j]

            # if generation % 0 == 0:
            print('Generation '+str(generation)+' :')
            print('Best objective of tasks : ', end='')
            print(best_tmp)
            best_of_gen.append(best_tmp)

            population_on_generation.append([ind.rnvec for ind in population])

        #print('Generation ' + str(generation) + ' :')
        #print('Best objective of tasks : ', end='')
        #print(best_tmp)

        for j in range(no_of_tasks):
            dim = tasks[j].dim
            nnn = np.argmin(factorial_costs[:, j])
            #bestind[rep, j, :dim] = population[nnn].rnvec[:dim]
            bestind[rep, j] = population[nnn].rnvec

    if plot is True:
        pass

    return TotalEvaluations, bestobj, bestind, population_on_generation, crossover_matrix, effective_crossovers, best_of_gen
