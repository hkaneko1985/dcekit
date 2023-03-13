# -*- coding: utf-8 -*- %reset -f
"""
@author: Hiromasa Kaneko
"""
# Demonstration of Genetic Algorithm-based Optimization with Constraints using Gaussian Mixture Regression (GAOC-GMR)
# DEAP is required to be installed.

import random
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from dcekit.generative_model import GMR
from deap import base
from deap import creator
from deap import tools

# general settings
y_targets = [[100, 100],
             [200, 300],
             [300, 700],
             [400, 500],
             [500, 800],
             [600, 1200],
             [700, 1400]]

# settings of GA
number_of_iterations = 10  # iteration number of GA
number_of_population = 100  # number of chromosome in GA
number_of_generation = 30  # nunber of generations in GA
probability_of_crossover = 0.5
probability_of_mutation = 0.2

# Settings of GMR
number_of_components = 20
covariance_type = 'full'

dataset = pd.read_csv('sample_dataset_for_gaoc_gmr.csv', index_col=0)
settings = pd.read_csv('sample_variable_settings_for_gaoc_gmr.csv', index_col=0)
numbers_of_x = list(range(settings.shape[1]))
numbers_of_y = [settings.shape[1], settings.shape[1]+1]
max_boundary = settings.iloc[0, :].values
min_boundary = settings.iloc[1, :].values
zero_one_variable_numbers = np.where(settings.iloc[2, :] == 1)[0]
group_numbers_total = settings.iloc[3, :].values
total_numbers = settings.iloc[4, :].values
group_numbers_total_b_c = settings.iloc[5, :].values
total_b_numbers = settings.iloc[6, :].values
total_c_numbers = settings.iloc[7, :].values
group_numbers_only_one = settings.iloc[8, :].values
only_one_numbers = settings.iloc[9, :].values
group_numbers_other_than_0 = settings.iloc[10, :].values
other_than_0_numbers = settings.iloc[11, :].values
rounding_numbers = settings.iloc[12, :].values
rounding_numbers[zero_one_variable_numbers] = 999

variables_train = np.array(dataset)
# Standardize x and y
autoscaled_variables_train = (variables_train - variables_train.mean(axis=0)) / variables_train.std(axis=0, ddof=1)

# GMR setting
model = GMR(random_state=10)
model.covariance_type = covariance_type
model.n_components = number_of_components

# Modeling
model.fit(autoscaled_variables_train)

# Inverse analysis
y_targets_arr = np.array(y_targets)
#y_targets_arr = y_targets_arr.reshape([len(y_targets), 1])
autoscaled_y_targets_arr = (y_targets_arr - variables_train[:, numbers_of_y].mean(axis=0)) / variables_train[:, numbers_of_y].std(axis=0, ddof=1)
autoscaled_estimated_means, autoscaled_estimated_covariances, weights = model.predict_mog(autoscaled_y_targets_arr, numbers_of_y, numbers_of_x)

for target_y_number in range(len(y_targets)):
    log_probs_ga = np.zeros([number_of_iterations, 1])
    xs_ga = np.zeros([number_of_iterations, settings.shape[1]])
    for iteration_number in range(number_of_iterations):
        # GA
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))  # for minimization, set weights as (-1.0,)
        creator.create('Individual', list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        def create_ind_uniform(min_boundary, max_boundary):
            index = []
            for min, max in zip(min_boundary, max_boundary):
                index.append(random.uniform(min, max))
            return index
        
        
        toolbox.register('create_ind', create_ind_uniform, min_boundary, max_boundary)
        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.create_ind)
        toolbox.register('population', tools.initRepeat, list, toolbox.individual)
        
        
        def evalOneMax(individual):
            value = -10 ** 10
            individual_array = np.array(individual)
    #        individual_array = np.array(np.floor(individual))
            
            if group_numbers_only_one.sum() != 0:
                for group_number in range(1, int(group_numbers_only_one.max()) + 1):
                    variable_numbers = np.where(group_numbers_only_one == group_number)[0]
                    target_number = np.where(individual_array[variable_numbers] == individual_array[variable_numbers].max())[0]
                    individual_array[variable_numbers] = 0
                    individual_array[variable_numbers[target_number]] = only_one_numbers[variable_numbers[target_number]]
            
            if group_numbers_other_than_0.sum() != 0:
                for group_number in range(1, int(group_numbers_other_than_0.max()) + 1):
                    variable_numbers = np.where(group_numbers_other_than_0 == group_number)[0]
                    target_values = individual_array[variable_numbers]
                    indexes = np.argsort(target_values)
                    individual_array[variable_numbers[indexes[0:(len(variable_numbers) - int(other_than_0_numbers[variable_numbers[0]]))]]] = 0
            
            individual_array[zero_one_variable_numbers] = np.array((np.round(individual_array[zero_one_variable_numbers])), dtype='int64')
                        
            if group_numbers_total.sum() != 0:
                for group_number in range(1, int(group_numbers_total.max()) + 1):
                    variable_numbers = np.where(group_numbers_total == group_number)[0]
                    actual_sum_of_components = individual_array[variable_numbers].sum()
                    individual_array[variable_numbers] = individual_array[variable_numbers] / actual_sum_of_components * total_numbers[variable_numbers[0]]
    
            for variable_number in range(len(individual_array)):
                if rounding_numbers[variable_number] != 999:
                    individual_array[variable_number] = np.round(individual_array[variable_number], int(rounding_numbers[variable_number]))
            
            deleting_sample_numbers = np.where(individual_array > max_boundary)[0]
            if len(deleting_sample_numbers) > 0:
                return value,
            deleting_sample_numbers = np.where(individual_array < min_boundary)[0]
            if len(deleting_sample_numbers) > 0:
                return value,
            if group_numbers_total_b_c.sum() != 0:
                for group_number in range(1, int(group_numbers_total_b_c.max()) + 1):
                    variable_numbers = np.where(group_numbers_total_b_c == group_number)[0]
                    actual_sum_of_components = individual_array[variable_numbers].sum()
                    if actual_sum_of_components > total_c_numbers[variable_numbers[0]]:
                        return value,
                    if actual_sum_of_components < total_b_numbers[variable_numbers[0]]:
                        return value,
    
            autoscaled_individual_array = (individual_array - variables_train[:, numbers_of_x].mean(axis=0)) / variables_train[:, numbers_of_x].std(axis=0, ddof=1)
            
            tmps = []
            for i in range(autoscaled_estimated_covariances.shape[0]):
                tmp = np.log(weights[i, target_y_number]) + multivariate_normal.logpdf(autoscaled_individual_array, mean=autoscaled_estimated_means[i, target_y_number, :], cov=autoscaled_estimated_covariances[i, :, :])
                tmps.append(tmp)
            value = logsumexp(tmps)
    
            return value,
        
        
        toolbox.register('evaluate', evalOneMax)
        toolbox.register('mate', tools.cxTwoPoint)
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.05)
        toolbox.register('select', tools.selTournament, tournsize=3)
        
        # random.seed(100)
        random.seed()
        pop = toolbox.population(n=number_of_population)
        
        print('Start of evolution')
        
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        print('  Evaluated %i individuals' % len(pop))
        
        for generation in range(number_of_generation):
            print('-- Generation {0} --'.format(generation + 1))
        
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
        
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < probability_of_crossover:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
        
            for mutant in offspring:
                if random.random() < probability_of_mutation:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
        
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        
            print('  Evaluated %i individuals' % len(invalid_ind))
        
            pop[:] = offspring
            fits = [ind.fitness.values[0] for ind in pop]
        
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
        
            print('  Min %s' % min(fits))
            print('  Max %s' % max(fits))
            print('  Avg %s' % mean)
            print('  Std %s' % std)
        
        print('-- End of (successful) evolution --')
        
        best_individual = tools.selBest(pop, 1)[0]
    #    best_individual_array = np.array(np.floor(best_individual))
        best_individual_array = np.array(best_individual)
        
        if group_numbers_only_one.sum() != 0:
            for group_number in range(1, int(group_numbers_only_one.max()) + 1):
                variable_numbers = np.where(group_numbers_only_one == group_number)[0]
                target_number = np.where(best_individual_array[variable_numbers] == best_individual_array[variable_numbers].max())[0]
                best_individual_array[variable_numbers] = 0
                best_individual_array[variable_numbers[target_number]] = only_one_numbers[variable_numbers[target_number]]
        
        if group_numbers_other_than_0.sum() != 0:
            for group_number in range(1, int(group_numbers_other_than_0.max()) + 1):
                variable_numbers = np.where(group_numbers_other_than_0 == group_number)[0]
                target_values = best_individual_array[variable_numbers]
                indexes = np.argsort(target_values)
                best_individual_array[variable_numbers[indexes[0:(len(variable_numbers) - int(other_than_0_numbers[variable_numbers[0]]))]]] = 0
        
        best_individual_array[zero_one_variable_numbers] = np.array((np.round(best_individual_array[zero_one_variable_numbers])), dtype='int64')
    
        if group_numbers_total.sum() != 0:
            for group_number in range(1, int(group_numbers_total.max()) + 1):
                variable_numbers = np.where(group_numbers_total == group_number)[0]
                actual_sum_of_components = best_individual_array[variable_numbers].sum()
                best_individual_array[variable_numbers] = best_individual_array[variable_numbers] / actual_sum_of_components * total_numbers[variable_numbers[0]]
        
        for variable_number in range(len(best_individual_array)):
            if rounding_numbers[variable_number] != 999:
                best_individual_array[variable_number] = np.round(best_individual_array[variable_number], int(rounding_numbers[variable_number]))
        
        autoscaled_best_individual_array = (best_individual_array - variables_train[:, numbers_of_x].mean(axis=0)) / variables_train[:, numbers_of_x].std(axis=0, ddof=1)
        
        tmps = []
        for i in range(autoscaled_estimated_covariances.shape[0]):
            tmp = np.log(weights[i, target_y_number]) + multivariate_normal.logpdf(autoscaled_best_individual_array, mean=autoscaled_estimated_means[i, target_y_number, :], cov=autoscaled_estimated_covariances[i, :, :])
            tmps.append(tmp)
        value = logsumexp(tmps)
        print(value)
        log_probs_ga[iteration_number, 0] = value
        xs_ga[iteration_number, :] = best_individual_array
        
    
    log_probs_ga = pd.DataFrame(log_probs_ga, columns=['GA'])
    log_probs_ga.to_csv(r'fitness_gaoc_gmr_target_{0}.csv'.format(target_y_number))
    
    xs_ga = pd.DataFrame(xs_ga, columns=settings.columns)
    xs_ga.to_csv(r'x_gaoc_gmr_target_{0}.csv'.format(target_y_number))
