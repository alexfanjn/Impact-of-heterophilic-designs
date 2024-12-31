import networkx as nx
import random
from networkx.algorithms import community
import math
import copy
import heapq
import time
import pandas
import os
import matplotlib.pyplot as plt
import numpy as np


class SGA(object):
    def __init__(self, attack_limit, pc, pm, population_list_num, temp_potential_edges, remove_duplicate, operation_mask):
        self.attack_limit = attack_limit
        self.pc = pc
        self.pm = pm
        self.population_list_num = population_list_num
        self.temp_potential_edges = temp_potential_edges
        self.remove_duplicate = remove_duplicate
        self.operation_mask = operation_mask




    def initialize(self, target):
        pop = []
        # generate real potential edges
        self.potential_edges = np.zeros((self.temp_potential_edges.shape[0], 2), dtype=np.int)
        for i in range(self.temp_potential_edges.shape[0]):
            self.potential_edges[i][0] = target
            self.potential_edges[i][1] = self.temp_potential_edges[i]



        candidate_edges = copy.deepcopy(self.potential_edges)
        for j in range(self.population_list_num):
            chromosome_id = np.random.choice(np.arange(len(candidate_edges)), size=self.attack_limit, replace=False)
            chromosome = list(candidate_edges[chromosome_id])
            pop.append(chromosome)

        return pop


    def solve_conflict(self, pop):
        tmp_pop = copy.deepcopy(pop)

        if self.remove_duplicate == False:
            #先处理行内相同的
            for i in range(len(tmp_pop)):
                edges = [tmp_pop[i][j][1] for j in range(len(tmp_pop[i]))]


                remove_dup_edges = list(set(edges))

                while len(remove_dup_edges) != len(edges):
                    # print('id need to be regenerate: ', i)
                    # 存在重复边，需要重新生成
                    still_need = len(edges) - len(remove_dup_edges)
                    candidate_edges = copy.deepcopy(self.potential_edges)
                    candidate_id = np.arange(len(self.potential_edges))
                    new_add = candidate_edges[np.random.choice(candidate_id, size=still_need, replace=False)][:, 1]

                    remove_dup_edges = list(set(remove_dup_edges + list(new_add)))

                for ttt in range(len(remove_dup_edges)):
                    tmp_pop[i][ttt][1] = remove_dup_edges[ttt]
        else:
            #处理行间冲突的
            #先处理行内相同的
            current_valid = []

            for i in range(len(tmp_pop)):
                edges = [tmp_pop[i][j][1] for j in range(len(tmp_pop[i]))]


                remove_dup_edges = list(set(edges))

                while len(remove_dup_edges) != len(edges):
                    # print('id need to be regenerate: ', i)
                    # 存在重复边，需要重新生成
                    still_need = len(edges) - len(remove_dup_edges)
                    candidate_edges = copy.deepcopy(self.potential_edges)
                    candidate_id = np.arange(len(self.potential_edges))
                    new_add = candidate_edges[np.random.choice(candidate_id, size=still_need, replace=False)][:, 1]

                    remove_dup_edges = list(set(remove_dup_edges + list(new_add)))

                for ttt in range(len(remove_dup_edges)):
                    tmp_pop[i][ttt][1] = remove_dup_edges[ttt]


                #行间相同，目前这个肯定是满足行内不重复的
                edges = [tmp_pop[i][j][1] for j in range(len(tmp_pop[i]))]
                remove_dup_edges = sorted(edges)
                while remove_dup_edges in current_valid:
                    #说明行间有重复的
                    still_need = self.attack_limit
                    candidate_edges = copy.deepcopy(self.potential_edges)
                    candidate_id = np.arange(len(self.potential_edges))
                    new_add = candidate_edges[np.random.choice(candidate_id, size=still_need, replace=False)][:, 1]
                    remove_dup_edges = sorted(list(set(list(new_add))))

                for ttt in range(len(remove_dup_edges)):
                    tmp_pop[i][ttt][1] = remove_dup_edges[ttt]

        return tmp_pop



    def crossover_operation(self, parents_pop):
        crossed_pop = copy.deepcopy(parents_pop)
        tag_id = list(np.arange(self.population_list_num))
        point = int(self.attack_limit / 2)

        while (len(tag_id) > 1):
            parents_id = random.sample(tag_id, 2)
            if (random.random() > self.pc):
                parents_pop[parents_id[0]][0:point], parents_pop[parents_id[1]][0:point] = parents_pop[parents_id[1]][0:point], parents_pop[parents_id[0]][0:point]

            crossed_pop.append(parents_pop[parents_id[0]])
            crossed_pop.append(parents_pop[parents_id[1]])

            tag_id.remove(parents_id[0])
            tag_id.remove(parents_id[1])

        crossed_pop = self.solve_conflict(crossed_pop)
        return crossed_pop

    def mutation_operation(self, crossed_pop):
        # print("mutating!")
        mutation_pop = copy.deepcopy(crossed_pop)
        for i in range(len(mutation_pop)):
            if i < self.population_list_num:
                continue
            if (random.random() > self.pm):
                point = np.random.choice(np.arange(self.attack_limit))
                replace_edge_id = np.random.choice(np.arange(len(self.potential_edges)), size=1, replace=False)
                mutation_pop[i][point] = copy.deepcopy(self.potential_edges[replace_edge_id][0])



        mutation_pop = self.solve_conflict(mutation_pop)
        return mutation_pop


    def elite_selection(self, mutation_pop, edges_ranks_score):
        elite_pop = []
        elite_score= []

        tag_id = list(np.arange(len(mutation_pop)))


        while len(tag_id) > 1:

            id_1, id_2 = random.sample(tag_id, 2)

            tag_id.remove(id_1)
            tag_id.remove(id_2)

            if edges_ranks_score[id_1] >= edges_ranks_score[id_2]:
                elite_pop.append(copy.deepcopy(mutation_pop[id_1]))
                elite_score.append(edges_ranks_score[id_1])
            else:
                elite_pop.append(copy.deepcopy(mutation_pop[id_2]))
                elite_score.append(edges_ranks_score[id_2])


        return elite_pop, elite_score





    def find_the_best(self, elite_pop, elite_score):
        id = elite_score.index(max(elite_score))

        return elite_pop[id], elite_score[id]






def main(attack_limit, g):




    pc = 0.5
    pm = 0.8
    population_list_num = 60
    max_iteration = 500

    # obj = 0: mou 模块度下降程度
    # obj = 1: weight 连边度数不明显程度，越大越不明显
    # obj = 2: mou + weight 越大越优
    obj = 0

    sga = SGA(attack_limit, pc, pm, population_list_num, g)

    print('initial mou: ', sga.mou_standard)

    parents_pop = sga.initialize()



    mou_list = []
    weight_list = []
    current_iteration = 0
    iteration_list = []




    while current_iteration < max_iteration:
        crossed_pop = sga.crossover_operation(parents_pop)
        mutation_pop = sga.mutation_operation(crossed_pop)
        elite_pop, elite_mou, elite_weight = sga.elite_selection(mutation_pop, obj)
        perturbation_list, max_mou, weight  = sga.find_the_best(elite_pop, elite_mou, elite_weight, obj)
        print('\n')
        print('current iteration', current_iteration)
        print('current_mou_decrease', max_mou)
        print('current_weight', weight)
        print('current_mou_re', (1-max_mou) * sga.mou_standard)
        print(elite_pop)
        print(elite_mou)


        mou_list.append(max_mou)
        weight_list.append(weight)
        parents_pop = elite_pop
        current_iteration += 1
        iteration_list.append(current_iteration)

    print('----------------')
    perturbation_list, max_mou, weight = sga.find_the_best(elite_pop, elite_mou, elite_weight, obj)
