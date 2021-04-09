import numpy as np
import random
import math
from collections import defaultdict

MUTATION_RATE = 60
MUTATION_REPEAT_COUNT = 2
WEAKNESS_THRESHOLD = 100000

# program
class Genome():
    chromosomes = []
    fitness = 999999


class Omc():
    """
    
    """
    def __init__(self, h_k_l: int, p_k_l: int, distance: list, e: float, p_0: int, v: float, E_high: list, E_low: list, theta_k_l: int, p_r, prev_position_mc, new_index_node, new_index_MC, dynamic_e_k):
        self.h_k_l = h_k_l
        self.p_k_l = p_k_l
        self.distance = distance
        self.e = e
        self.p_0 = p_0
        self.v = v
        self.E_high = E_high
        self.E_low = E_low
        self.theta_k_l = theta_k_l
        self.p_r = p_r
        self.number_fixed_elements = 4
        self.prev_position_mc = prev_position_mc
        self.new_index_node = new_index_node
        self.new_index_MC = new_index_MC
        self.dynamic_e_k = dynamic_e_k

    def create_new_population(self, size, number_of_nodes, number_bit_zero=0):
        population = []
        for x in range(size):
            new_genome = Genome()
            new_genome.chromosomes = random.sample(range(0, number_of_nodes), number_of_nodes)
            if number_bit_zero:
                for i in range(len(new_genome.chromosomes)):
                    if new_genome.chromosomes[i] >= number_of_nodes - number_bit_zero:
                        new_genome.chromosomes[i] = -1
            new_genome.fitness = self.evaluate_fitness(new_genome.chromosomes)
            # print(new_genome.chromosomes)

            population.append(new_genome)

        return population


    def evaluate_fitness(self, chromosomes: list):
        calculated_fitness = 0
        for i in range(len(chromosomes)):
            if chromosomes[i] == -1:
                continue
            if i >= len(self.prev_position_mc):
                print("i = {}".format(i))
            if chromosomes[i] >= len(self.new_index_node):
                print("chromosomes[i] = {}".format(chromosomes[i]))
            d_i_j = self.distance[self.prev_position_mc[i]][self.new_index_node[chromosomes[i]]]
            moving_enery = d_i_j * self.e
            if chromosomes[i] > len(self.E_high):
                print("1 chromosomes[i] = {}".format(chromosomes[i]))
            charging_energy =  self.p_0 * min(self.E_high[chromosomes[i]], self.E_low[chromosomes[i]]) / self.p_r 
            calculated_fitness += moving_enery + charging_energy

        calculated_fitness = np.round(calculated_fitness, 2)
        return calculated_fitness


    def find_best_genome(self, population):
        all_fitness = [i.fitness for i in population]
        bestFitness = min(all_fitness)
        return population[all_fitness.index(bestFitness)]


    def tournament_selection(self, population, k):
        selected = [population[random.randrange(0, len(population))] for i in range(k)]
        best_genome = self.find_best_genome(selected)
        return best_genome


    def reproduction(self, population):
        parent1 = self.tournament_selection(population, 10).chromosomes
        parent2 = self.tournament_selection(population, 6).chromosomes
        while parent1 == parent2:
            parent2 = self.tournament_selection(population, 6).chromosomes

        return self.order_one_crossover(parent1, parent2)


    def order_one_crossover(self, parent1, parent2):
        if -1 not in parent1:
            size = len(parent1)
            child = [-1] * size

            point = random.randrange(1, size - self.number_fixed_elements)

            for i in range(point, point + self.number_fixed_elements):
                child[i] = parent1[i]
            point += self.number_fixed_elements
            point2 = point
            # print(child)
            # print(parent1)
            # print(parent2)
            for i in range(point2, point2 + size - self.number_fixed_elements):
                j = i
                # print(child, parent2[j%size])
                # print("HEHEHEHE {}".format(j%size))
                while parent2[j%size] in child:
                    j += 1
                child[i%size] = parent2[j%size]
            # print(child)

            # print("\n")
            if random.randrange(0, 100) < MUTATION_RATE:
                child = self.swap_mutation(child)

        else:
            child = self.swap_mutation(parent1)

        # Create new genome for child
        new_genome = Genome()
        new_genome.chromosomes = child
        new_genome.fitness = self.evaluate_fitness(child)
        return new_genome


    def swap_mutation(self, chromo):
        for x in range(MUTATION_REPEAT_COUNT):
            p1, p2 = [random.randrange(1, len(chromo) - 1) for i in range(2)]
            while p1 == p2 or chromo[p1] == chromo[p2]: # chromo[p1] == chromo[p2] == -1
                p2 = random.randrange(1, len(chromo) - 1)
            log = chromo[p1]
            chromo[p1] = chromo[p2]
            chromo[p2] = log
        return chromo


    def generic_algorithm(self, pop_size, max_generation):
        is_last_round = self.h_k_l != self.p_k_l
        all_best_fitness = []
        population = self.create_new_population(pop_size, self.h_k_l, self.h_k_l - self.p_k_l)
        generation = 0
        while generation < max_generation:
            generation += 1

            # split population to reproduct
            group_parent1, group_parent2 = [], []
            for i in range(0, int(len(population) / 2 - 1), 2):
                group_parent1.append(population[i])
                group_parent2.append(population[i+1])

            for i in range(len(group_parent1)):
                if group_parent1[i].chromosomes != group_parent2[i].chromosomes:                
                    population.append(self.order_one_crossover(group_parent1[i].chromosomes, group_parent2[i].chromosomes))  # ??????????????????????? chia 2 tập

            population = sorted(population, key= lambda x: x.fitness)[:100]

            average_fitness = round(np.sum([genom.fitness for genom in population]) / len(population), 2)
            best_genome = self.find_best_genome(population)
            # if generation % 100 == 0:
            #     print("\n")
            #     print("Generation: {0}\nPopulation Size: {1}\t Average Fitness: {2}\nBest Fitness: {3}"
            #             .format(generation, len(population), average_fitness,
            #                     best_genome.fitness))

            all_best_fitness.append(best_genome.fitness)
        print("Best solution: ", best_genome.fitness)
        q,t,g = self.get_q_t_g(best_genome.chromosomes)
        for i in range(self.p_k_l):
            for j in range(self.h_k_l):
                if q[i][j] > 0:
                    print("\tMC {} charged node {}".format(self.new_index_MC[j], self.new_index_node[i]), end=': ')
                    print("Node {}, time {} + {} ".format(self.new_index_node[i], t[i][j], g[i][j]))
        return self.get_position_of_MCs_in_round_L(q,t,g,self.h_k_l, self.p_k_l, self.new_index_MC, self.new_index_node), self.calculate_total_time_for_round_L(q,t,g,self.h_k_l,self.p_k_l), self.get_consumed_energy(q,t,g,self.p_k_l, self.h_k_l, self.new_index_MC, self.new_index_node, self.prev_position_mc), best_genome.fitness
        
    def get_q_t_g(self, best_chromo):
        # best_chromo = [0,1,2,3-1]
        q = [[0 for _ in range(self.h_k_l)] for _ in range(self.p_k_l)]
        t = [[0 for _ in range(self.h_k_l)] for _ in range(self.p_k_l)]
        g = [[0 for _ in range(self.h_k_l)] for _ in range(self.p_k_l)]
        for idx, val in enumerate(best_chromo):
            if val != -1:
                q[val][idx] = 1
                t[val][idx] = min(self.E_low[val], self.E_high[val]) / self.p_r
                g[val][idx] = self.theta_k_l - t[val][idx]
        return q,t,g
    
    def calculate_total_time_for_round_L(self, q, t, g, h_k_l, p_k_l):  
        res = 0
        for i in range(p_k_l):
            for j in range(h_k_l):
                if q[i][j] > 0:
            # print("Node {}, time {} + {} ".format(new_index_node[i], t[i][j].solution_value(), g[i][j].solution_value()))
                    res = max(res, t[i][j] + g[i][j])
        return res 
    def get_position_of_MCs_in_round_L(self, q,t,g,h_k_l,p_k_l, new_index_MC, new_index_node):
        position = defaultdict(int)
        for j in range(h_k_l):
            for i in range(p_k_l):
                if q[i][j] > 0:
                    position[new_index_MC[j]] = new_index_node[i]
        return position

    # def update_e_k(q,t,g,h_k_l,p_k_l, new_index_node):
    #     total_time_for_round_L = self.calculate_total_time_for_round_L(q,t,g,h_k_l,p_k_l)
    #     for i in range(len(self.dynamic_e_k)):
    #         self.dynamic_e_k[i] -= total_time_for_round_L * consumption_energy_rate[i]
    #     for j in range(h_k_l):
    #         for i in range(p_k_l):
    #         if q[i][j] > 0:
    #             # cộng lại phần bên trên bị trừ
    #             self.dynamic_e_k[new_index_node[i]] += total_time_for_round_L * consumption_energy_rate[new_index_node[i]]  
    #             # trừ đi năng lượng đợi MC di chuyển tới
    #             self.dynamic_e_k[new_index_node[i]] -= g[i][j] * consumption_energy_rate[new_index_node[i]] 
    #             # cộng thêm năng lượng được sạc 
    #             self.dynamic_e_k[new_index_node[i]] += t[i][j] * self.p_r 
    #             # trừ đi năng lượng trong lúc đợi MC lâu nhất sạc xong
    #             self.dynamic_e_k[new_index_node[i]] -= (total_time_for_round_L - t[i][j] - g[i][j]) * consumption_energy_rate[new_index_node[i]]    

    def get_consumed_energy(self, q,t,g, p_k_l, h_k_l, new_index_MC, new_index_node, prev_pos_MC):
        consumed = defaultdict(int)
        for i in range(p_k_l):
            for j in range(h_k_l):
                if q[i][j] > 0:
                    consumed[new_index_MC[j]] = t[i][j] * self.p_0 + self.distance[new_index_node[i]][prev_pos_MC[j]] * self.e
        return consumed

def create_distance(locations: list, N):
    distance = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            distance[i][j] = math.sqrt((locations[i][0]-locations[j][0])**2 + (locations[i][1]-locations[j][1])**2)
    return distance


if __name__ == "__main__":
    node_position = [(894, 215), (458, 563), (388, 393), (825, 403), (443, 590), (492, 641), (548, 669), (605, 939), (806, 137), (293, 147), (820, 346), (714, 817), (205, 423), (218, 209), (337, 192), (945, 123), (897, 41), (368, 996), (309, 958), (196, 963)]
    E_HIGH = [6736.2912671750855, 5266.42126145219, 7556.794023270528, 4784.749399347136, 7646.283247952383, 7154.447297804435, 6842.014744365264, 5831.040883587823, 8569.016743308594, 8437.348122322983, 6100.846066734253, 6287.910820892774, 5681.271006367664, 7858.563061034141, 4693.042104921846, 6051.780010668692, 8466.231698913072, 4348.154296947191, 6787.664903269214, 4353.454447416565]
    E_LOW = [3028.4127220587748, 4504.99844584818, 2046.7100068065693, 4634.347986304589, 1930.5446741816622, 2364.972616573158, 2604.5164617592854, 3475.767954917671, 956.3725641937804, 1025.9100912315712, 2887.780534782347, 2662.915623223749, 3141.6131753485097, 1326.7573391861433, 3749.2933326006423, 2665.926652271616, 800.6043560205367, 3876.896865458234, 2044.4043590737774, 3830.7707535036916]

    THETA_K_L = 11633.411810055528
    P_R = 5
    charging_efficency = 1
    p0 = P_R / charging_efficency
    e = 1

    list_node_size = len(node_position)

    h_k_l = 20 # MCs
    p_k_l = 11 # nodes
    dist = create_distance(node_position, h_k_l)
    
    new_index_MC = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    new_index_node = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    dynamic_e_k=[[2030.07491869, 2093.11751573, 2258.46942358, 2707.81118006, 3085.86166198, 3645.12318381, 2170.07477154, 2101.95279494
, 2232.03139128, 4441.88783409, 4407.70591832, 4969.09276082
, 3440.3188679,  3157.5526229,  3941.14672228, 3260.87562516
, 5504.63483831, 5940.53486818, 4865.13193977, 4854.17735668
, 4844.00755732, 6295.62892772, 6189.83722586, 6153.00646977
, 6455.43089568, 4316.88182158, 6399.08349163, 6989.34304853
, 7630.56000186, 6344.73601199, 5401.738482, 7692.05808551
, 5791.63766039, 5135.32630211, 5932.95753396, 8057.4174139
, 5338.85125662, 8389.42309417, 7313.39552119, 4522.26045774
, 7461.04638738, 6402.9025617,  7593.95706372, 6517.60838828
, 8742.21026077, 4578.56061837, 8002.55661009, 6551.60456583
, 3481.50836334, 4244.0743416,  9375.19608298, 5455.57765555
, 7214.46667671, 3499.21728874, 5902.81580927, 2351.37754992
, 8093.13123236, 8530.20267754, 6335.9361621,  7441.48889082,
 10173.65652753, 8655.57002396, 8722.48309763, 6815.28115638
, 2043.19137185, 7858.59621854, 10442.59585195, 7597.35034962
, 6541.09458428, 7116.61701528, 10589.26524348, 10472.73939793
, 5932.4796447,  7737.0483425,  6705.8730996, 3707.35128393, 5491.46198404, 6865.15815876, 10505.9066442, 4647.68530441
, 3797.41788784, 5649.113668, 8261.17976438, 10104.07158132
, 9695.08573304, 7844.32881966, 10733.64118598, 9324.12406682
, 6418.35281041, 8914.31790863, 9409.72262046, 10693.78189714
, 7294.98066797, 7131.04248584, 9065.19984783, 10694.51788222
, 8259.58414609, 10164.14936362, 10706.83309282, 9384.87956289]]
    prev_position_mc = [i for i in range(h_k_l)]
    import time
    start = time.time()
    omc = Omc(h_k_l, p_k_l, distance=dist, e=1, p_0=p0, v=1, E_high=E_HIGH, E_low=E_LOW, theta_k_l=THETA_K_L, p_r=P_R, prev_position_mc=prev_position_mc, new_index_node=new_index_node, new_index_MC=new_index_MC, dynamic_e_k=dynamic_e_k)
    omc.generic_algorithm(pop_size=100, max_generation=1000)
    print(time.time() - start)
