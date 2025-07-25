
from copy import deepcopy
from random import shuffle
import time

from loguru import logger
import numpy as np
from objetos.problem import Problem
from objetos.solution import Solution


class GreedySearch():
    """
    Algoritmo para inserir estações de energia em todos as rotas para cada veículo.
    
    """
    def __init__(self) -> None:
        pass

    def set_problem(self, problem: Problem):
        self.problem = problem
        self.nearest_dist_customer_matrix = {}
        self.calc_nearest_dist_customer_matrix()

    def free(self):
        pass
    
    def init_solution(self) -> Solution:
        solution = self.create_clustering_solution()
        solution = self.balancing_capacity(solution)
        return solution
    
    def optimize(self, solution: Solution) -> Solution:
        # solution = self.local_search(solution)
        solution = self.insert_depots(solution)
        solution = self.insert_charging_stations(solution)
        solution = self.greedy_optimize_stations(solution)
        solution.set_tour_length(self.problem.calculate_tour_length(solution))
        return solution

    def solve(self, problem, verbose=False) -> Solution:
        self.set_problem(problem)
        self.verbose = verbose
        solution = self.init_solution()
        solution = self.optimize(solution)
        return solution
    
    def calc_nearest_dist_customer_matrix(self):
        all_customers = self.problem.get_all_customers()
        self.nearest_dist_customer_matrix = {}
        
        for i in range(len(all_customers)):
            distances = []
            for j in range(len(all_customers)):
                distances.append(all_customers[i].distance(all_customers[j]))
            argsort_dist = np.argsort(distances)
            self.nearest_dist_customer_matrix[all_customers[i].get_id()] = \
                [all_customers[j].get_id() for j in argsort_dist if i != j]
                
        all_charging_stations = self.problem.get_all_stations()
        self.nearest_dist_charging_matrix = {}
        
        all_nodes = all_customers + all_charging_stations + [self.problem.get_depot()]
        for i in range(len(all_nodes)):
            distances = []
            for j in range(len(all_charging_stations)):
                distances.append(all_nodes[i].distance(all_charging_stations[j]))
            argsort_dist = np.argsort(distances)
            self.nearest_dist_charging_matrix[all_nodes[i].get_id()] = \
                [all_charging_stations[j].get_id() for j in argsort_dist]
                
            
                
    def create_clustering_solution(self) -> Solution:
        solution = Solution()
        node_list = self.problem.get_customer_ids()
        capacity_max = self.problem.get_capacity()
        
        shuffle(node_list)
        skipping_nodes = np.zeros(self.problem.get_num_dimensions())
        idx = 0
        n_tours = 0

        while idx < len(node_list):
            if skipping_nodes[node_list[idx]] == 1:
                idx += 1
                continue
            
            center_node = node_list[idx]
            
            n_tours += 1
            if n_tours == self.problem.get_max_num_vehicles():
                remaining_nodes = [self.problem.get_node_from_id(node_id) for node_id in node_list if skipping_nodes[node_id] == 0]
                solution.add_tour(remaining_nodes)
                break
            
            skipping_nodes[center_node] = 1
            tour = [center_node]
            capacity = self.problem.get_node_from_id(center_node).get_demand()
            
            for candidate_node_id in self.nearest_dist_customer_matrix[center_node]:
                if skipping_nodes[candidate_node_id] == 0:
                    demand = self.problem.get_node_from_id(candidate_node_id).get_demand()
                    if capacity + demand > capacity_max:
                        break
                    tour.append(candidate_node_id)
                    skipping_nodes[candidate_node_id] = 1
                    capacity += demand

            solution.add_tour([self.problem.get_node_from_id(tour[i]) for i in range(len(tour))])

        while n_tours < self.problem.get_max_num_vehicles():
            n_tours += 1
            solution.add_tour([])
            
        solution.set_tour_index()
        return solution
    
    def balancing_capacity(self, solution: Solution) -> Solution:
        tours = solution.get_tours()
        last_tour_idx = len(solution.get_tours()) - 1
        last_tour = tours[last_tour_idx]
        is_customer_in_last_tour = {}
        sum_demand = 0
        
        for node in last_tour:
            is_customer_in_last_tour[node.get_id()] = True
            sum_demand += node.get_demand()
            
        if sum_demand >= self.problem.get_capacity():
            return solution
            
        # escolha um cliente na última rota
        rd_idx = np.random.randint(0, len(last_tour))
        moving_node_id = last_tour[rd_idx].get_id()
        
        for candidate_node_id in self.nearest_dist_customer_matrix[moving_node_id]:
            if candidate_node_id in is_customer_in_last_tour:
                continue
            
            demand = self.problem.get_node_from_id(candidate_node_id).get_demand()
            candidate_tour_index = solution.get_tour_index_by_node(candidate_node_id)
            curr_tour_demand = sum([node.get_demand() for node in tours[candidate_tour_index]])
            
            new_delta = abs((sum_demand + demand) - (curr_tour_demand - demand));
            delta = abs(sum_demand - curr_tour_demand);
            
            if new_delta < delta and sum_demand + demand <= self.problem.get_capacity():
                sum_demand += demand
                is_customer_in_last_tour[candidate_node_id] = True
                last_tour.append(self.problem.get_node_from_id(candidate_node_id))
                tours[candidate_tour_index] = [node for node in tours[candidate_tour_index] if node.get_id() != candidate_node_id]
                break
            
        solution.set_vehicle_tours(tours)
        return solution
    
    def local_search(self, solution: Solution) -> Solution:
        tours = solution.get_basic_tours()
        for i, tour in enumerate(tours):
            tours[i] = self.local_search_2opt(tour)
        solution.set_vehicle_tours(tours)
        return solution
    
    def local_search_2opt(self, tour):
        """
        Execute uma busca local usando o algoritmo 2-opt no rota fornecida.

        Argumentos:
        tour (List[Node]): A rota inicial a ser otimizada.

        Retorna:
        List[Node]: A rota a ser otimizada após aplicar o algoritmo 2-opt.

        Descrição:
        O algoritmo 2-opt é um algoritmo de otimização heurística para encontrar o caminho mais curto em um gráfico.
        Ele busca repetidamente um par de arestas que, se invertidas, resultariam em um caminho mais curto.
        Este processo é repetido até que nenhuma outra melhoria possa ser feita.

        A rota é inicialmente estendida adicionando o nó de depósito no início e no fim.
        Então, o algoritmo itera sobre todos os pares possíveis de arestas dentro da rota.
        Para cada par, uma nova rota é criada invertendo a ordem das arestas.
        """
        n = len(tour)
        tour = [self.problem.get_depot()] + tour + [self.problem.get_depot()]
        while True:
            improvement = False
            for i in range(1, n - 2):
                for j in range(i + 1, n):
                    if j - i == 1:
                        continue
                    new_tour = deepcopy(tour)
                    new_tour[i:j] = reversed(new_tour[i:j])
                    new_distance = sum([new_tour[k].distance(new_tour[k + 1]) for k in range(n - 1)])
                    if new_distance < sum([tour[k].distance(tour[k + 1]) for k in range(n - 1)]):
                        tour = new_tour
                        improvement = True
            if not improvement:
                break
        return tour[1:-1]

    
    def insert_depots(self, solution: Solution) -> Solution:
        vehicle_tours = solution.get_basic_tours()
        for i, tour in enumerate(vehicle_tours):
            vehicle_tours[i] = self.insert_depot_for_single_tour(tour)
        solution.set_vehicle_tours(vehicle_tours)
        return solution
    
    def insert_depot_for_single_tour(self, tour):
        """
        Insere depósitos em uma rota com base na capacidade da rota e na demanda de nós.
        A função garante que a restrição de capacidade seja satisfeita para cada veículo.
        Se a demanda de um nó for maior que a capacidade do veículo, a função insere um depósito,
        o veículo de volta ao depósito para pegar um novo lote de mercadorias e então continua a rota.
        """
        _tour = [self.problem.get_depot()]
        cappacity = self.problem.get_capacity()
        for node in tour:
            if node.is_customer():
                if node.get_demand() > cappacity:
                    _tour.append(self.problem.get_depot())
                    _tour.append(node)
                    cappacity = self.problem.get_capacity() - node.get_demand()
                else:
                    _tour.append(node)
                    cappacity -= node.get_demand()
            if node.is_depot():
                _tour.append(node)
                cappacity = self.problem.get_capacity()
                
        if not _tour[-1].is_depot():
            _tour.append(self.problem.get_depot())

        """
        Otimização gananciosa para posição de depósito:

        Otimize a posição do depósito se o veículo precisar retornar
        ao depósito durante o transporte devido à capacidade excedente.
        depósito -> c1 -> c2 -> depósito -> c3 -> depósito
        =>
        depósito -> c1 -> depósito -> c2 -> c3 -> depósito
        se distância(c1, depósito) + distância(depósito, c2) + distância(c2, c3) < distância(c1, c2) + distância(c2, depósito) + distância(depósito, c3)
        e demanda(c2) + demanda(c3) <= max_capacity
        então troque c2 e depósito
        """
        
        curr_capacity = 0
        
        for i in reversed(range(len(_tour))):
            if i < 2 or i == len(_tour) - 1:
                continue
            
            node = _tour[i]
            if node.is_customer():
                curr_capacity += node.get_demand()
                
            if node.is_depot():
                c1 = _tour[i - 2]
                c2 = _tour[i - 1]
                depot = _tour[i]
                c3 = _tour[i + 1]
                
                d1 = self.problem.get_distance(c1, c2)
                d2 = self.problem.get_distance(c2, depot)
                d3 = self.problem.get_distance(depot, c3)
                
                new_d1 = self.problem.get_distance(c1, depot)
                new_d2 = self.problem.get_distance(depot, c2)
                new_d3 = self.problem.get_distance(c2, c3)
                
                demand_condition = c2.get_demand() + c3.get_demand() <= self.problem.get_capacity()
                distance_condition = d1 + d2 + d3 > new_d1 + new_d2 + new_d3
                
                if demand_condition and distance_condition:
                    _tour[i] = c2
                    _tour[i - 1] = depot
                    curr_capacity += c2.get_demand()
                else:
                    curr_capacity = 0
                
        return _tour
    
    def insert_charging_stations(self, solution: Solution) -> Solution:
        vehicle_tours = solution.get_tours()
        for i, tour in enumerate(vehicle_tours):
            vehicle_tours[i] = self.insert_charging_station_for_single_tour(tour)
        solution.set_vehicle_tours(vehicle_tours)
        return solution
    
    def insert_charging_station_for_single_tour(self, tour):
        remaining_energy = dict()
        min_required_energy = dict()
        complete_tour = []
        skip_node = dict()
        
        depotID = self.problem.get_depot_id()
        remaining_energy[depotID] = self.problem.get_battery_capacity()
        """
        No nó do cliente atual, calcule a energia mínima necessária para que um
        veículo elétrico chegue à estação de carregamento mais próxima.
        """
        for node in tour:
            nearest_station = self.nearest_station(node, node, self.problem.get_battery_capacity())
            min_required_energy[node.get_id()] = self.problem.get_energy_consumption(node, nearest_station)
        
        if len(tour) < 2:
            return tour
        
        i = 0
        from_node = tour[0]
        to_node = tour[1]
        
        while i < len(tour) - 1:
            
            """vá em frente, a energia útil não é suficiente para visitar o próximo nó""" 
            energy_consumption = self.problem.get_energy_consumption(from_node, to_node)
            if energy_consumption <= remaining_energy[from_node.get_id()]:
                if to_node.is_charging_station():
                    remaining_energy[to_node.get_id()] = self.problem.get_battery_capacity()
                else:
                    remaining_energy_node = remaining_energy[from_node.get_id()] - energy_consumption
                    if to_node.get_id() in remaining_energy and remaining_energy_node > remaining_energy[to_node.get_id()]:
                        skip_node[to_node.id] = False
                    remaining_energy[to_node.get_id()] = remaining_energy_node
                complete_tour.append(from_node)
                i += 1
                from_node = tour[i]
                if i < len(tour) - 1:
                    to_node = tour[i + 1]
                continue
            
            find_charging_station = True
            """
            Se houver energia suficiente, encontre a estação mais próxima.
            Se não houver energia suficiente para chegar à estação mais próxima, volte para
            o nó anterior e encontre a próxima estação mais próxima a partir daí.
            """
            while find_charging_station:
                while i > 0 and min_required_energy[from_node.get_id()] > remaining_energy[from_node.get_id()]:
                    i -= 1
                    from_node = tour[i]
                    complete_tour.pop()
                if i == 0:
                    return tour[1:-1]
                if from_node.get_id() in skip_node:
                    return tour[1:-1]
                skip_node[from_node.get_id()] = True
                to_node = tour[i + 1]
                best_station = self.nearest_station(from_node, to_node, remaining_energy[from_node.get_id()])
                if best_station == -1:
                    return tour[1:-1]
                
                complete_tour.append(from_node)
                from_node = best_station
                to_node = tour[i + 1]
                remaining_energy[from_node.get_id()] = self.problem.get_battery_capacity()
                min_required_energy[from_node.get_id()] = 0
                find_charging_station = False                    
                      
        if not complete_tour[-1].is_depot():
            complete_tour.append(self.problem.get_depot())     
        
        return complete_tour
    
    def greedy_optimize_stations(self, solution: Solution) -> Solution:
        vehicle_tours = solution.get_tours()
        for i, tour in enumerate(vehicle_tours):
            vehicle_tours[i] = self.greedy_optimize_station_for_single_tour(tour)
        solution.set_vehicle_tours(vehicle_tours)
        return solution
    
    def greedy_optimize_station_for_single_tour(self, tour):
        """
        * Observe que nesta função, o número de estações de carregamento contínuo S e S' é 1. Mas pode ser mais de 1.
        tour válido após inserir estações de energia
        : depot_L -> c6 -> c5 -> c4 -> c3 -> S(S1 -> S2) -> c2 -> c1 -> depot_R
        Tour reverso
        : depot_R -> c1 -> c2 -> S(S1 -> S2) -> c3 -> c4 -> c5 -> c6 -> depot_L
        Substitua S por outro:
        etapa 1. de depot_R, obtenha uma subrota que o veículo chegue mais longe de depot_R, mas não visite nenhuma estação de carregamento
        : depot_R -> c1 -> c2 -> c3 -> c4 - (energia insuficiente para chegar a c5) -> c5
        : delta_L1 = (d(c2, s1) + d(s1, s2) + d(s2, c3) - d(c2, c3))
        etapa 2: De c2->c3, c3->c4, c4->c5, encontre S' (>= 1 estações de carregamento):
        : delta_L2 = d(c3, S') + d(S', c3) - d(c2, c3)
        : delta_L2 = d(c3, S') + d(S', c4) - d(c3, c4)
        : delta_L2 = d(c4, S') + d(S', c5) - d(c4, c5)
        se delta_L2 < delta_L1 então substitua S por S'
        # veja o artigo: https://doi.org/10.1007/s10489-022-03555-8 para mais detalhes
        """
        
         # calculate required energy to reach node i if vehicle travel in the reverse tour
        required_energy = dict()
        depotID = tour[0].get_id()
        required_energy[depotID] = 0
        optimal_tour = [self.problem.get_depot()]
        
        for i in range(1, len(tour)):
            if tour[i].is_charging_station() or tour[i].is_depot():
                required_energy[tour[i].get_id()] = 0
            else:
                previous_required_energy = required_energy[tour[i - 1].get_id()]
                required_energy[tour[i].get_id()] = previous_required_energy + self.problem.get_energy_consumption(tour[i - 1], tour[i])
                if required_energy[tour[i].get_id()] > self.problem.get_battery_capacity():
                    return tour
        
        """ Viagem para o depósito_R """
        tour = list(reversed(tour))
        energy = self.problem.get_battery_capacity()
        i = 1
        
        while i < len(tour) - 1:
            if not tour[i].is_charging_station():
                energy -= self.problem.get_energy_consumption(optimal_tour[-1], tour[i])
                optimal_tour.append(tour[i])
                i += 1
                continue
            
            if tour[i].is_depot():
                energy = self.problem.get_battery_capacity()
                optimal_tour.append(tour[i])
                i += 1
                continue
                
            # tour[i] é uma estação de carregamento, tente substituí-la por uma melhor
            if i == len(tour) - 1:
                optimal_tour.append(tour[i])
                break
            
            # Calcular delta_L1
            _from_node = optimal_tour[-1]
            num_stations_in_row = 0
            original_distance = 0
            
            while i + num_stations_in_row < len(tour) - 1 and \
                    tour[i + num_stations_in_row].is_charging_station():
                original_distance += _from_node.distance(tour[i + num_stations_in_row])
                _from_node = tour[i + num_stations_in_row]
                num_stations_in_row += 1
            
            next_customer_idx = i + num_stations_in_row
            original_distance += _from_node.distance(tour[next_customer_idx]) 
            delta_L1 = original_distance - tour[i].distance(tour[next_customer_idx])
            _from_node = optimal_tour[-1]
            considered_nodes = []  
            tmp_energy = energy
            for node in tour[next_customer_idx:]:
                considered_nodes.append(node)
                if node.is_charging_station():
                    break
                tmp_energy -= self.problem.get_energy_consumption(_from_node, node)
                if tmp_energy <= 0:
                    break
                _from_node = node
            
            from_node = optimal_tour[-1]
            best_station = tour[i]
            best_station_index = 0 # índice da melhor estação inserido logo após o nó consider_nodes[best_station_index]
            
            for j, node in enumerate(considered_nodes):
                to_node = node
                required_energy_node = required_energy[to_node.get_id()]
                station = self.nearest_station_back(from_node, to_node, energy, required_energy_node)
                if station != -1:
                    delta_L2 = self.problem.get_distance(from_node, station) + self.problem.get_distance(station, to_node) \
                        - self.problem.get_distance(from_node, to_node)
                    if delta_L2 < delta_L1:
                        delta_L1 = delta_L2
                        best_station = station
                        best_station_index = j

                energy -= self.problem.get_energy_consumption(from_node, to_node)
                from_node = to_node

            optimal_tour.extend(considered_nodes[:best_station_index])
            optimal_tour.append(best_station)
            i = i + num_stations_in_row + best_station_index
            energy = self.problem.get_battery_capacity() - self.problem.get_energy_consumption(best_station, tour[i])

        if not optimal_tour[-1].is_depot():
            optimal_tour.append(self.problem.get_depot())
            
        return list(reversed(optimal_tour))
    
    def nearest_station(self, from_node, to_node, energy):
        best_station = -1

        for s in self.nearest_dist_charging_matrix[to_node.get_id()]:
            s_node = self.problem.get_node_from_id(s)
            if self.problem.get_energy_consumption(from_node, s_node) <= energy:
                return s_node

        return best_station
    
    def nearest_station_back(self, from_node, to_node, energy, required_energy):
        min_length = float("inf")
        best_station = -1

        for s in self.nearest_dist_charging_matrix[from_node.get_id()] + [self.problem.get_depot_id()]:
            s_node = self.problem.get_node_from_id(s)
            if self.problem.get_energy_consumption(from_node, s_node) <= energy and \
                self.problem.get_energy_consumption(s_node, to_node) + required_energy < \
                    self.problem.get_battery_capacity():
                length1 = s_node.distance(from_node)
                length2 = s_node.distance(to_node)
                if min_length > length1 + length2:
                    min_length = length1 + length2
                    best_station = s_node
                
                if length1 > min_length:
                    break

        return best_station