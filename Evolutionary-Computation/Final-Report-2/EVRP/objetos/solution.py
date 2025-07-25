from copy import deepcopy
from hashlib import md5
import numpy as np
from src.utils import logger

class Solution():
    def __init__(self, tours=None):
        """
        A solução contém uma lista de rotas (tour). Cada veículo começa e termina no depósito
        Veja a descrição completa na documentação https://mavrovouniotis.github.io/EVRPcompetition2020/TR-EVRP-Competition.pdf
        Exemplo de uma solução com dois veículos: 0 -> 1 -> 2 -> 3 -> 0 -> 4 -> 0 -> 5 -> 6 -> 0
        as rotas incluem [1, 2, 3, 4] e [5, 6]
        Neste problema, o depósito pode ser visitado várias vezes para cada veículo. (Primeira rota do veículo: 0 -> 1 -> 2 -> 3 -> 4)
        O depósito é considerado e também as estações de carregamento.
        """
        self.tour_index = {}
        self.tour_length = np.inf
        if tours:
            self.tours = tours
            self.set_tour_index()
        else:
            self.tours = []
        
    def add_tour(self, tour):
        self.tours.append(tour)
        
    def get_num_tours(self):
        return len(self.tours)

    def set_tour_index(self):
        self.tour_index = {}
        for idx, tour in enumerate(self.tours):
            for node in tour:
                if node.is_customer():
                    if node.id not in self.tour_index:
                        self.tour_index[node.id] = idx
                    else:
                        logger.warning('Nó {} já em rota {}'.format(node.id, idx))
                        return 0
        return 1
    
    def get_tour_index_by_node(self, node_id):
        return self.tour_index[node_id]
    
    def get_presentation(self):
        list_node = [[x.get_id() for x in tour] for tour in self.tours]
        return md5(str(list_node).encode()).hexdigest()
    
    def __ge__(self, other):
        return self.tour_length >= other.tour_length
    
    def __gt__(self, other):
        return self.tour_length > other.tour_length
    
    def __le__(self, other):
        return self.tour_length <= other.tour_length
    
    def __lt__(self, other):
        return self.tour_length < other.tour_length
    
    def __repr__(self) -> str:
        if self.tour_length:
            presentation = "Fitness Minimo: {}\n".format(self.tour_length)
        else:
            presentation = ""
        for i, tour in enumerate(self.tours):
            presentation += 'Rota {}: '.format(i) + ' -> '.join([str(node.id) for node in tour]) + '\n'
            
        return presentation
    
    def get_tours(self):
        return deepcopy(self.tours)
    
    def get_basic_tours(self):
        tours = []
        for tour in self.tours:
            _tour = [node for node in tour if node.is_customer()]
            tours.append(_tour)
        return tours

    def get_tour_length(self):
        return self.tour_length
    
    def set_tour_length(self, tour_length):
        self.tour_length = tour_length
    
    def to_array(self):
        return np.array([node.id for node in self.tours])
    
    def get_vehicle_tours(self, skip_depot=False, full=True):
        
        if full:
            tours = self.complete_tours
        else:
            tours = self.tours
        """ O veículo não iniciou ou terminou o depósito """
        if len(tours) == 0:
            tours = deepcopy(self.tours)
        if not tours[0].is_depot() or not tours[-1].is_depot():
            return None
        
        vehicle_tours = []
        
        if not skip_depot:
            tour = [tours[0]]
        else:
            tour = []
        
        for idx, node in enumerate(tours):
            if idx == 0 and not skip_depot:
                continue
            if node.is_depot():
                if skip_depot:
                    vehicle_tours.append(tour)
                    continue
                else:
                    tour.append(tours[0])
                    vehicle_tours.append(tour)
                    tour = [tours[0]]
            else:
                tour.append(node)
        return vehicle_tours
    
    def set_vehicle_tours(self, tours):
        self.tours = tours
        self.set_tour_index()
            