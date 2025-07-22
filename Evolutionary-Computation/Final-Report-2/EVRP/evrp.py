import argparse
import os

import numpy as np
from algoritmo.GSGA import GSGA
from objetos.problem import Problem
from algoritmo.GreedySearch import GreedySearch
from src.utils import get_problem_name, logger
import random

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem-path', type=str, default='C:/Users/mathe/OneDrive/Área de Trabalho/EVRP/dados/E-n51-k5.evrp')
    parser.add_argument('-a', '--algorithm', type=str, default='GSGA')
    parser.add_argument('-o', '--result-path', type=str, default='./results/GSGA/')
    parser.add_argument('-n', '--nruns', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
if __name__ == "__main__":
    args = argparser()
    set_random_seed(args.seed)
    problem_name = get_problem_name(args.problem_path)
    problem = Problem(args.problem_path)
    
    if args.algorithm == 'GreedySearch':
        algorithm = GreedySearch()
        
        kwargs = {
            'problem': problem,
            'verbose': True
        }
        
    elif args.algorithm == 'GSGA':
        algorithm = GSGA(population_size=200, generations=1000, 
                          crossover_prob=0.95, mutation_prob=0.1, elite_rate=0.2)
        
        kwargs = {
            'problem': problem,
            'verbose': True,
            'plot_path': os.path.join(args.result_path, problem_name, 'fitness_history.png')
        }
    else:
        raise ValueError(f'Algoritmo inválido {args.algorithm}')
        
    results = []
    
    for i in range(args.nruns):
        result_path = os.path.join(args.result_path, problem_name)
        result_file = os.path.join(result_path, f"run_{i}.txt")
        figure_file = os.path.join(result_path, f"run_{i}.png")
        
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        
        best_individual, mean_fit, best_fit, max_fit, std_fit = algorithm.solve(**kwargs)    
        
        if problem.check_valid_solution(best_individual, verbose=True):
            with open(result_file, 'w') as f:
                f.write(f"Fitness Medio: {mean_fit:.3f}\n")
                f.write(f"Fitness Minimo: {best_fit:.3f}\n")
                f.write(f"Fitness Maximo: {max_fit:.3f}\n")
                f.write(f"Desvio Padrao: {std_fit:.3f}\n")
                for idx, tour in enumerate(best_individual.get_tours()):
                    route_str = ' -> '.join(map(str, tour))
                    f.write(f"Rota {idx + 1}: {route_str}\n")
                
            results.append(best_fit)
            algorithm.free()
            problem.plot(best_individual, figure_file)
            print(best_individual)
        else:
            logger.error('Solução Inválida')
            results.append(np.inf)
            with open(result_file, 'w') as f:
                f.write(f"{np.inf}\n")
        