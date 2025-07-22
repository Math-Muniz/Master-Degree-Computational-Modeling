import random
import pprint
from typing import Any, Dict, Sequence, Union, Optional
from keras.optimizers import Adam  # type: ignore
from models import matrix_factorization
from clean_data import clean_100k, clean_1M
from pre_processing import train_test_cf
from models_test import matrix_factorization, neural_network ,calculate_precision_recall
import operator
import math
import uuid

class Individual:
    def __init__(self, genes: Sequence, handler, x_train, y_train, hyperparameters: Dict[str, Any], limits: Dict[str, Any], **kwargs: Any):
        self.genes = genes
        self.handler = handler
        self.x_train = x_train
        self.y_train = y_train
        self.hyperparameters = hyperparameters
        self.limits = limits
        self.kwargs = kwargs
        self.fitness = None
        self.precision = None
        self.recall = None
        self.origin = "inicial"  # Padrão é "inicial", mas será alterado dependendo da criação
        self.parents = []   # Atributo para armazenar os pais, se houver
        self.id = uuid.uuid4()   
        
    def evaluate_fitness(self) -> float:
        """Avalia a fitness do indivíduo e exibe mais métricas."""
        if self.fitness is not None:
            return self.fitness

        # Treinar o modelo e calcular o fitness
        model, history = self.handler(
            df=self.x_train,
            train=self.x_train,
            valid=self.y_train,
            n_latent_factors=self.hyperparameters['n_latent_factors'],
            batch_size=self.hyperparameters['batch_size'],
            learning_rate=self.hyperparameters['learning_rate'],
            epochs=10,
            optimizer_fn=Adam,
            loss="mae",
            verbose=0,
        )

        # Fitness (usando a perda de validação final)
        self.fitness = history.history['val_loss'][-1]

        # Mostrar perda de treino e validação ao longo das épocas
        print("Perda de treino:", history.history['loss'])
        print("Perda de validação:", history.history['val_loss'])

        # Calcular precision e recall e armazenar nos atributos
        self.precision, self.recall = calculate_precision_recall(model, self.y_train.user_id, self.y_train.movie_id, self.y_train.rating)

        print(f"Precisão: {self.precision}")
        print(f"Recall: {self.recall}")
        print(f"Fitness(MAE): {self.fitness}")

        return self.fitness

    # Método de reprodução
    def reproduce(self, partner, rate=1.0) -> 'Individual':
        child_hyperparameters = {}
        crossover_occurred = False

        for param in self.hyperparameters:
            if random.random() < rate:
                child_hyperparameters[param] = partner.hyperparameters[param]
                crossover_occurred = True
            else:
                child_hyperparameters[param] = self.hyperparameters[param]

        child = Individual(self.genes, self.handler, self.x_train, self.y_train, hyperparameters=child_hyperparameters, limits=self.limits, **self.kwargs)
        
        # Marcar origem como crossover e registrar pais
        if crossover_occurred:
            child.origin = "crossover"
            child.parents = [self, partner]  # Registrar os pais

        return child

    def mutate(self, rate: float = 0.1) -> None:
        mutated = False
        for gene in self.genes:
            if random.random() < rate:
                mutated = True
                if gene == 'n_latent_factors':
                    # Valor completamente aleatório dentro dos limites
                    self.hyperparameters[gene] = random.randint(
                        self.limits['n_latent_factors'][0], 
                        self.limits['n_latent_factors'][1]
                    )
                elif gene == 'batch_size':
                    # Valor completamente aleatório dentro dos limites
                    self.hyperparameters[gene] = random.randint(
                        self.limits['batch_size'][0], 
                        self.limits['batch_size'][1]
                    )
                elif gene == 'learning_rate':
                    # Valor completamente aleatório dentro dos limites
                    self.hyperparameters[gene] = random.uniform(
                        self.limits['learning_rate'][0], 
                        self.limits['learning_rate'][1]
                    )
        if mutated:
            self.origin = "mutação"
            print(f"Mutação ocorrida em {gene}: Novo valor {self.hyperparameters[gene]}")


    def duplicate(self) -> 'Individual':
        """Cria uma cópia do indivíduo e mantém a origem."""
        clone = Individual(self.genes, self.handler, self.x_train, self.y_train, hyperparameters=self.hyperparameters.copy(), limits=self.limits, **self.kwargs)
        clone.origin = self.origin  # Preservar a origem ao duplicar
        return clone

    def __str__(self):
        """Retorna os hiperparâmetros que identificam o indivíduo."""
        return pprint.pformat(self.hyperparameters)

class Population:
    """
    Uma coleção de indivíduos (representados como soluções com hiperparâmetros).
    Pode ser inicializada com uma sequência de indivíduos ou especificando um tamanho de população,
    onde indivíduos aleatórios serão gerados.
    """

    def __init__(self, genes: Sequence[str], handler: Any, individuals: Union[Sequence[Dict[str, Any]], Sequence[Individual], int], x_train: Any = None, y_train: Any = None, limits: Optional[Dict[str, Any]] = None, **kwargs):
        self.genes = genes
        self.handler = handler
        self.x_train = x_train
        self.y_train = y_train
        self.kwargs = kwargs
        self.limits = limits or {  # Se os limites não forem fornecidos, define valores padrões
            'n_latent_factors': (32, 128),
            'batch_size': (32, 256),
            'learning_rate': (1e-4, 1e-2)
        }
        
        # Criar indivíduos (aleatoriamente ou a partir de uma lista)
        if isinstance(individuals, int):
            self.individuals = [self.spawn() for _ in range(individuals)]
        elif isinstance(individuals, Sequence):
            self.individuals = []
            for individual in individuals:
                self.add_individual(individual)
        else:
            raise ValueError("'individuals' deve ser um `int` ou uma sequência.")

    # Modifique o spawn para incluir limites de hiperparâmetros
    def spawn(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Individual:
        """Gera um novo indivíduo para a população com limites personalizados."""
        if hyperparameters is None:
            # Criar indivíduo aleatoriamente dentro dos limites fornecidos
            hyperparameters = {
                'n_latent_factors': random.randint(self.limits['n_latent_factors'][0], self.limits['n_latent_factors'][1]),
                'batch_size': random.randint(self.limits['batch_size'][0], self.limits['batch_size'][1]),
                'learning_rate': random.uniform(self.limits['learning_rate'][0], self.limits['learning_rate'][1])
            }
        return Individual(self.genes, self.handler, self.x_train, self.y_train, hyperparameters=hyperparameters, limits=self.limits, **self.kwargs)

    def add_individual(self, individual: Optional[Union[Dict[str, Any], Individual]] = None) -> None:
        """Adiciona um novo indivíduo à população."""
        if isinstance(individual, dict) or individual is None:
            self.individuals.append(self.spawn(individual))
        elif isinstance(individual, Individual):
            self.individuals.append(individual)
        else:
            raise ValueError

    def get_fittest(self, maximize: bool = False) -> Individual:
        """Retorna o indivíduo com melhor fitness na população."""
        if maximize:
            return max(self.individuals, key=operator.methodcaller("evaluate_fitness"))
        return min(self.individuals, key=operator.methodcaller("evaluate_fitness"))

    def __len__(self) -> int:
        return len(self.individuals)

    def __iter__(self):
        return iter(self.individuals)

    def __getitem__(self, item: Union[int, slice]) -> Union[Individual, Sequence[Individual]]:
        return self.individuals[item]

class Tournament:
    """
    Evolui uma população usando o método de seleção por torneio.
    """
    def __init__(self, population: Population, tournament_size: int = 5, reproduction_rate: float = 0.5, mutation_rate: float = 0.015, elitism: bool = True):
        self.population = population
        self.tournament_size = tournament_size
        self.reproduction_rate = reproduction_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism  # O melhor indivíduo é preservado

    def evolve(self, maximize: bool, generation_seed: int) -> None:
        random.seed(generation_seed)

        new_population = Population(
            genes=self.population.genes,
            handler=self.population.handler,
            individuals=[],
            x_train=self.population.x_train,
            y_train=self.population.y_train,
            **self.population.kwargs
        )

        # Aplicar elitismo: preservar os 10% melhores indivíduos
        if self.elitism:
            num_to_preserve = math.ceil(0.1 * len(self.population))
            best_individuals = sorted(self.population.individuals, key=operator.methodcaller("evaluate_fitness"), reverse=maximize)[:num_to_preserve]

            for individual in best_individuals:
                elite = individual.duplicate()
                elite.origin = "elitismo"
                new_population.add_individual(elite)

            print(f"{num_to_preserve} indivíduos preservados por elitismo.")
            print(f"Indivíduo preservado por elitismo: {elite.hyperparameters}")

        # Gerar novos indivíduos usando torneio, crossover e mutação
        while len(new_population) < len(self.population):
            parent1 = self.run_tournament(maximize)
            parent2 = self.run_tournament(maximize)

            print(f"Reproduzindo Pais (Seed {generation_seed}): {parent1.hyperparameters} e {parent2.hyperparameters}")

            child = parent1.reproduce(parent2, self.reproduction_rate)
            child.mutate(self.mutation_rate)

            print(f"Filho gerado (Seed {generation_seed}): {child.hyperparameters}, Origem: {child.origin}")

            new_population.add_individual(child)

        # Atualiza a população com a nova geração
        self.population = new_population

    def run_tournament(self, maximize: bool) -> Individual:
        """Executa o torneio e retorna o indivíduo mais apto."""
        tournament = random.sample(self.population.individuals, self.tournament_size)
        if maximize:
            return max(tournament, key=operator.methodcaller("evaluate_fitness"))
        return min(tournament, key=operator.methodcaller("evaluate_fitness"))
    
    def run(self, generations: int, maximize: bool = False, patience: Optional[int] = None, verbose: bool = True) -> None:
        """Executa o algoritmo genético por várias gerações, com uma seed específica para cada geração."""
        best_fitness = None
        patience_counter = 0  # Inicializar o contador de paciência

        for generation in range(generations):
            generation_seed = generation + 1  # Define a seed com base na geração

            if verbose:
                print(f"\nGeração {generation+1} (Seed: {generation_seed}):")

            # Evoluir a população com a seed específica para essa geração
            self.evolve(maximize, generation_seed)

            # Avaliar o indivíduo mais apto da geração
            fittest = self.population.get_fittest(maximize)
            fitness = fittest.evaluate_fitness()

            if verbose:
                print(f"Melhor fitness da geração {generation+1}: {fitness:.5f}")

            # Critério de parada baseado na paciência (opcional)
            if patience is not None:
                if best_fitness is None or (maximize and fitness > best_fitness) or (not maximize and fitness < best_fitness):
                    best_fitness = fitness
                    patience_counter = 0  # Resetar o contador de paciência quando houver melhora
                else:
                    patience_counter += 1  # Incrementar contador se não houver melhora

                if patience_counter >= patience:
                    print(f"Paciência esgotada após {patience} gerações sem melhora. Finalizando...")
                    break  # Finaliza o loop quando a paciência se esgota

# Carregar seus dados reais (100K ou 1M)
items_merged = clean_1M()  # Função que já limpa e carrega os dados

# Separar em treino e validação
train, valid = train_test_cf(items_merged)

# Converter os IDs de usuários e filmes para índices contínuos
train['user_id'] = train['user_id'].astype('category').cat.codes
train['movie_id'] = train['movie_id'].astype('category').cat.codes
valid['user_id'] = valid['user_id'].astype('category').cat.codes
valid['movie_id'] = valid['movie_id'].astype('category').cat.codes

# Definir os genes (hiperparâmetros) a serem otimizados
genes = ['n_latent_factors', 'batch_size', 'learning_rate']

# Criar os hiperparâmetros iniciais para o indivíduo
hyperparameters = {
    'n_latent_factors': 64,
    'batch_size': 128,
    'learning_rate': 1e-4
}

# Criar limites personalizados
limits = {
    'n_latent_factors': (32, 256),  # Por exemplo, limite entre 32 e 100 para n_latent_factors
    'batch_size': (64, 512),        # Por exemplo, batch_size entre 64 e 512
    'learning_rate': (1e-10, 1e-2)   # Por exemplo, learning_rate entre 1e-5 e 1e-3
}

ind = Individual(
    genes=genes,
    handler=matrix_factorization,
    x_train=train,
    y_train=valid,
    hyperparameters=hyperparameters,
    limits=limits  # Passando os limites
)

# Avaliar a fitness do indivíduo
fitness = ind.evaluate_fitness()
print(f"Fitness do indivíduo: {fitness}")

# Criar a população com limites personalizados
pop = Population(
    genes=genes,
    handler=matrix_factorization,
    individuals=10,
    x_train=train,  # Dados de treino
    y_train=valid,  # Dados de validação
)

# Evoluir com os novos limites editáveis
tournament_ga = Tournament(
    population=pop, 
    tournament_size=2, 
    reproduction_rate=0.85, 
    mutation_rate=0.15, 
    elitism=True
)

# Rodar o algoritmo genético
tournament_ga.run(generations=2, maximize=False, patience=2)

# Abrir o arquivo para salvar os resultados de todas as gerações  
with open("population_fitness_results.txt", "w", encoding='utf-8') as f:

    # Definir o número de gerações
    generations = 2  # Exemplo com 2 gerações
    # Evoluir a população (método evolve do algoritmo genético)
    for generation in range(generations):
        f.write(f"Geracao {generation+1}:\n")
        f.write("=" * 50 + "\n")

        # Evoluir a população a partir da primeira geração
        tournament_ga.evolve(maximize=False, generation_seed=generation+1)

        # Salvar os indivíduos da geração atual
        f.write(f"\nIndividuos da geracao {generation+1} (Pais e Filhos):\n")
        f.write("-" * 50 + "\n")

        for i, individual in enumerate(tournament_ga.population):
            fitness = individual.evaluate_fitness()

            # Captura as informações sobre a origem (pais, mutação ou elitismo)
            if individual.origin == "crossover":
                parent_info = f"Origem: Crossover de pais {individual.parents[0].id} e {individual.parents[1].id}"
            elif individual.origin == "mutacao":
                parent_info = f"Origem: Mutacao de pai {individual.parents[0].id}"
            elif individual.origin == "elitismo":
                parent_info = "Origem: Preservado por elitismo"
            else:
                parent_info = "Origem: Individuo inicial (sem pais)"

            # Salvar detalhes do indivíduo
            f.write(f"Individuo {i+1} (Geracao {generation+1}):\n")
            f.write(f"{parent_info}\n")  # Informação sobre a origem
            f.write(f"Hiperparametros: {individual.hyperparameters}\n")
            f.write(f"Fitness (MAE): {round(fitness, 5)}\n")
            f.write(f"Precisao: {round(individual.precision, 3)}\n")
            f.write(f"Recall: {round(individual.recall, 3)}\n")
            f.write("-" * 40 + "\n")

        # Adicionar um sumário da população inteira ao final da geração
        f.write(f"\nSumário da População - Geração {generation+1}:\n")
        f.write("-" * 50 + "\n")

        for i, individual in enumerate(tournament_ga.population):
            fitness = individual.evaluate_fitness()
            f.write(f"Individuo {i+1}: Fitness (MAE)={round(fitness, 5)}, Precisao={round(individual.precision, 3)}, Recall={round(individual.recall, 3)}\n")
            f.write(f"Hiperparametros: {individual.hyperparameters}\n")
            f.write("-" * 40 + "\n")

        # Encontrar o melhor indivíduo da geração
        best_individual = tournament_ga.population.get_fittest(maximize=False)
        f.write("\nMelhor individuo da geracao:\n")
        f.write(f"Hiperparametros: {best_individual.hyperparameters}\n")
        f.write(f"Fitness (MAE): {round(best_individual.fitness, 5)}\n")
        f.write(f"Precisao: {round(best_individual.precision, 3)}\n")
        f.write(f"Recall: {round(best_individual.recall, 3)}\n")
        f.write("=" * 50 + "\n\n")