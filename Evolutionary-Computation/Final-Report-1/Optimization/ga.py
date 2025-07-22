import random
import pprint
from typing import Any, Dict, Sequence,  Union, Type, Optional
import pandas as pd
from keras.optimizers import Adam # type: ignore
from models import matrix_factorization
from clean_data import clean_100k, clean_1M
from pre_processing import train_test_cf
from models_test import matrix_factorization, calculate_precision_recall
from sklearn.metrics import precision_score, recall_score
import operator

random.seed(42)

class Individual:
    """
    Representa um membro da população com valores de genes (hiperparâmetros) para o modelo.
    """

    def __init__(self, genes: Sequence, handler, x_train, y_train, hyperparameters: Dict[str, Any], **kwargs: Any):
        self.genes = genes  # Os hiperparâmetros (genes) do indivíduo
        self.handler = handler  # Função que instancia e treina o modelo
        self.x_train = x_train  # Dados de treino
        self.y_train = y_train  # Labels de treino
        self.hyperparameters = hyperparameters  # Hiperparâmetros do indivíduo
        self.kwargs = kwargs  # Outros argumentos opcionais para o modelo
        self.fitness = None  # Fitness inicialmente desconhecida
        self.precision = None  # Precisão do indivíduo
        self.recall = None  # Recall do indivíduo
        
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

    def reproduce(self, partner: 'Individual', rate: float = 1.0) -> 'Individual':
        """Mistura os genes de dois indivíduos e retorna um novo indivíduo (filho)."""
        child_hyperparameters = {}
        for param, value in self.hyperparameters.items():
            if random.random() < rate:
                child_hyperparameters[param] = partner.hyperparameters[param]
            else:
                child_hyperparameters[param] = value
        return Individual(self.genes, self.handler, self.x_train, self.y_train, hyperparameters=child_hyperparameters, **self.kwargs)

    def crossover(self, partner: 'Individual', rate: float = 1.0) -> None:
        """Troca genes entre dois indivíduos."""
        for param, value in self.hyperparameters.items():
            if random.random() < rate:
                partner_value = partner.hyperparameters[param]
                partner.hyperparameters[param] = value
                self.hyperparameters[param] = partner_value

    def mutate(self, rate: float = 1.0) -> None:
        """Aplica mutações aos genes do indivíduo."""
        for gene in self.genes:
            self.hyperparameters[gene] = random.uniform(self.hyperparameters[gene] * (1 - rate), self.hyperparameters[gene] * (1 + rate))

    def duplicate(self) -> 'Individual':
        """Cria uma cópia do indivíduo."""
        return Individual(self.genes, self.handler, self.x_train, self.y_train, hyperparameters=self.hyperparameters.copy(), **self.kwargs)

    def __str__(self):
        """Retorna os hiperparâmetros que identificam o indivíduo."""
        return pprint.pformat(self.hyperparameters)

class Population:
    """
    Uma coleção de indivíduos (representados como soluções com hiperparâmetros).
    Pode ser inicializada com uma sequência de indivíduos ou especificando um tamanho de população,
    onde indivíduos aleatórios serão gerados.
    """

    def __init__(
        self,
        genes: Sequence[str],
        handler: Any,  # A função para o modelo (nesse caso, `matrix_factorization`)
        individuals: Union[Sequence[Dict[str, Any]], Sequence[Individual], int],
        x_train: Any = None,
        y_train: Any = None,
        **kwargs,
    ):
        self.genes = genes
        self.handler = handler
        self.x_train = x_train
        self.y_train = y_train
        self.kwargs = kwargs  # Outros parâmetros opcionais para o modelo
        
        # Criar indivíduos (aleatoriamente ou a partir de uma lista)
        if isinstance(individuals, int):
            self.individuals = [self.spawn() for _ in range(individuals)]
        elif isinstance(individuals, Sequence):
            self.individuals = []
            for individual in individuals:
                self.add_individual(individual)
        else:
            raise ValueError("'individuals' deve ser um `int` ou uma sequência.")

    def spawn(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Individual:
        """Gera um novo indivíduo para a população."""
        if hyperparameters is None:
            # Criar indivíduo aleatoriamente
            hyperparameters = {
                'n_latent_factors': random.randint(32, 128),
                'batch_size': random.randint(32, 256),
                'learning_rate': random.uniform(1e-4, 1e-2)
            }
        return Individual(
            self.genes, self.handler, self.x_train, self.y_train, hyperparameters=hyperparameters, **self.kwargs
        )

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

# Carregar seus dados reais (100K ou 1M)
items_merged = clean_100k()  # Função que já limpa e carrega os dados

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

# Criar o indivíduo
ind = Individual(
    genes=genes,
    handler=matrix_factorization,
    x_train=train,  # Dados de treino reais
    y_train=valid,  # Dados de validação reais
    hyperparameters=hyperparameters
)

# Avaliar a fitness do indivíduo
fitness = ind.evaluate_fitness()
print(f"Fitness do indivíduo: {fitness}")

# Definir o número de indivíduos na população
population_size = 10

# Criar uma população com 10 indivíduos
pop = Population(
    genes=genes,
    handler=matrix_factorization,
    individuals=population_size,
    x_train=train,  # Dados de treino
    y_train=valid,  # Dados de validação
)

# Abrir o arquivo .txt para salvar os resultados
with open("population_fitness_results.txt", "w") as f:
    # Salvar o indivíduo padrão com valores pré-definidos
    f.write("Individuo Padrao:\n")
    f.write(f"Hiperparametros: {pprint.pformat(ind.hyperparameters)}\n")
    f.write(f"Fitness (MAE) = {round(fitness, 5)}\n")
    f.write(f"Precisao = {round(ind.precision, 3)}\n")  # Usando o atributo do indivíduo
    f.write(f"Recall = {round(ind.recall, 3)}\n")  # Usando o atributo do indivíduo
    f.write("-" * 40 + "\n")

    # Agora gerar a população
    # Avaliar a fitness de cada indivíduo na população
    for i, individual in enumerate(pop):
        fitness = individual.evaluate_fitness()

        # Escrever os parâmetros, fitness, precisão e recall no arquivo
        f.write(f"Individuo {i+1}:\n")
        f.write(f"Hiperparametros: {pprint.pformat(individual.hyperparameters)}\n")
        f.write(f"Fitness (MAE) = {round(fitness, 5)}\n")
        f.write(f"Precisao = {round(individual.precision, 3)}\n")  # Usando o atributo do indivíduo
        f.write(f"Recall = {round(individual.recall, 3)}\n")  # Usando o atributo do indivíduo
        f.write("-" * 40 + "\n")

    # Encontrar o indivíduo com a melhor fitness (minimizar val_loss)
    best_individual = pop.get_fittest(maximize=False)
    
    f.write("\nIndividuo com o melhor fitness:\n")
    f.write(f"ID do Melhor Individuo: {pop.individuals.index(best_individual) + 1}\n")
    f.write(f"Hiperparametros: {pprint.pformat(best_individual.hyperparameters)}\n")
    f.write(f"Melhor Fitness (MAE) = {round(best_individual.fitness, 5)}\n")
    f.write(f"Precisao = {round(best_individual.precision, 3)}\n")
    f.write(f"Recall = {round(best_individual.recall, 3)}\n")

print("Resultados salvos no arquivo 'population_fitness_results.txt'.")