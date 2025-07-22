from clean_data import clean_100k, clean_1M
from pre_processing import train_test_cf
from models import matrix_factorization, calculate_precision_recall, save_best_results, neural_network

if __name__ == "__main__":
    # Carregar e processar os dados 100k
    print("Resultados para o dataset de 100K:")
    items_merged = clean_100k()
    # Dividir os dados em treino e validação
    train, valid = train_test_cf(items_merged)
    
    # Treinar o modelo Matrix Factorization para 100K
    print("Resultados para a Matrix Factorization 100K:")
    history = matrix_factorization(items_merged, train, valid)  # Mantendo como está
    model = history.model  # Obtendo o modelo a partir do histórico
    # Calcular precision e recall
    precision, recall = calculate_precision_recall(model, valid.user_id, valid.movie_id, valid.rating)
    # Gravar os melhores resultados
    save_best_results(history, precision, recall, "best_results_100k_matrix.txt")

    # Treinar o modelo Neural Network para 100K
    print("Resultados para a Neural Network 100K:")
    history_nn = neural_network(items_merged, train, valid)  # Mantendo como está
    model_nn = history_nn.model  # Obtendo o modelo a partir do histórico
    # Calcular precision e recall
    precision_nn, recall_nn = calculate_precision_recall(model_nn, valid.user_id, valid.movie_id, valid.rating)
    # Gravar os melhores resultados
    save_best_results(history_nn, precision_nn, recall_nn, "best_results_100k_nn.txt")

    # Carregar e processar os dados 1M
    print("Resultados para o dataset de 1M:")
    items_merged1 = clean_1M()
    # Dividir os dados em treino e validação
    train1, valid1 = train_test_cf(items_merged1)

    # Treinar o modelo Matrix Factorization para 1M
    print("Resultados para a Matrix Factorization 1M:")
    history1 = matrix_factorization(items_merged1, train1, valid1)  # Mantendo como está
    model1 = history1.model  # Obtendo o modelo a partir do histórico
    # Calcular precision e recall
    precision1, recall1 = calculate_precision_recall(model1, valid1.user_id, valid1.movie_id, valid1.rating)
    # Gravar os melhores resultados
    save_best_results(history1, precision1, recall1, "best_results_1m_matrix.txt")

    # Treinar o modelo Neural Network para 1M
    print("Resultados para a Neural Network 1M:")
    history_nn1 = neural_network(items_merged1, train1, valid1)  # Mantendo como está
    model_nn1 = history_nn1.model  # Obtendo o modelo a partir do histórico
    # Calcular precision e recall
    precision_nn1, recall_nn1 = calculate_precision_recall(model_nn1, valid1.user_id, valid1.movie_id, valid1.rating)
    # Gravar os melhores resultados
    save_best_results(history_nn1, precision_nn1, recall_nn1, "best_results_1m_nn.txt")
