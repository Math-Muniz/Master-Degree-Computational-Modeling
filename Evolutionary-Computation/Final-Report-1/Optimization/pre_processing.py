# Bibliotecas
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


# Matriz com pivot_table
def matrix_pivot(df):
    # Convertendo os IDs para categorias numéricas
    df.user_id = df.user_id.astype('category').cat.codes.values
    df.movie_id = df.movie_id.astype('category').cat.codes.values

    # Criando nossa matriz de utilidade
    index = list(df['user_id'].unique())
    columns = list(df['movie_id'].unique())
    index = sorted(index)
    columns = sorted(columns)
    # Criando a matriz de utilidade com a função pivot_table
    utility_matrix = pd.pivot_table(
        data = df,
        values = 'rating',  # O valor da matriz será a coluna 'rating'
        index = 'user_id',  # Índices serão os IDs dos usuários
        columns = 'movie_id',  # Colunas serão os IDs dos filmes
        fill_value = 0  # Preenche os valores faltantes com 0 (caso o usuário não tenha avaliado o filme)
    )
        
    print(utility_matrix)

# Matriz com csr_matrix
def matrix_csr(df):
    # Convertendo os IDs para categorias numéricas
    df.user_id = df.user_id.astype('category').cat.codes.values
    df.movie_id = df.movie_id.astype('category').cat.codes.values

    # Obtendo o número de usuários e filmes únicos
    N = df['user_id'].nunique()
    M = df['movie_id'].nunique()

    # Criando a matriz esparsa CSR
    # As linhas representam os usuários (user_id), e as colunas representam os filmes (movie_id)
    utility_matrix = csr_matrix(
        (df['rating'], 
        (df['user_id'], df['movie_id'])),
        shape=(N, M))

    # Informações da matriz criada
    n_total = utility_matrix.shape[0] * utility_matrix.shape[1]  # Total de elementos possíveis na matriz
    n_ratings = utility_matrix.nnz  # Número de elementos não nulos
    sparsity = n_ratings / n_total  # Proporção de elementos não nulos
    print(f"Esparsidade da Matriz: {round(sparsity * 100, 2)}%")
    print("Formato da Matriz:", utility_matrix.shape)
    print("Número de elementos não nulos na matriz:", utility_matrix.nnz)
    
    # Opcional: Converter para um formato denso para visualização (somente para debug, cuidado com memória)
    utility_dense = utility_matrix.todense()
    print(utility_dense)
    return utility_matrix

# Creating Training and Validation Sets
def train_test_cf(df):
    """
    Cria os conjuntos de treino e validação para matrix factorization, 
    convertendo IDs para índices numéricos usando astype('category').cat.codes.
    
    Args:
        df (pd.DataFrame): DataFrame contendo os dados
    Returns:
        train (pd.DataFrame): Conjunto de treino
        valid (pd.DataFrame): Conjunto de validação
    """
    # Convertendo os IDs de usuário e filme para índices numéricos
    df['user_id'] = df['user_id'].astype('category').cat.codes
    df['movie_id'] = df['movie_id'].astype('category').cat.codes

    # Mantendo apenas as colunas 'user_id', 'movie_id', 'rating', 'unix_timestamp'
    items_filtered = df[['user_id', 'movie_id', 'rating', 'unix_timestamp']]

    # Dividindo em 80% treino e 20% validação
    train, valid = train_test_split(items_filtered, test_size=0.2, random_state=42)
    
    # Informações sobre o conjunto de dados
    print(f"Tamanho do conjunto de treino: {train.shape}")
    print(f"Tamanho do conjunto de teste: {valid.shape}")
    
    return train, valid