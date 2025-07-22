import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
from load_data import load_100K, load_1M

def clean_100k():
    items_merged_100k = load_100K()

    # Occupation
    # Inicializar o OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    # Ajustar e transformar a coluna 'occupation'
    encoded_occupation = encoder.fit_transform(items_merged_100k[['occupation']])
    # Criar um DataFrame com as novas colunas
    encoded_occupation_df = pd.DataFrame(encoded_occupation, columns=encoder.get_feature_names_out(['occupation']))
    # Concatenar o novo DataFrame ao original, removendo a coluna 'occupation' original
    items_merged_100k = pd.concat([items_merged_100k, encoded_occupation_df], axis=1)

    #Release Date
    # Converter a coluna 'release date' para datetime
    items_merged_100k['release date'] = pd.to_datetime(items_merged_100k['release date'], format='%d-%b-%Y', errors='coerce')
    # Extrair o ano da 'release date'
    items_merged_100k['release_year'] = items_merged_100k['release date'].dt.year
    # Calcular a mediana da coluna 'release_year'
    median_year = items_merged_100k['release_year'].median()
    # Substituir valores nulos pela mediana
    items_merged_100k['release_year'].fillna(median_year, inplace=True)

    #Genre
    # Verificar quantos valores Unknown == 1
    unknown_mask = items_merged_100k['Unknown'] == 1
    # Transferir os 10 valores de Unknown para o genero médio Drama
    items_merged_100k.loc[unknown_mask, 'Drama'] = 1

    #Gender
    # Usar map para converter 'M' em 0 e 'F' em 1
    items_merged_100k['gender'] = items_merged_100k['gender'].map({'M': 0, 'F': 1})

    #Title
    # Função para remover o ano em parênteses do título
    def remove_year(title):
        return re.sub(r'\s\(\d{4}\)$', '', title)
    # Aplicar a função à coluna 'title'
    items_merged_100k['title'] = items_merged_100k['title'].apply(remove_year)

    #Drop    
    # Dropar as colunas que não vamos usar
    clean_merged_100k = items_merged_100k.drop(['occupation', 'release date', 'video release date', 'Unknown','IMDb URL', 'zip_code'], axis=1)

    return clean_merged_100k

def clean_1M():
    # Definir uma seed para garantir que os valores gerados sejam sempre os mesmos
    np.random.seed(42)
    #Carregar nosso Dataset
    items_merged_1M = load_1M()

    # Occupation
    # Inicializar o OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop=None)
    # Dicionário para mapear os códigos de ocupação com os nomes descritivos
    occupation_map = {
        0: "other",
        1: "academic",
        2: "artist",
        3: "clerical",
        4: "college",
        5: "customer_service",
        6: "doctor",
        7: "executive",
        8: "farmer",
        9: "homemaker",
        10: "K-12_student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales",
        15: "scientist",
        16: "self-employed",
        17: "technician",
        18: "tradesman",
        19: "unemployed",
        20: "writer"
    }

    # Mapeamento dos valores numéricos para os nomes das ocupações
    items_merged_1M['occupation_name'] = items_merged_1M['occupation'].map(occupation_map)
    # Ajustar e transformar a coluna 'occupation_name' com OneHotEncoder
    encoded_occupation = encoder.fit_transform(items_merged_1M[['occupation_name']])
    # Obter os nomes das colunas com a substituição de 'occupation_name_' por 'occupation_'
    columns_occupation = [col.replace('occupation_name_', 'occupation_') for col in encoder.get_feature_names_out(['occupation_name'])]
    # Criar um DataFrame com as novas colunas
    encoded_occupation_df = pd.DataFrame(encoded_occupation, columns=columns_occupation)
    # Concatenar as novas colunas ao DataFrame original
    items_merged_1M = pd.concat([items_merged_1M, encoded_occupation_df], axis=1)

    # Age
    # Mapeamento das faixas etárias para gerar valores aleatórios dentro das faixas
    age_ranges = {
        1: (7, 17),     # Under 18
        18: (18, 24),   # 18-24
        25: (25, 34),   # 25-34
        35: (35, 44),   # 35-44
        45: (45, 49),   # 45-49
        50: (50, 55),   # 50-55
        56: (56, 78)    # 56+
    }

    # Função para gerar uma idade aleatória dentro da faixa
    def generate_random_age(age_group):
        low, high = age_ranges[age_group]
        return np.random.randint(low, high+1)
    # Aplicando a função para substituir os valores de idade
    items_merged_1M['age'] = items_merged_1M['age'].apply(generate_random_age)

    # Gender
    # Usar map para converter 'M' em 0 e 'F' em 1
    items_merged_1M['gender'] = items_merged_1M['gender'].map({'M': 0, 'F': 1})

    # Title
    # Função para remover o ano e extraí-lo do título
    def extract_year(title):
        match = re.search(r'\(\d{4}\)', title)
        if match:
            year = match.group(0).strip('()')
            title_without_year = re.sub(r'\s\(\d{4}\)$', '', title)
            return title_without_year, year
        return title, None

        # Aplicar a função e criar duas novas colunas: título limpo e ano extraído
    items_merged_1M[['title', 'release_year']] = items_merged_1M['title'].apply(
        lambda x: pd.Series(extract_year(x)))
    
    # Genres
    # Separar os gêneros e aplicar One-Hot Encoding
    genres_dummies = items_merged_1M['genres'].str.get_dummies(sep='|')
    # Concatenar as novas colunas ao DataFrame original
    items_merged_1M = pd.concat([items_merged_1M, genres_dummies], axis=1)

    #Drop    
    # Dropar as colunas que não vamos usar
    clean_merged_1M = items_merged_1M.drop(['occupation', 'occupation_name', 'genres', 'zip_code'], axis=1)

    return clean_merged_1M