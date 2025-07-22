import pandas as pd

# Load 100k DataFrame
def load_100K():
    # Carregando o csv u.data
    r_cols_100k = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('Dados/100K/u.data', sep='\t', names=r_cols_100k, encoding='latin-1')

    # Carregando o csv u.item
    i_cols_100k = ['movie_id', 'title' ,'release date','video release date', 'IMDb URL', 'Unknown', 'Action', 'Adventure',
    'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    items = pd.read_csv('Dados/100K/u.item', sep='|', names=i_cols_100k,
    encoding='latin-1')

    # Carregando o csv u.user
    u_cols_100k  = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    users = pd.read_csv('Dados/100K/u.user', sep='|', names=u_cols_100k , encoding='latin-1')

    # Merge users and ratings on user_id
    movies_users_ratings = pd.merge(users, ratings, on='user_id')
    # Merge movies_users_ratings and items on movie_id
    items_merged = pd.merge(movies_users_ratings, items, on='movie_id')

    return items_merged


# Load 1M DataFrame
def load_1M():
    # Carregando os dados de ratings
    r_cols_1M = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings1 = pd.read_table('Dados/1M/ratings.dat', sep='::', header=None, names=r_cols_1M, engine="python")

    # Carregando os dados de movies
    i_cols_1M = ['movie_id', 'title', 'genres']
    items1 = pd.read_table("Dados/1M/movies.dat", sep="::", header=None, names=i_cols_1M, engine="python", encoding='latin1')

    # Carregando os dados de users
    u_cols_1M = ['user_id', 'gender', 'age', 'occupation', 'zip_code']
    users1 = pd.read_table('Dados/1M/users.dat', sep='::', header=None, names=u_cols_1M, engine="python")

    # Merge users and ratings on user_id
    movies_users_ratings1 = pd.merge(users1, ratings1, on='user_id')
    # Merge movies_users_ratings and items on movie_id
    items_merged1 = pd.merge(movies_users_ratings1, items1, on='movie_id')

    return items_merged1