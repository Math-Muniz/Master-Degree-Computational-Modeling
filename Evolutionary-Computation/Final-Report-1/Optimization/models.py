# Bibliotecas
import numpy as np
from sklearn.metrics import precision_score, recall_score
import keras
from keras.optimizers import Adam, SGD  # type: ignore
from keras.layers import Dense, Dropout, Input, Embedding, Flatten, Dot  # type: ignore
from keras.regularizers import l2 # type: ignore
from keras.models import Model  # type: ignore

# Low Rank Matrix Factorization
def matrix_factorization(
    df,
    train,
    valid,
    n_latent_factors=64,
    batch_size=128,
    epochs=10,
    optimizer_fn=Adam,
    learning_rate=1e-4,
    loss="mae",
    verbose=1,
):

    # Obter o número de usuários e filmes únicos
    n_movies = len(df["movie_id"].unique())
    n_users = len(df["user_id"].unique())

    # Entrada para usuários e filmes
    user_input = Input(shape=(1,), name="user_input", dtype="int64")
    movie_input = Input(shape=(1,), name="movie_input", dtype="int64")

    # Criação dos embeddings para usuários e filmes
    user_embedding = Embedding(
        input_dim=n_users, output_dim=n_latent_factors, name="user_embedding"
    )(user_input)
    movie_embedding = Embedding(
        input_dim=n_movies, output_dim=n_latent_factors, name="movie_embedding"
    )(movie_input)

    # Achatar os embeddings
    user_vec = Flatten(name="FlattenUsers")(user_embedding)
    movie_vec = Flatten(name="FlattenMovies")(movie_embedding)

    # Produto escalar entre os vetores de usuário e filme
    sim = Dot(axes=1, name="Similarity-Dot-Product")([user_vec, movie_vec])

    # Criar o modelo Keras
    model = keras.models.Model([user_input, movie_input], sim)

    # Exibir o resumo do modelo
    model.summary()

    # Criar um novo otimizador sempre que o modelo for compilado
    optimizer = optimizer_fn(learning_rate=learning_rate)

    # Compilar o modelo
    model.compile(optimizer=optimizer, loss=loss)

    # Treinar o modelo
    history = model.fit(
        [train.user_id, train.movie_id],
        train.rating,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([valid.user_id, valid.movie_id], valid.rating),
        verbose=verbose,

    )

    return history

# Neural Network
def neural_network(
    df,
    train,
    valid,
    n_latent_factors=50,
    batch_size=128,
    epochs=10,
    optimizer_fn=Adam,
    learning_rate=1e-4,
    loss="mae",
    activation_initial="relu",  # camadas intermediarias
    activation_final="linear",  # camada de saida
    dropout=0.40,
    verbose=1,
):

    # Obter o número de usuários e filmes únicos
    n_movies = len(df["movie_id"].unique())
    n_users = len(df["user_id"].unique())

    user_input = Input(shape=(1,), name="user_input", dtype="int64")
    user_embedding = Embedding(n_users, n_latent_factors, name="user_embedding")(
        user_input
    )
    user_vec = Flatten(name="FlattenUsers")(user_embedding)
    user_vec = Dropout(dropout)(user_vec)

    movie_input = Input(shape=(1,), name="movie_input", dtype="int64")
    movie_embedding = Embedding(n_movies, n_latent_factors, name="movie_embedding")(
        movie_input
    )
    movie_vec = Flatten(name="FlattenMovies")(movie_embedding)
    movie_vec = Dropout(dropout)(movie_vec)

    # Produto escalar entre os vetores de usuário e filme
    sim = Dot(axes=1, name="Similarity-Dot-Product")([user_vec, movie_vec])

    nn_inp = Dense(96, activation=activation_initial)(sim)
    nn_inp = Dropout(dropout)(nn_inp)
    nn_inp = Dense(1, activation=activation_final)(nn_inp)
    nn_model = keras.models.Model([user_input, movie_input], nn_inp)
    nn_model.summary()

    # Criar um novo otimizador sempre que o modelo for compilado
    optimizer = optimizer_fn(learning_rate=learning_rate)

    # Compilar o modelo
    nn_model.compile(optimizer=optimizer, loss=loss)

    # Treinar o modelo
    nn_history = nn_model.fit(
        [train.user_id, train.movie_id],
        train.rating,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([valid.user_id, valid.movie_id], valid.rating),
        verbose=verbose,
    )

    return nn_history

# Função para calcular precision e recall
def calculate_precision_recall(model, X_val_user, X_val_movie, y_val, threshold=3.0):
    # Previsões do modelo
    predictions = model.predict([X_val_user, X_val_movie]).flatten()
    
    # Binarizar previsões e rótulos com base no threshold ajustado
    y_pred_binary = (predictions >= threshold).astype(int)
    y_true_binary = (y_val >= threshold).astype(int)
    
    precision = precision_score(y_true_binary, y_pred_binary)
    recall = recall_score(y_true_binary, y_pred_binary)
    
    return precision, recall

# Função para gravar os melhores resultados em um arquivo .txt
def save_best_results(history, precision, recall, filename="best_results.txt"):
    # Encontrar a melhor época baseada na menor perda
    best_epoch = np.argmin(history.history['val_loss'])
    best_loss = history.history['val_loss'][best_epoch]
    
    # Gravar os resultados em um arquivo .txt
    with open(filename, "w") as f:
        f.write(f"Quantidade de interacoes: {best_epoch + 1}\n")
        f.write(f"(MAE): {round(best_loss, 2)}\n")
        f.write(f"Precisao: {round(precision, 2)}\n")
        f.write(f"Recall: {round(recall, 2)}\n")