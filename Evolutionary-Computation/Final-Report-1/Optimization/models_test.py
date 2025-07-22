# Bibliotecas
from sklearn.metrics import precision_score, recall_score
from keras.optimizers import Adam  # type: ignore
from keras.layers import Dense, Dropout, Input, Embedding, Flatten, Dot  # type: ignore
from keras.models import Model  # type: ignore
from sklearn.metrics import precision_score, recall_score

def matrix_factorization(df,
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

    # Converter os IDs de usuários e filmes para índices numéricos
    train['user_id'] = train['user_id'].astype('category').cat.codes
    train['movie_id'] = train['movie_id'].astype('category').cat.codes
    valid['user_id'] = valid['user_id'].astype('category').cat.codes
    valid['movie_id'] = valid['movie_id'].astype('category').cat.codes

    # Criar o modelo de matrix factorization
    user_input = Input(shape=(1,), name="user_input", dtype="int64")
    movie_input = Input(shape=(1,), name="movie_input", dtype="int64")

    # Embeddings para usuários e filmes
    n_users = len(train['user_id'].unique())
    n_movies = len(train['movie_id'].unique())

    user_embedding = Embedding(input_dim=n_users, output_dim=n_latent_factors)(user_input)
    movie_embedding = Embedding(input_dim=n_movies, output_dim=n_latent_factors)(movie_input)

    user_vec = Flatten()(user_embedding)
    movie_vec = Flatten()(movie_embedding)

    # Produto escalar entre os vetores de usuário e filme
    dot_product = Dot(axes=1, name="Similarity-Dot-Product")([user_vec, movie_vec])
    # Criar o modelo Keras
    model = Model(inputs=[user_input, movie_input], outputs=dot_product)

    # Compilar o modelo
    optimizer = optimizer_fn(learning_rate=learning_rate)
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

    # Retornar o histórico e o modelo treinado
    return model, history

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

    # Converter os IDs de usuários e filmes para índices numéricos
    train['user_id'] = train['user_id'].astype('category').cat.codes
    train['movie_id'] = train['movie_id'].astype('category').cat.codes
    valid['user_id'] = valid['user_id'].astype('category').cat.codes
    valid['movie_id'] = valid['movie_id'].astype('category').cat.codes

    # Embeddings para usuários e filmes
    n_users = len(train['user_id'].unique())
    n_movies = len(train['movie_id'].unique())

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
    dot_product = Dot(axes=1, name="Similarity-Dot-Product")([user_vec, movie_vec])

    nn_inp = Dense(96, activation=activation_initial)(dot_product)
    nn_inp = Dropout(dropout)(nn_inp)
    nn_inp = Dense(1, activation=activation_final)(nn_inp)
    nn_model = Model(inputs=[user_input, movie_input], outputs=nn_inp)
    nn_model.summary()

    # Criar um novo otimizador sempre que o modelo for compilado
    optimizer = optimizer_fn(learning_rate=learning_rate)

    # Compilar o modelo
    optimizer = optimizer_fn(learning_rate=learning_rate)
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

    # Retornar o histórico e o modelo treinado
    return nn_model, nn_history

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