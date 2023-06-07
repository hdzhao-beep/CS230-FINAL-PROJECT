from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Bidirectional, GlobalMaxPool1D, Dense, LSTM, Conv1D, SimpleRNN, Dropout, Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the preprocessed variables from the pickle file
with open('preprocessed_data.pickle', 'rb') as f:
    X, y, X_test, y_test, df_train, df_test = pickle.load(f)
# Word2Vec training
Embedding_dimensions = 100

Word2vec_train_data = list(map(lambda x: x.split(), X_train))

word2vec_model = Word2Vec(Word2vec_train_data,
                          vector_size=Embedding_dimensions,
                          workers=8,
                          min_count=5)

# Tokenization
vocab_length = 60000
tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X)
tokenizer.num_words = vocab_length

X_train = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=input_length)
X_test = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=input_length)

# Embedding Matrix
embedding_matrix = np.zeros((vocab_length, Embedding_dimensions))

for word, token in tokenizer.word_index.items():
    if word2vec_model.wv.__contains__(word):
        embedding_matrix[token] = word2vec_model.wv.__getitem__(word)

# BiLSTM Model
def getModel():
    embedding_layer = Embedding(input_dim=vocab_length,
                                output_dim=Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    model = Sequential([
        embedding_layer,
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Bidirectional(LSTM(100, dropout=0.3, return_sequences=True)),
        Conv1D(100, 5, activation='relu'),
        GlobalMaxPool1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid'),
    ],
    name="Sentiment_Model")
    return model

# Create and compile the BiLSTM model
training_model = getModel()
training_model.summary()

# Reduce learning rate on plateau and early stopping callbacks
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
             EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=5)]

training_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the BiLSTM model
history = training_model.fit(
    X_train, y_train,
    batch_size=1024,
    epochs=10,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Plot the training and validation accuracy
acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
loss, val_loss = history.history['loss'], history.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Predicting on the Test dataset
y_pred = training_model.predict(X_test)

# Converting predictions to reflect the predicted sentiment
y_pred = np.where(y_pred >= 0.5, 1, 0)

# Printing out the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("BiLSTM Model Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# TextCNN Model
def getModel2():
    embedding_layer = Embedding(input_dim=vocab_length,
                                output_dim=Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    num_filters = 128
    kernel_sizes = [3, 4, 5]
    dropout_rate = 0.5

    model = Sequential([
        embedding_layer,
        Concatenate(),
    ])
    
    # CNN layers
    for kernel_size in kernel_sizes:
        model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
    
    model.add(GlobalMaxPool1D())
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    return model

# Create and compile the TextCNN model
training_model2 = getModel2()
training_model2.summary()
training_model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the TextCNN model
history2 = training_model2.fit(
    X_train, y_train,
    batch_size=1024,
    epochs=10,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Plot the training and validation accuracy
acc, val_acc = history2.history['accuracy'], history2.history['val_accuracy']
loss, val_loss = history2.history['loss'], history2.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Predicting on the Test dataset
y_pred = training_model2.predict(X_test)

# Converting predictions to reflect the predicted sentiment
y_pred = np.where(y_pred >= 0.5, 1, 0)

# Printing out the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("TextCNN Model Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# TEXTRNN Model
def getModel3():
    embedding_layer = Embedding(input_dim=vocab_length,
                                output_dim=Embedding_dimensions,
                                weights=[embedding_matrix],
                                input_length=input_length,
                                trainable=False)

    rnn_units = 128
    dropout_rate = 0.5

    model = Sequential([
        embedding_layer,
        SimpleRNN(units=rnn_units, dropout=dropout_rate, return_sequences=True),
        Dense(1, activation='sigmoid')
    ],
    name="TEXTRNN_Model")
    return model

# Create and compile the TEXTRNN model
training_model3 = getModel3()
training_model3.summary()
training_model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the TEXTRNN model
history3 = training_model3.fit(
    X_train, y_train,
    batch_size=1024,
    epochs=10,
    validation_split=0.1,
    callbacks=callbacks,
    verbose=1
)

# Plot the training and validation accuracy
acc, val_acc = history3.history['accuracy'], history3.history['val_accuracy']
loss, val_loss = history3.history['loss'], history3.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

# Plot the training and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Predicting on the Test dataset
y_pred = training_model3.predict(X_test)

# Converting predictions to reflect the predicted sentiment
y_pred = np.where(y_pred >= 0.5, 1, 0)

# Printing out the evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("TEXTRNN Model Evaluation Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
