from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, Conv1D, MaxPooling1D, Embedding, Input, GlobalMaxPooling1D, Convolution1D
from keras_self_attention import SeqSelfAttention


class DlModels:

    def __init__(self, categories, embed_dim, sequence_length):

        self.categories = categories
        self.embed_dim = embed_dim
        self.sequence_length = sequence_length

    def brnn_complex(self, char_index):

        model = Sequential()
        voc_size = len(char_index.keys())
        print("voc_size: {}".format(voc_size))
        model.add(Embedding(voc_size + 1, self.embed_dim))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        model.add(Bidirectional(LSTM(128)))

        model.add(Dense(len(self.categories), activation='sigmoid'))

        return model
