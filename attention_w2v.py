#! /usr/bin/env python
# coding=utf-8

from keras.layers import Input, Dense, Multiply, Embedding
from keras.models import Model


class Attw2v:
    def __init__(self, dim, hidden_size=200):
        self.dim = dim
        self.hidden_size = hidden_size
        self.model = self._build_model()

    def _build_model(self):
        input = Input(shape=(self.dim,))
        # have some questions in Embedding, especially the input_length
        vec = Embedding(output_dim=self.hidden_size, embeddings_initializer='glorot_uniform', input_length=1)(input)

        attention_probs = Dense(self.dim, activation='softmax')(vec)
        attention_mul = Multiply()([vec, attention_probs])
        # questions
        attention_mul = Dense(self.hidden_size, activation='relu')(attention_mul)

        output = Dense(self.dim, activation='softmax')(attention_mul)
        model = Model(input=input, output=output)

        return model
