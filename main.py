from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import os

import tensorflow as tf 
import numpy as np
import re
import pickle
from pathlib import Path
with open(Path("data") / "word_list.txt", "rb") as f:
    word_list = pickle.load(f)

# define constants here
max_length = 20
batch_size = 32
vocab_size = len(word_list)
embedding_dim = 256
hidden_units = 1024
attention_units = 16
epochs = 32

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
    
    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

encoder = Encoder(vocab_size, embedding_dim, hidden_units, batch_size)
decoder = Decoder(vocab_size, embedding_dim, hidden_units, batch_size)
optimizer = tf.optimizers.Adam()

checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

checkpoint.restore(str(Path("data") / "model"))

def preprocess_sentence(msg):
    msg = msg.replace('\n', ' ').lower()
    msg = msg.replace("\xc2\xa0", "")
    msg = re.sub('([\(\).,!?])', "", msg)
    msg = re.sub(" +"," ", msg)
    return msg

def softmax_choose(a, weights):
    exps = np.exp(weights - np.max(weights))
    scaled_exps = exps / np.sum(exps)
    return np.random.choice(a, p=scaled_exps)

def encode(sentence):
    inputs = [word_list.index(i) for i in sentence.split(' ') if i in word_list]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    return inputs

def predict(inputs):
    result = []

    hidden = [tf.zeros((1, hidden_units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([word_list.index('<padding>')], 0)

    for t in range(max_length):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        options = 8 if t == 0 else 1
        top_prediction_indices = tf.argsort(predictions[0])[::-1][:options]
        top_confidences = []
        for index in top_prediction_indices:
            top_confidences.append(float(predictions[0][index]))
        top_confidences = np.array(top_confidences) 
        predicted_id = softmax_choose(top_prediction_indices, top_confidences)
        
        result.append(predicted_id)
        if word_list[predicted_id] == '<eos>':
            return result

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result

def decode(outputs):
    return " ".join([word_list[w] for w in outputs if word_list[w] != "<eos>" and word_list[w] != "<padding>"])

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)
    inputs = encode(sentence)
    outputs = predict(inputs)
    response = decode(outputs)

    return sentence, response

app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

@app.route("/Chat", methods=["POST"])
@cross_origin()
def chat():
    if not request.json:
        return jsonify({"message": "ERR: must send JSON"})
    if 'msg' not in request.json.keys():
        return jsonify({"message": "ERR: field msg must be in JSON"})
    response = evaluate(str(request.json["msg"]))[1]
    return jsonify({"data": response})

@app.route("/")
def hello():
    return "Hello world"

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    app.run(host="127.0.0.1", port=5000, debug=True)
    #http_server = WSGIServer(('', int(os.environ["PORT"])), app)
    #http_server = WSGIServer(('', 8080), app)
    #http_server.serve_forever()