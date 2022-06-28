from flask import Flask, render_template, request
import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
from predict import predict
from modeltransform import TransformerModel
from predicttransform import predict as predicttransform

parser = argparse.ArgumentParser()
parser.add_argument('--max-epochs', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--sequence-length', type=int, default=4)

#args = parser.parse_args(args=[])
args = parser.parse_args()

dataset = Dataset(args)
ntokens = len(dataset.uniq_words)
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 4  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 4  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(dataset)
model_transform = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)




dictionnaires = dataset.get_alphabet()

#train(dataset, model, args)

#model = Model(dataset)
model.load_state_dict(torch.load("our-model.pth"))
model.eval()

model_transform.load_state_dict(torch.load("model-generator.pth"))
model_transform.eval()

couleurs = ["#8dd3c7",
            "#ffffb3",
            "#bebada",
            "#fb8072",
            "#80b1d3",
            "#fdb462",
            "#b3de69",
            "#fccde5",
            "#d9d9d9",
            "#bc80bd",
            "#ccebc5",
            "#ffed6f"]



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/transform')
def transform():
    return render_template('transformer.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/araf')
def araf():
    return render_template('araf.html', data=dictionnaires, zip=zip, np=np, couleurs=couleurs)

@app.route('/research', methods=['GET', 'POST'])
def research():
    if request.method == 'POST':
        if 'search' in request.form:
            text = request.form['search']
            nw = request.form['size']
            prediction = predict(dataset, model, text=text, next_words=int(nw))
            return render_template('home.html',  data=" ".join(prediction).capitalize())

    else:
        return render_template('home.html')

@app.route('/research_t', methods=['GET', 'POST'])
def research_t():
    if request.method == 'POST':
        if 'submit_t' in request.form:
            text = request.form['search_t']
            nw = request.form['size_t']
            prediction = predicttransform(dataset, model_transform, text=text, next_words=int(nw))
            return render_template('transformer.html',  data_t=" ".join(prediction).capitalize())

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True, port=3000)















