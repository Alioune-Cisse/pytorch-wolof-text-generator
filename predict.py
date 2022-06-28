import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset

def predict(dataset, model, text, next_words=100):
    model.eval()

    words = text.lower().split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = []
        for w in words[i:]:
            try:
                x.append(dataset.word_to_index[w])
            except:
                x.append(dataset.word_to_index["a"])

        # x = torch.tensor([[dataset.word_to_index[w] if w in dataset.get_uniq_words() else dataset.word_to_index['a'] for w in words[i:] ]])
        y_pred, (state_h, state_c) = model(torch.tensor([x]), (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words