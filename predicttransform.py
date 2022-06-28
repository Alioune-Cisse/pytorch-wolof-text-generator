import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from position import PositionalEncoding
from mask import generate_square_subsequent_mask
import numpy as np

def predict(dataset, model, text, next_words=100):
    # model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_mask = generate_square_subsequent_mask(1).to(device)

    words = text.lower().split(' ')
    #words_d = list([w if w in dataset.get_uniq_words() else "a" for w in words])
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):

        # x = torch.tensor([[ dataset.word_to_index[w]  for w in words[i:] ]]).to(torch.long)
        x = []
        for w in words[i:]:
            try:
                x.append(dataset.word_to_index[w])
            except:
                x.append(dataset.word_to_index["a"])
        # print(x)
        # y_pred, (state_h, state_c) = model(x, (state_h, state_c))
        y_pred = model(torch.tensor([x]), src_mask)
        # print(y_pred)

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])

    return words