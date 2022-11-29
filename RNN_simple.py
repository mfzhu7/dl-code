
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

class RNNModel(nn.Module):
    """循环神经网络模型。"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size 
        self.num_hiddens = self.rnn.hidden_size
        self.Linear = nn.Linear(self.num_hiddens, self.vocab_size)

    
    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X =  X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.Linear(Y.reshape((-1, Y.shape[-1])))
        return output, state 

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros(( self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros(( self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros(( self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device))




device = d2l.try_gpu()
num_hiddens = 256
# rnn_layer = nn.RNN(len(vocab), num_hiddens)
# rnn_layer = nn.GRU(len(vocab), num_hiddens)
rnn_layer = nn.LSTM(len(vocab), num_hiddens)
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)