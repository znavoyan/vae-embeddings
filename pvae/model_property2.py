import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
from params import *


def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.94 ** (epoch // 1))
    #print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer,lr 
            

def to_var(x,device_id,gpu_exist, volatile=False):
    
    if gpu_exist == True:
        x = x.cuda(device_id)
    else:
        x = x
    x = Variable(x, volatile= volatile)
       
    return x

def batch2tensor(batch,Args):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = to_var(v,Args['device_id'],Args['gpu_exist'])
        if k == 'Effect':
            batch[k] = to_var(v.type(torch.FloatTensor) ,Args['device_id'])
    return batch
    


class SentenceVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, max_sequence_length,device_id, num_layers=1, 
                predict_prop = False, nr_prop = None, bidirectional=False, gpu_exist = True):

        super().__init__()
        #self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.gpu_exist  =  gpu_exist
        self.device_id = device_id

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.predict_prop = predict_prop
        self.hidden_size = hidden_size
        

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout = nn.Dropout(p=word_dropout)
        #self.prediction = nn.Linear(latent_size,nr_classes)
        #if self.predict_prop:
        self.prediction = nn.Linear(latent_size, nr_prop)
        
        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, input_sequence, length):

        batch_size = input_sequence.size(0)
        # print('batch size: ', batch_size)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        # print('input_sequence: ', input_sequence.shape)
        #print('length:', length)
        #print('sorted_lengths:', sorted_lengths)
        #print('sorted_idx:', sorted_idx)
        #print('input_sequence:', input_sequence)

        # ENCODER
        input_embedding = self.embedding(input_sequence)
        # print('input_embedding: ', input_embedding.shape)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)
        # print('hidden: ', hidden.shape)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # print('hidden: ', hidden.shape)

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        #z = to_var(torch.randn([batch_size, self.latent_size]))
        z = to_var(torch.normal(mean=torch.zeros([batch_size, self.latent_size]), std=0.01*torch.ones([batch_size, self.latent_size])),self.device_id,self.gpu_exist)
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        input_embedding = self.word_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
       
        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        #print('reversed_idx:', reversed_idx)

        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)
        z = z[reversed_idx]

        # print('mean: ', mean.shape)
        # print(reversed_idx)

        # if batch_size == 1:
        #     return mean
        # else:
        #     mean = mean[reversed_idx] 
        #     return mean

        if batch_size > 1:
            mean = mean[reversed_idx]
            logv = logv[reversed_idx]

        # mean = mean[reversed_idx] 
        # print('returning mean: ', mean)
        
        # logv = logv[reversed_idx] 
        #prediction = 0
        if self.predict_prop:
            prediction = self.prediction(z)
            return logp, mean, logv, z, prediction
        else:
            return logp, mean, logv, z

    def inference(self, z=None):
        pad_idx = 0
        sos_idx = self.sos_idx
        eos_idx = self.eos_idx
        input_sequence = sos_idx
        batch_size = z.size(0)
        generation = []
        hidden = self.latent2hidden(z)
        #print('hidden >>>> ', hidden.size())
        t = 0
        hidden = hidden.unsqueeze(0)
        while (t < self.max_sequence_length and input_sequence != eos_idx):
            #hidden = hidden.unsqueeze(0)
            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(sos_idx).long(), self.device_id,self.gpu_exist)
                #input_sequence = torch.Tensor(batch_size).fill_(sos_idx).long()
            #print(input_sequence.shape)
            input_sequence = input_sequence.unsqueeze(1)
            
            input_embedding = self.embedding(input_sequence)
            
            output, hidden = self.decoder_rnn(input_embedding, hidden)
            hidden = hidden
          
            logits = self.outputs2vocab(output)
           
            input_sequence = self._sample(logits)
            input_sequence =  input_sequence.unsqueeze(0)
           
            generation.append(input_sequence[0].cpu().tolist())

            t = t + 1
        
        return generation#input_sequence#generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to


