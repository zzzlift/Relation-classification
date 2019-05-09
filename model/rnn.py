"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from utils import constant, torch_utils
from model import layers

class RelationModel(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.model = PositionAwareRNN(opt, emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
    
    def update(self, batch):
        """ Run a step of forward and backward model update. """
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict(self, batch, unsort=True):
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = [b.cuda() for b in batch[:7]]
            labels = batch[7].cuda()
        else:
            inputs = [b for b in batch[:7]]
            labels = batch[7]

        orig_idx = batch[8]

        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, dim=1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.data.item()

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                'epoch': epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

# class PositionAwareRNN(nn.Module):
#     """ A sequence model for relation extraction. """

#     def __init__(self, opt, emb_matrix=None):
#         super(PositionAwareRNN, self).__init__()
#         self.drop = nn.Dropout(opt['dropout'])
#         self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
#         if opt['pos_dim'] > 0:
#             self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
#                     padding_idx=constant.PAD_ID)
#         if opt['ner_dim'] > 0:
#             self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
#                     padding_idx=constant.PAD_ID)
        
#         input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
#         self.rnn = nn.LSTM(input_size, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
#                 dropout=opt['dropout'])
#         self.linear = nn.Linear(opt['hidden_dim'], opt['num_class'])

#         if opt['attn']:
#             self.attn_layer = layers.PositionAwareAttention(opt['hidden_dim'],
#                     opt['hidden_dim'], 2*opt['pe_dim'], opt['attn_dim'])
#             self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])

#         self.opt = opt
#         self.topn = self.opt.get('topn', 1e10)
#         self.use_cuda = opt['cuda']
#         self.emb_matrix = emb_matrix
#         self.init_weights()
    
#     def init_weights(self):
#         if self.emb_matrix is None:
#             self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
#         else:
#             self.emb_matrix = torch.from_numpy(self.emb_matrix)
#             self.emb.weight.data.copy_(self.emb_matrix)
#         if self.opt['pos_dim'] > 0:
#             self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
#         if self.opt['ner_dim'] > 0:
#             self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)

#         self.linear.bias.data.fill_(0)
#         init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
#         if self.opt['attn']:
#             self.pe_emb.weight.data.uniform_(-1.0, 1.0)

#         # decide finetuning
#         if self.topn <= 0:
#             print("Do not finetune word embedding layer.")
#             self.emb.weight.requires_grad = False
#         elif self.topn < self.opt['vocab_size']:
#             print("Finetune top {} word embeddings.".format(self.topn))
#             self.emb.weight.register_hook(lambda x: \
#                     torch_utils.keep_partial_grad(x, self.topn))
#         else:
#             print("Finetune all embeddings.")

#     def zero_state(self, batch_size): 
#         state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
#         h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
#         if self.use_cuda:
#             return h0.cuda(), c0.cuda()
#         else:
#             return h0, c0
    
#     def forward(self, inputs):
#         words, masks, pos, ner, deprel, subj_pos, obj_pos = inputs # unpack
#         seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
#         batch_size = words.size()[0]
        
#         # embedding lookup
#         word_inputs = self.emb(words)
#         inputs = [word_inputs]
#         if self.opt['pos_dim'] > 0:
#             inputs += [self.pos_emb(pos)]
#         if self.opt['ner_dim'] > 0:
#             inputs += [self.ner_emb(ner)]
#         inputs = self.drop(torch.cat(inputs, dim=2)) # add dropout to input
#         input_size = inputs.size(2)
        
#         # rnn
#         h0, c0 = self.zero_state(batch_size)
#         inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
#         outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
#         outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
#         hidden = self.drop(ht[-1,:,:]) # get the outmost layer h_n
#         outputs = self.drop(outputs)
        
#         # attention
#         if self.opt['attn']:
#             # convert all negative PE numbers to positive indices
#             # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
#             subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
#             obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
#             pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)
#             final_hidden = self.attn_layer(outputs, masks, hidden, pe_features)
#         else:
#             final_hidden = hidden

#         logits = self.linear(final_hidden)
#         return logits, final_hidden


class PositionAwareRNN(nn.Module):
    """ A sequence model for relation extraction. """

    def __init__(self, opt, emb_matrix=None):
        super(PositionAwareRNN, self).__init__()
        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        if opt['pos_dim'] > 0:
            self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim'],
                    padding_idx=constant.PAD_ID)
        if opt['ner_dim'] > 0:
            self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim'],
                    padding_idx=constant.PAD_ID)
        
        input_size = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']
        self.rnn = nn.LSTM(input_size+60, opt['hidden_dim'], opt['num_layers'], batch_first=True,\
                dropout=opt['dropout'])
        self.linear = nn.Linear(900, opt['num_class'])

        if opt['attn']:
            self.attn_layer = layers.PositionAwareAttention(opt['hidden_dim'],
                    opt['hidden_dim'], 2*opt['pe_dim'], opt['attn_dim'])
            self.pe_emb = nn.Embedding(constant.MAX_LEN * 2 + 1, opt['pe_dim'])

        self.opt = opt
        self.topn = self.opt.get('topn', 1e10)
        self.use_cuda = opt['cuda']
        self.emb_matrix = emb_matrix
        self.entity_linear = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
        self.init_weights()
    
    def init_weights(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0) # keep padding dimension to be 0
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        if self.opt['pos_dim'] > 0:
            self.pos_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        if self.opt['ner_dim'] > 0:
            self.ner_emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        self.entity_linear.weight.data.normal_(std=0.001)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
        if self.opt['attn']:
            self.pe_emb.weight.data.uniform_(-1.0, 1.0)

        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")

    def zero_state(self, batch_size): 
        state_shape = (self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = torch.zeros(*state_shape, requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0
    
    def forward(self, inputs):
        words, masks, pos, ner, deprel, subj_pos, obj_pos = inputs # unpack
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = words.size()[0]
        
        # embedding lookup
        word_inputs = self.emb(words)
        inputs = [word_inputs]
        if self.opt['pos_dim'] > 0:
            inputs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            inputs += [self.ner_emb(ner)]
        
        subj_pe_inputs = self.pe_emb(subj_pos + constant.MAX_LEN)
        obj_pe_inputs = self.pe_emb(obj_pos + constant.MAX_LEN)
        pe_features = torch.cat((subj_pe_inputs, obj_pe_inputs), dim=2)

        inputs += [pe_features]

        inputs = self.drop(torch.cat(inputs, dim=2)) # add dropout to input
        input_size = inputs.size(2)
        
        # rnn
        h0, c0 = self.zero_state(batch_size)
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=True)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        outputs, output_lens = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        hidden = self.drop(ht[-1,:,:]) # get the outmost layer h_n
        outputs = self.drop(outputs)
        
        sub_ind=(subj_pos==0)*(1-masks)
        obj_ind=(obj_pos==0)*(1-masks)
        
        # middle=sub_ind+obj_ind
        # middle=middle.cpu().numpy()
        # middle_array=np.zeros(middle.shape)
        # for i in range(middle.shape[0]):
        #     in_f=False
        #     out_f=False
        #     for j in range(middle.shape[1]):
        #         if j+1==middle.shape[1]:
        #             break
        #         if middle[i,j]>0.5 and middle[i,j+1]<0.5:
        #             in_f=True
        #         if in_f:
        #             if middle[i,j]<0.5 and middle[i,j+1]>0.5:
        #                 out_f=True
        #         if in_f and out_f==False:
        #             middle_array[i,j]=1

        # middle_tsr=torch.cuda.ByteTensor(middle_array)
        # middle_ind=middle_tsr.unsqueeze(-1).float()

        sub_ind=sub_ind.unsqueeze(-1).float()
        obj_ind=obj_ind.unsqueeze(-1).float()

        # attention
        if self.opt['attn']:
            # convert all negative PE numbers to positive indices
            # e.g., -2 -1 0 1 will be mapped to 98 99 100 101
            # tmasks=(1-middle_tsr+masks)>0.5
            # final_hidden = self.attn_layer(outputs*middle_ind, tmasks, hidden, pe_features*middle_ind)
            add_max=(1-masks).float().unsqueeze(-1)
            final_hidden = self.attn_layer(outputs*add_max, masks, hidden, pe_features*add_max)
        else:
            final_hidden = hidden

        # print(sub_ind,obj_ind)
        sub_embed=torch.sum(outputs*sub_ind,dim=1)/torch.sum(sub_ind,dim=1)
        obj_embed=torch.sum(outputs*obj_ind,dim=1)/torch.sum(obj_ind,dim=1)

        final_hidden=torch.cat([self.entity_linear(sub_embed),
            final_hidden,
            self.entity_linear(obj_embed)],dim=-1)

        logits = self.linear(final_hidden)
        return logits, final_hidden
    

