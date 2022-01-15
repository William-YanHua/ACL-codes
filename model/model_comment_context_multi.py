#-*- coding: utf8 -*-
from numpy.core.fromnumeric import size
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import os
import sys
current_path = os.path.abspath(__file__)
parent_path = os.path.abspath(os.path.join(current_path, os.path.pardir, os.path.pardir))
sys.path.append(parent_path)
from transformers  import BertModel, BertForPreTraining
class SASICM(BertForPreTraining):
    def __init__(self, args, vocab_size, position_size, embedding_dim, embeddings, bert_config, using_bert = False):
        super(SASICM, self).__init__(bert_config)
        self.args = args
        if (using_bert):
            self.bert = BertModel(bert_config)
            # import pdb; pdb.set_trace()
            embedding_dim = bert_config.__dict__['hidden_size']
            vocab_size = bert_config.__dict__['vocab_size']
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedder = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=args.padding_idx).from_pretrained(embeddings=torch.from_numpy(embeddings), freeze=False)
        self.comment_lstm = nn.GRU(input_size = embedding_dim, hidden_size = embedding_dim, batch_first=True, bidirectional = True, dropout = args.comment_dropout_rate)
        self.context_lstm = nn.GRU(input_size = embedding_dim, hidden_size = embedding_dim, batch_first=True, bidirectional = True, dropout=args.comment_context_dropout_rate)
        self.comment_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)
        self.context_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, batch_first=True)
        self.comment_feature_ext = nn.Linear(in_features=embedding_dim * 3, out_features=args.meaning_size)
        self.comment_context_feature_ext = nn.Linear(in_features=embedding_dim * 3, out_features=args.meaning_size)
        self.subt_classifier = nn.Sequential(nn.Linear(in_features=2*args.meaning_size, out_features=args.meaning_size), nn.ReLU(), nn.Linear(in_features=args.meaning_size, out_features=3))

        self.positional_embedder = nn.Embedding(num_embeddings=position_size, embedding_dim=args.position_dim, padding_idx=args.position_padding_idx)
        self.embed_dim_reduction = nn.Linear(in_features=embedding_dim + args.position_dim, out_features=embedding_dim)
        self.using_bert = using_bert
        # self.init_weights()

    def forward(self, comment=None, comment_context=None, comment_length=None, comment_context_length=None, \
        subt_label=None, sarc_label=None, meta_label=None, classes_weight=None, token_type_ids = None, comment_attention_mask=None, comment_context_attention_mask=None, comment_position_input=None, comment_context_position_input=None): # inputs comment, outputs context
        if (self.using_bert):
            try:
                comment_inputs = self.bert(input_ids=comment, attention_mask = comment_attention_mask)
                comment_context_inputs = self.bert(input_ids=comment_context, attention_mask = comment_context_attention_mask, token_type_ids=token_type_ids)
            except:
                import pdb; pdb.set_trace()
            comment_inputs = comment_inputs[0]
            comment_context_inputs = comment_context_inputs[0]
        else:
            comment_inputs = self.embedder(comment)
            comment_context_inputs = self.embedder(comment_context)

        if (comment_position_input is not None):
            try:
                comment_position_embedding = self.positional_embedder(comment_position_input)
                comment_context_position_embedding = self.positional_embedder(comment_context_position_input)
            except:
                import pdb; pdb.set_trace()
            comment_inputs = torch.cat([comment_inputs, comment_position_embedding], -1)
            comment_context_inputs = torch.cat([comment_context_inputs, comment_context_position_embedding], -1)
            comment_inputs = self.embed_dim_reduction(comment_inputs.float())
            comment_context_inputs = self.embed_dim_reduction(comment_context_inputs.float())
        # import pdb; pdb.set_trace()

        pack_comment_inputs = nn.utils.rnn.pack_padded_sequence(comment_inputs, comment_length.tolist(), batch_first=True, enforce_sorted=False)
        pack_comment_features, _ = self.comment_lstm(pack_comment_inputs.float())
        comment_features, _ = nn.utils.rnn.pad_packed_sequence(pack_comment_features, batch_first=True)
        # h_f_c = h_f_c.permute(1,0,2).reshape(batch_size, 2 * embedding_size)
        h_f_c = torch.sum(comment_features, 1)

        pack_comment_context_inputs = nn.utils.rnn.pack_padded_sequence(comment_context_inputs, comment_context_length.tolist(), batch_first=True, enforce_sorted=False)
        pack_comment_context_features, h_f_c_c = self.context_lstm(pack_comment_context_inputs.float())
        comment_context_features, _ = nn.utils.rnn.pad_packed_sequence(pack_comment_context_features, batch_first=True)
        # h_f_c_c = h_f_c_c.permute(1,0,2).reshape(batch_size, 2 * embedding_size)
        h_f_c_c = torch.sum(comment_context_features, 1)

        comment_attention_feature, comment_weights = self.comment_attention(comment_inputs.float(), comment_inputs.float(), comment_inputs.float(), comment==0)

        comment_attention_feature = torch.sum(comment_attention_feature, axis=1)
        comment_meaning = self.comment_feature_ext(torch.cat([h_f_c, comment_attention_feature], -1))

        context_attention_features, context_weights = self.context_attention(comment_context_inputs.float(), comment_context_inputs.float(), comment_context_inputs.float(), comment_context==0)
        context_attention_features = torch.sum(context_attention_features, axis=1)
        context_meaning = self.comment_context_feature_ext(torch.cat([h_f_c_c, context_attention_features], -1))
        meaning = torch.cat([comment_meaning, context_meaning], -1)
        subt_pred = self.subt_classifier(meaning)
        if (subt_pred is not None):
            loss_func = CrossEntropyLoss(reduction='none')
            loss = 0
            subt_loss = loss_func(subt_pred, subt_label)
            subt_w = torch.zeros(subt_label.size()).to(self.args.devices)
            subt_w[subt_label == 0] = classes_weight['subt'][0]
            subt_w[subt_label == 1] = classes_weight['subt'][1]
            subt_w[subt_label == 2] = classes_weight['subt'][2]
            # import pdb; pdb.set_trace()
            loss += torch.sum(subt_loss * subt_w, -1)
            
            loss += 0.01 * (torch.norm(self.comment_feature_ext.weight, 1) + torch.norm(self.comment_context_feature_ext.weight, 1))
            for params in self.subt_classifier.parameters():
                loss += 0.01 * (torch.norm(params, 1))
            
        return loss, subt_pred
