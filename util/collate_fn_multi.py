#-*- coding: utf8 -*-
import torch
import numpy as np
def collate_fn(batch):
    # import pdb; pdb.set_trace()
    batch_size = len(batch)
    max_comment_context_lengths = max([v.comment_context_token_length for v in batch])
    comment_context = torch.zeros(batch_size, max_comment_context_lengths, dtype=torch.long)
    subtext_label = torch.zeros(batch_size, dtype=torch.long)
    sarcasm_label = torch.zeros(batch_size, dtype=torch.long)
    metaphor_label = torch.zeros(batch_size, dtype=torch.long)
    comment_length = torch.from_numpy(np.array([v.comment_range_for_token[1]-v.comment_range_for_token[0] for v in batch]))
    max_comment_length = max(comment_length)
    comment_inputs = torch.zeros(batch_size, max_comment_length, dtype=torch.long)
    comment_context_lengths = torch.zeros(batch_size, dtype=torch.int)
    token_type_ids = torch.zeros((batch_size, max_comment_context_lengths), dtype=torch.long)
    comment_attention_mask = torch.zeros((batch_size, max_comment_length), dtype=torch.int)
    comment_context_attention_mask = torch.zeros((batch_size, max_comment_length), dtype=torch.int)
    comment_position_ids = torch.zeros(batch_size, max_comment_length, dtype=torch.long)
    comment_context_position_ids = torch.zeros(batch_size, max_comment_context_lengths, dtype=torch.long)
    for idx, v in enumerate(batch):
        comment_context_len = v.comment_context_token_length
        comment_context[idx, :comment_context_len] = torch.from_numpy(v.comment_context_token_id)
        subtext_label[idx] = v.subtext_label
        sarcasm_label[idx] = v.sarcasm_label
        metaphor_label[idx] = v.metaphor_label
        comment_context_lengths[idx] = comment_context_len
        comment_inputs[idx, :comment_length[idx]] = torch.from_numpy(v.comment_context_token_id[v.comment_range_for_token[0]:v.comment_range_for_token[1]])
        comment_position_ids[idx, :comment_length[idx]]=torch.from_numpy(np.arange(1, comment_length[idx]+1))
        comment_context_position_ids[idx, :comment_context_len]=torch.from_numpy(np.arange(1, comment_context_len+1))
    comment_attention_mask = (comment_inputs == 0).int()
    comment_context_attention_mask = (comment_context == 0).int()
    return comment_inputs, comment_context, comment_length, comment_context_lengths, subtext_label, sarcasm_label, metaphor_label, token_type_ids, comment_attention_mask, comment_context_attention_mask, comment_position_ids, comment_context_position_ids

def collate_fn_bert(batch):
    # import pdb; pdb.set_trace()
    batch_size = len(batch)
    max_comment_context_lengths = max([v.comment_context_bert_length for v in batch])
    comment_context = torch.zeros(batch_size, max_comment_context_lengths, dtype=torch.long)
    subtext_label = torch.zeros(batch_size, dtype=torch.long)
    sarcasm_label = torch.zeros(batch_size, dtype=torch.long)
    metaphor_label = torch.zeros(batch_size, dtype=torch.long)
    comment_length = torch.from_numpy(np.array([v.comment_range_for_bert[1]-v.comment_range_for_bert[0] for v in batch]))
    max_comment_length = max(comment_length)
    comment_inputs = torch.zeros(batch_size, max_comment_length, dtype=torch.long)
    comment_context_lengths = torch.zeros(batch_size, dtype=torch.int)
    token_type_ids = torch.zeros((batch_size, max_comment_context_lengths), dtype=torch.long)
    comment_attention_mask = torch.zeros((batch_size, max_comment_length), dtype=torch.int)
    comment_context_attention_mask = torch.zeros((batch_size, max_comment_length), dtype=torch.int)
    comment_position_ids = torch.zeros(batch_size, max_comment_length, dtype=torch.long)
    comment_context_position_ids = torch.zeros(batch_size, max_comment_context_lengths, dtype=torch.long)
    for idx, v in enumerate(batch):
        comment_context_len = v.comment_context_bert_length
        comment_context[idx, :comment_context_len] = torch.from_numpy(v.comment_context_bert_id)
        subtext_label[idx] = v.subtext_label
        sarcasm_label[idx] = v.sarcasm_label
        metaphor_label[idx] = v.metaphor_label
        comment_context_lengths[idx] = comment_context_len
        comment_inputs[idx, :comment_length[idx]] = torch.from_numpy(v.comment_context_bert_id[v.comment_range_for_bert[0]:v.comment_range_for_bert[1]])
        try:
            token_type_ids[idx, :comment_context_len] = torch.from_numpy(v.token_type_ids)
        except:
            import pdb; pdb.set_trace()
        comment_position_ids[idx, :comment_length[idx]]=torch.from_numpy(np.arange(1, comment_length[idx]+1))
        comment_context_position_ids[idx, :comment_context_len]=torch.from_numpy(np.arange(1, comment_context_len+1))
    comment_attention_mask = (comment_inputs == 0).int()
    comment_context_attention_mask = (comment_context == 0).int()
    return comment_inputs, comment_context, comment_length, comment_context_lengths, subtext_label, sarcasm_label, metaphor_label, token_type_ids, comment_attention_mask, comment_context_attention_mask, comment_position_ids, comment_context_position_ids

collate_fn_dict = {
    'glove': collate_fn,
    'bert': collate_fn_bert
}
