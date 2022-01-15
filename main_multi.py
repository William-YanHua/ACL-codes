#-*- coding: utf8 -*-
import os
from re import split
import sys
current_path = os.path.abspath(__file__)
parent_path = os.path.abspath(os.path.join(current_path, os.path.pardir))
sys.path.append(parent_path)
grand_path = os.path.abspath(os.path.join(parent_path, os.path.pardir))
sys.path.append(grand_path)
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import DataParallel
from model.model_comment_context_multi import SASICM
from util.collate_fn_multi import collate_fn_dict
from util.Averager import Average, EvaluatorAverger
from util.Evaluator import Evaluator
from util.argparser import parse
from common_util.util import set_random_seed
from common_util.data_processor import Processor
from common_util.load_original_data import load_dataset
from common_util.pretrained_embedding_loader import PretrainedEmbedding
from torch.optim import Adam, AdamW, SGD
from common_util.callback import get_linear_schedule_with_warmup
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, BertConfig
from copy import deepcopy

cuda_devices_num = torch.cuda.device_count()
print(cuda_devices_num)
def load_custom_dataset(processor: Processor=None, args: dict=None, tokenizer=None):
    file_folder = args.dataset_file_folder
    fname = os.path.join(file_folder, 'cached_dataset')
    # print(fname, os.path.exists(fname), args.overwrite_cached_file)
    if (os.path.exists(fname) and (args.overwrite_cached_file is False)):
        print(f'Loading dataset from cached file: {fname}')
        datasets = torch.load(fname)
        return datasets
    else:
        datasets = load_dataset(os.path.join(file_folder, 'unified_datasets.json'))
        features = processor.process(datasets, tokenizer=tokenizer)
        if (args.cache_file or args.overwrite_cached_file):
            print(f'Saving dataset into cached file: {fname}')
            torch.save(features, fname)
        return features

def train(dataset, model, args, valid_dataset=None):
    schedule_file_name = 'single_schedule.sc'
    optimizer_file_name = 'single_optimizer.op'
    model_file_name = 'single_model.pt'
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()], "weight_decay": args.weight_decay}, # params without decay 
    ]
    optimizer = SGD(params=optimizer_grouped_parameters, lr=args.learning_rate)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=len(dataset)//(args.batch_size * 2)+1)
    start_epochs = 0
    if (args.continue_training):
        files = os.listdir(args.model_path)
        model_file = list(filter(lambda item: item.endswith(model_file_name), files))
        if (len(model_file) != 0):
            model_file = model_file[0]
            start_epochs = model_file.split('_')[0]
            start_epochs = int(start_epochs)+1
            model_state_dict = torch.load(os.path.join(args.model_path, model_file))
            model.load_state_dict(model_state_dict)
            # schedule_state_dict = torch.load(os.path.join(args.model_path, schedule_file_name))
            # scheduler.load_state_dict(schedule_state_dict)
            optimizer_state_dict = torch.load(os.path.join(args.model_path, optimizer_file_name))
            optimizer.load_state_dict(optimizer_state_dict)
    print(f'Training from {start_epochs} epochs!')
    average_loss = Average()
    last_best_f1 = 0.0
    datasampler = RandomSampler(dataset)
    if (args.devices == 'cuda'):
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=args.batch_size * cuda_devices_num, collate_fn=collate_fn_dict[args.mode])
    else:
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=args.batch_size, collate_fn=collate_fn_dict[args.mode])
    classes_weight = {
        'subt': np.array([1,3,5])
    }
    if (valid_dataset is None):
        classes_weight = np.array([len(list(filter(lambda item: item.subtext_label == 0, dataset))), \
            len(list(filter(lambda item: item.subtext_label == 1, dataset))), \
                len(list(filter(lambda item: item.subtext_label == 2, dataset)))])
        classes_weight = 10 * (1-np.exp(classes_weight)/np.sum(np.exp(classes_weight)))
    patience = 0
    for epoch in range(start_epochs, args.epochs):
        print(classes_weight)
        print(f'Epoch {epoch}/{args.epochs}:')
        for step, batch_data in enumerate(dataloader):
            model.train()
            batch_data = [t.to(args.devices) for t in batch_data]
            if (cuda_devices_num > 1):
                batch_data = [t.cuda(device=0) for t in batch_data]
            inputs = {'comment': batch_data[0], 'comment_context': batch_data[1], \
                'comment_length': batch_data[2], 'comment_context_length': batch_data[3], \
                'subt_label': batch_data[4], 'sarc_label': batch_data[5], 'meta_label': batch_data[6],\
                'classes_weight': classes_weight, 'token_type_ids': batch_data[7], 'comment_attention_mask': \
                batch_data[8], 'comment_context_attention_mask': batch_data[9], 'comment_position_input':\
                batch_data[10], 'comment_context_position_input': batch_data[11]}
            # if (step == 42):
            #     import pdb; pdb.set_trace()
            loss, subt_pred = model(**inputs)
            if (cuda_devices_num > 1):
                loss=loss.mean()
            average_loss.update(loss)
            loss /= args.accumulation_step
            loss.backward()
            
            if ((step + 1) % args.accumulation_step == 0):
                # 防止梯度爆炸
                for n, p in model.named_parameters():
                    if (p.grad is not None and torch.isnan(p.grad).any()):
                        q = p.grad.detach().clone()
                        q[torch.isnan(p.grad)] = 0.0
                        p = torch.where(torch.isnan(p.grad), q, p.grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, args.norm_type)
                optimizer.step()
                optimizer.zero_grad()
            # scheduler.step() # update learning rate
            if ((step + 1) % (args.echo_per_step*args.accumulation_step) == 0):
                print(f'Average_loss at step {step + 1}:\t {average_loss.average()}')
                average_loss.reset()
            if 'cuda' in str(args.devices):
                torch.cuda.empty_cache()
        if (valid_dataset):
            evaluate_score, evaluate_value = evaluate(valid_dataset, model, args)
            f1 = evaluate_value
            if (last_best_f1 < f1):
                patience = 0
                print(f'Current evaluate value {f1: 0.4f} is better than {last_best_f1: 0.4f}')
                print(f"Saving to {os.path.join(args.model_path, f'{epoch}_{model_file_name}')}.")
                files = os.listdir(args.model_path)
                old_model_file = list(filter(lambda item: item.endswith(model_file_name), files))
                if (not args.continue_training and len(old_model_file) > 0):
                    os.system(f"rm {os.path.join(args.model_path, old_model_file[0])}")
                    old_model_file = []
                torch.save(model.state_dict(), os.path.join(args.model_path, f'{epoch}_{model_file_name}'))
                torch.save(optimizer.state_dict(), os.path.join(args.model_path, optimizer_file_name))
                # torch.save(scheduler.state_dict(),os.path.join(args.model_path, schedule_file_name))
                if (len(old_model_file) > 0):
                    os.system(f"rm {os.path.join(args.model_path, old_model_file[0])}")
                last_best_f1 = f1
            else:
                patience += 1
            type_map = {
                'subt': {
                    0: 'No subtext',
                    1: 'Unsure',
                    2: 'Subtext'
                }
            }
            for key in ['subt']:
                f1s = [evaluate_score[key][type_map[key][0]]['f1'], evaluate_score[key][type_map[key][1]]['f1'], evaluate_score[key][type_map[key][2]]['f1']]
                f1s = np.array(f1s)
                classes_weight[key] = 10 * (1-np.exp(f1s)/np.sum(np.exp(f1s)))

        if (args.early_stop and patience >= args.patience):
            break

def evaluate(dataset, model, args):
    datasampler = RandomSampler(dataset)
    if (args.devices == 'cuda'):
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=args.batch_size * cuda_devices_num, collate_fn=collate_fn_dict[args.mode])
    else:
        dataloader = DataLoader(dataset, sampler=datasampler, batch_size=args.batch_size, collate_fn=collate_fn_dict[args.mode])
    type_map = {
        'subt': {
            0: 'No subtext',
            1: 'Unsure',
            2: 'Subtext'
        }
    }
    subt_evaluator = Evaluator(type_map=type_map['subt'])
    average_loss = Average()
    classes_weight = {
        'subt': np.array([1,1,1])
    }
    for step, batch_data in tqdm(enumerate(dataloader), desc='Evalutating: '):
        batch_data = [t.to(args.devices) for t in batch_data]
        model.eval()
        if (cuda_devices_num > 1):
            batch_data = [t.cuda(device=0) for t in batch_data]
        with torch.no_grad():
            inputs = {'comment': batch_data[0], 'comment_context': batch_data[1], \
                'comment_length': batch_data[2], 'comment_context_length': batch_data[3], \
                'subt_label': batch_data[4], 'sarc_label': batch_data[5], 'meta_label': batch_data[6],\
                'classes_weight': classes_weight, 'token_type_ids': batch_data[7], 'comment_attention_mask': \
                    batch_data[8], 'comment_context_attention_mask': batch_data[9], 'comment_position_input': batch_data[10], 'comment_context_position_input': batch_data[11]}
            loss, subt_pred= model(**inputs)
            if (cuda_devices_num > 1):
                loss = loss.mean()
            average_loss.update(loss)

            subt_pred = torch.argmax(subt_pred, -1)
            subt_pred = subt_pred.cpu().numpy().tolist()
            subt_gt = batch_data[4].cpu().numpy().tolist()
            subt_evaluator.update(subt_gt, subt_pred)


            if (args.devices == 'cuda'):
                torch.cuda.empty_cache()
    subt_results, subt_eval_value = subt_evaluator.result()
    eval_value = subt_eval_value
    print(f'Evaluation loss： {average_loss.average(): 0.4f}')
    return {'subt': subt_results}, eval_value

def test(dataset, model, args):
    model_file_name = 'model.pt'
    files = os.listdir(args.model_path)
    model_file = list(filter(lambda item: item.endswith(model_file_name), files))
    if (len(model_file) != 0):
        model_file = model_file[0]
        start_epochs = model_file.split('_')[0]
        start_epochs = int(start_epochs)+1
        model_state_dict = torch.load(os.path.join(args.model_path, model_file))
        model.load_state_dict(model_state_dict)
    print(f'Testing at {start_epochs} epochs!')
    return evaluate(dataset, model, args)

def cross_validation(dataset, model, args):
    def get_kfold_train_test(dataset, n_splits):
        indexes = np.arange(0, len(dataset))
        kfolder = KFold(n_splits=n_splits, shuffle=True, random_state=args.random_seed)
        for train_indexes, test_indexes in kfolder.split(indexes):
            train_datasets = [dataset[i] for i in train_indexes]
            test_datasets = [dataset[i] for i in test_indexes]
            yield train_datasets, test_datasets
    def get_train_valid_test(dataset, split_rate: list=[0.6, 0.2]):
        dataset_size = len(dataset)
        indexes = np.arange(0, dataset_size)
        classes = np.array([v.subtext_label for v in dataset])
        train_indexes, test_indexes, _, _ = train_test_split(indexes, indexes, test_size=split_rate[1], stratify=classes)
        train_indexes, valid_indexes, _, _ = train_test_split(train_indexes, train_indexes, test_size=1-split_rate[1]-split_rate[0])
        train_datasets = [dataset[i] for i in train_indexes]
        test_datasets = [dataset[i] for i in test_indexes]
        valid_datasets = [dataset[i] for i in valid_indexes]
        return train_datasets, valid_datasets, test_datasets
    if (args.validation_folder == 1):
        train_datasets, valid_datasets, test_datasets = get_train_valid_test(dataset, [args.train_rate, args.test_rate])
        if (args.train):
            train(train_datasets, model, args, valid_datasets)
            if (args.eval):
                test(test_datasets, model, args)
        elif (args.eval):
            test(test_datasets, model, args)
    else:
        kfold = get_kfold_train_test(dataset, args.validation_folder)
        subt_avg = EvaluatorAverger(['No subtext', 'Subtext', 'Unsure', 'Macro F1', 'Weighted F1'])
        for fold_idx, (train_dataset, test_dataset) in enumerate(kfold):
            print(f'==== fold {fold_idx} ==== ')
            trained_model = deepcopy(model)
            classes = np.array([v.subtext_label for v in train_dataset])
            indexes = np.arange(0, len(train_dataset))
            train_indexes, valid_indexes, _, _ = train_test_split(indexes, indexes, test_size=0.1, stratify=classes)
            # import pdb; pdb.set_trace()
            train_dataset_ = [train_dataset[i] for i in train_indexes]
            valid_dataset = [train_dataset[i] for i in valid_indexes]
            train_dataset = train_dataset_
            train(train_dataset, trained_model, args, valid_dataset)
            if (args.eval):
                results, evaluate_values = test(test_dataset, trained_model, args)
                subt_avg.update(results['subt'])
        print('+++++++     Subtext Results  +++++++')
        subt_avg.average()

if __name__ == '__main__':
    args = parse()
    set_random_seed(args.random_seed)
    if (not os.path.exists(args.model_path)):
        os.makedirs(args.model_path)
    # import pdb; pdb.set_trace()
    pretrainedEmbedding = PretrainedEmbedding()
    pretrainedEmbedding.load_dict(args.pretrained_file_folder, 'token')
    vocab_size = pretrainedEmbedding.get_vocab_size()
    pretrainedEmbedding.load_embedding(args.pretrained_file_folder, 'token')
    embedding_size = pretrainedEmbedding.get_embedding_size()
    # tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    # model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-bert-wwm-ext")
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_file_path)
    bert_config = BertConfig.from_pretrained(args.pretrained_file_path)
    # load dataset
    # print(args.pretrained_file_folder)
    processor = Processor(args.pretrained_file_folder)
    datasets = load_custom_dataset(processor, args, tokenizer)
    position_size = {
        'glove': 2048,
        'bert': 513
    }
    model = SASICM(args, vocab_size=vocab_size, position_size=position_size[args.mode], embedding_dim=embedding_size, \
        embeddings = pretrainedEmbedding.get_embedding(), bert_config=bert_config, using_bert=(args.mode == 'bert')).to(args.devices)
    if (cuda_devices_num != 0):
        model = DataParallel(model, device_ids=[i for i in range(cuda_devices_num)])
        model = model.cuda(device=0)
    cross_validation(datasets, model, args)
