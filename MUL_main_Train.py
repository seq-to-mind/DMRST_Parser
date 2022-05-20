import re
import os
import pickle
import torch
import numpy as np
import random
import argparse
from Training import Train
import time
from transformers import AutoTokenizer, AutoModel
import config
from model_depth import ParsingNet

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.global_gpu_id)


def parse_args():
    parser = argparse.ArgumentParser(description='RSTParser')
    parser.add_argument('--GPUforModel', type=int, default=config.global_gpu_id, help='Which GPU to run')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')

    parser.add_argument('--hidden_size', type=int, default=config.hidden_size, help='Hidden size of RNN')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--dropout_e', type=float, default=0.5, help='Dropout rate for encoder')
    parser.add_argument('--dropout_d', type=float, default=0.5, help='Dropout rate for decoder')
    parser.add_argument('--dropout_c', type=float, default=0.5, help='Dropout rate for classifier')
    parser.add_argument('--input_is_word', type=str, default='True', help='Whether the encoder input is word or EDU')

    parser.add_argument('--atten_model', choices=['Dotproduct', 'Biaffine'], default='Dotproduct', help='Attention mode')
    parser.add_argument('--classifier_input_size', type=int, default=config.hidden_size, help='Input size of relation classifier')
    parser.add_argument('--classifier_hidden_size', type=int, default=int(config.hidden_size / 1), help='Hidden size of relation classifier')
    parser.add_argument('--classifier_bias', type=str, default='True', help='Whether classifier has bias')
    parser.add_argument('--seed', type=int, default=config.random_seed, help='Seed number')
    parser.add_argument('--eval_size', type=int, default=30, help='Evaluation size')
    parser.add_argument('--epoch', type=int, default=15, help='Epoch number')

    parser.add_argument('--lr', type=float, default=0.00002, help='Initial lr')
    parser.add_argument('--lr_decay_epoch', type=int, default=1, help='Lr decay epoch')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay rate')

    base_path = config.tree_infer_mode + "_mode/"
    parser.add_argument('--datapath', type=str, default=base_path + './pkl_data_for_train/en-gum/', help='Data path')
    parser.add_argument('--savepath', type=str, default=base_path + './Savings', help='Model save path')
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:" + str(args.GPUforModel) if USE_CUDA else "cpu")

    batch_size = args.batch_size
    hidden_size = args.hidden_size
    rnn_layers = args.rnn_layers
    dropout_e = args.dropout_e
    dropout_d = args.dropout_d
    dropout_c = args.dropout_c
    input_is_word = args.input_is_word
    atten_model = args.atten_model
    classifier_input_size = args.classifier_input_size
    classifier_hidden_size = args.classifier_hidden_size
    classifier_bias = args.classifier_bias

    data_path = args.datapath
    save_path = args.savepath
    seednumber = args.seed
    eval_size = args.eval_size
    epoch = args.epoch
    lr = args.lr
    lr_decay_epoch = args.lr_decay_epoch
    weight_decay = args.weight_decay

    """ BERT tokenizer and model """
    bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    bert_model = AutoModel.from_pretrained("xlm-roberta-base")

    """ freeze some layers """
    for name, param in bert_model.named_parameters():
        layer_num = re.findall("layer\.(\d+)\.", name)
        if len(layer_num) > 0 and int(layer_num[0]) > 2:
            param.requires_grad = True
        else:
            param.requires_grad = False

    language_model = bert_model.cuda()

    # Setting random seeds 
    torch.manual_seed(seednumber)
    if USE_CUDA:
        torch.cuda.manual_seed_all(seednumber)
    np.random.seed(seednumber)
    random.seed(seednumber)

    # Process bool args       
    if args.classifier_bias == 'True':
        classifier_bias = True

    elif args.classifier_bias == 'False':
        classifier_bias = False

    Tr_InputSentences = []
    Tr_EDUBreaks = []
    Tr_DecoderInput = []
    Tr_RelationLabel = []
    Tr_ParsingBreaks = []
    Tr_GoldenMetric = []
    Tr_ParentsIndex = []
    Tr_SiblingIndex = []

    # Load Testing data
    Test_InputSentences = []
    Test_EDUBreaks = []
    Test_DecoderInput = []
    Test_RelationLabel = []
    Test_ParsingBreaks = []
    Test_GoldenMetric = []

    # Load Training data
    Tr_InputSentences.extend(pickle.load(open(os.path.join(data_path, "Training_InputSentences.pickle"), "rb")))
    Tr_EDUBreaks.extend(pickle.load(open(os.path.join(data_path, "Training_EDUBreaks.pickle"), "rb")))
    Tr_DecoderInput.extend(pickle.load(open(os.path.join(data_path, "Training_DecoderInputs.pickle"), "rb")))
    Tr_RelationLabel.extend(pickle.load(open(os.path.join(data_path, "Training_RelationLabel.pickle"), "rb")))
    Tr_ParsingBreaks.extend(pickle.load(open(os.path.join(data_path, "Training_ParsingIndex.pickle"), "rb")))
    Tr_GoldenMetric.extend(pickle.load(open(os.path.join(data_path, "Training_GoldenLabelforMetric.pickle"), "rb")))
    Tr_ParentsIndex.extend(pickle.load(open(os.path.join(data_path, "Training_ParentsIndex.pickle"), "rb")))
    Tr_SiblingIndex.extend(pickle.load(open(os.path.join(data_path, "Training_Sibling.pickle"), "rb")))

    # Load Testing data
    Test_InputSentences.extend(pickle.load(open(os.path.join(data_path, "Testing_InputSentences.pickle"), "rb")))
    Test_EDUBreaks.extend(pickle.load(open(os.path.join(data_path, "Testing_EDUBreaks.pickle"), "rb")))
    Test_DecoderInput.extend(pickle.load(open(os.path.join(data_path, "Testing_DecoderInputs.pickle"), "rb")))
    Test_RelationLabel.extend(pickle.load(open(os.path.join(data_path, "Testing_RelationLabel.pickle"), "rb")))
    Test_ParsingBreaks.extend(pickle.load(open(os.path.join(data_path, "Testing_ParsingIndex.pickle"), "rb")))
    Test_GoldenMetric.extend(pickle.load(open(os.path.join(data_path, "Testing_GoldenLabelforMetric.pickle"), "rb")))

    # To check data
    sent_temp = ''
    print("Checking Data...")
    for word_temp in Tr_InputSentences[2]:
        sent_temp = sent_temp + ' ' + word_temp
    print(sent_temp)
    print('... ...')
    print("That's great! No error found!")
    print("All train sample number:", len(Tr_InputSentences))

    # To save model and data
    FileName = str(seednumber) + "_" + config.tree_infer_mode + '_Batch_' + str(batch_size) + 'Hidden_' + str(hidden_size) + \
               'LR' + str(lr) + "_" + str(time.time())

    SavePath = os.path.join(save_path, FileName)
    print(SavePath)

    """ relation number is set at 42 """
    model = ParsingNet(language_model, hidden_size, hidden_size,
                       hidden_size, atten_model, classifier_input_size, classifier_hidden_size, 42,
                       classifier_bias, rnn_layers, dropout_e, dropout_d, dropout_c, bert_tokenizer=bert_tokenizer)

    model = model.cuda()


    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print("Total trainable parameter number is: ", count_parameters(model))

    TrainingProcess = Train(model, Tr_InputSentences, Tr_EDUBreaks, Tr_DecoderInput,
                            Tr_RelationLabel, Tr_ParsingBreaks, Tr_GoldenMetric,
                            Tr_ParentsIndex, Tr_SiblingIndex,
                            Test_InputSentences, Test_EDUBreaks, Test_DecoderInput,
                            Test_RelationLabel, Test_ParsingBreaks, Test_GoldenMetric,
                            batch_size, eval_size, epoch, lr, lr_decay_epoch,
                            weight_decay, SavePath)

    best_epoch, best_F_relation, best_P_relation, best_R_relation, best_F_span, \
    best_P_span, best_R_span, best_F_nuclearity, best_P_nuclearity, \
    best_R_nuclearity = TrainingProcess.train()

    print('--------------------------------------------------------------------')
    print('Training Completed!')
    print('Processing...')
    print('The best F1 points for Relation is: %f.' % (best_F_relation))
    print('The best F1 points for Nuclearity is: %f' % (best_F_nuclearity))
    print('The best F1 points for Span is: %f' % (best_F_span))

    # Save result
    with open(os.path.join(args.savepath, 'Results.csv'), 'a') as f:
        f.write(FileName + ',' + ','.join(map(str, [best_epoch, best_F_relation, \
                                                    best_P_relation, best_R_relation, best_F_span, \
                                                    best_P_span, best_R_span, best_F_nuclearity, \
                                                    best_P_nuclearity, best_R_nuclearity])) + '\n')
