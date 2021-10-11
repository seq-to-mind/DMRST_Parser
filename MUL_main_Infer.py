import os
import torch
import numpy as np
import argparse
import os
import config
from transformers import AutoTokenizer, AutoModel
from model_depth import ParsingNet

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.global_gpu_id)


def parse_args():
    parser = argparse.ArgumentParser()
    """ config the saved checkpoint """
    parser.add_argument('--ModelPath', type=str, default='depth_mode/Savings/multi_all_checkpoint.torchsave', help='pre-trained model')
    base_path = config.tree_infer_mode + "_mode/"
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--savepath', type=str, default=base_path + './Savings', help='Model save path')
    args = parser.parse_args()
    return args


def inference(model, tokenizer, input_sentences, batch_size):
    LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

    input_sentences = [tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences]
    all_segmentation_pred = []
    all_tree_parsing_pred = []

    with torch.no_grad():
        for loop in range(LoopNeeded):
            StartPosition = loop * batch_size
            EndPosition = (loop + 1) * batch_size
            if EndPosition > len(input_sentences):
                EndPosition = len(input_sentences)

            input_sen_batch = input_sentences[StartPosition:EndPosition]
            _, _, SPAN_batch, _, predict_EDU_breaks = model.TestingLoss(input_sen_batch, input_EDU_breaks=None, LabelIndex=None,
                                                                        ParsingIndex=None, GenerateTree=True, use_pred_segmentation=True)
            all_segmentation_pred.extend(predict_EDU_breaks)
            all_tree_parsing_pred.extend(SPAN_batch)
    return input_sentences, all_segmentation_pred, all_tree_parsing_pred


if __name__ == '__main__':

    args = parse_args()
    model_path = args.ModelPath
    batch_size = args.batch_size
    save_path = args.savepath

    """ BERT tokenizer and model """
    bert_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base", use_fast=True)
    bert_model = AutoModel.from_pretrained("xlm-roberta-base")

    bert_model = bert_model.cuda()

    for name, param in bert_model.named_parameters():
        param.requires_grad = False

    model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)

    model = model.cuda()
    model.load_state_dict(torch.load(model_path))
    model = model.eval()

    Test_InputSentences = open("./data/text_for_inference.txt").readlines()

    input_sentences, all_segmentation_pred, all_tree_parsing_pred = inference(model, bert_tokenizer, Test_InputSentences, batch_size)
    print(input_sentences[0])
    print(all_segmentation_pred[0])
    print(all_tree_parsing_pred[0])
