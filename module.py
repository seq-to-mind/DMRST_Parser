import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
import numpy as np


class EncoderRNN(nn.Module):
    def __init__(self, language_model, word_dim, hidden_size, rnn_layers, dropout, bert_tokenizer=None, segmenter=None):

        super(EncoderRNN, self).__init__()
        '''
        Input:
            [batch,length]
        Output: 
            encoder_output: [batch,length,hidden_size]    
            encoder_hidden: [rnn_layers,batch,hidden_size]
        '''

        self.rnn_layers = rnn_layers
        self.hidden_size = hidden_size
        self.word_dim = word_dim

        self.nnDropout = nn.Dropout(dropout)
        self.language_model = language_model

        self.layer_norm = nn.LayerNorm(word_dim, elementwise_affine=True)
        self.layer_norm_for_seg = nn.LayerNorm(word_dim, elementwise_affine=True)

        self.bert_tokenizer = bert_tokenizer
        self.reduce_dim_layer = nn.Linear(word_dim * 3, word_dim, bias=False)

        self.segmenter = segmenter
        self.doc_gru_enc = nn.GRU(word_dim, int(word_dim / 2), num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)

    def forward(self, input_sentence, EDU_breaks, is_test=False):
        if EDU_breaks is not None or is_test is False:
            max_edu_break_num = max([len(tmp_l) for tmp_l in EDU_breaks])
        all_outputs = []
        all_hidden = []

        batch_token_len_list = [len(i) for i in input_sentence]
        batch_token_len_max = max(batch_token_len_list)

        """ version 3.0 """
        # for segmenter initialization
        total_edu_loss = torch.FloatTensor([0.0]).cuda()
        predict_edu_breaks_list = []
        tem_outputs = []

        """ For averaging the edu level embeddings START """
        for i in range(len(input_sentence)):
            bert_token_ids = [self.bert_tokenizer.convert_tokens_to_ids(input_sentence[i])]
            bert_token_ids = torch.LongTensor(bert_token_ids).cuda()
            # print(bert_token_ids.shape)

            """ fixed sliding window for encoding long sequence """
            window_size = 300
            sequence_length = len(input_sentence[i])
            slide_steps = int(np.ceil(len(input_sentence[i]) / window_size))
            # print(sequence_length, slide_steps)
            window_embed_list = []
            for tmp_step in range(slide_steps):
                if tmp_step == 0:
                    one_win_res = self.language_model(bert_token_ids[:, :500])[0][:, :window_size, :]
                    window_embed_list.append(one_win_res)
                elif tmp_step == slide_steps - 1:
                    one_win_res = self.language_model(bert_token_ids[:, -((sequence_length - (window_size * tmp_step)) + 200):])[0][:, 200:, :]
                    window_embed_list.append(one_win_res)
                else:
                    one_win_res = self.language_model(bert_token_ids[:, (window_size * tmp_step - 100):(window_size * (tmp_step + 1) + 100)])[0][:, 100:400, :]
                    window_embed_list.append(one_win_res)

            embeddings = torch.cat(window_embed_list, dim=1)
            assert embeddings.size(1) == sequence_length
            embeddings = self.layer_norm(embeddings)

            """ add segmentation process """
            if is_test:
                predict_edu_breaks = self.segmenter.test_segment_loss(embeddings.squeeze())
                cur_edu_break = predict_edu_breaks
                predict_edu_breaks_list.append(predict_edu_breaks)

            else:
                cur_edu_break = EDU_breaks[i]
                seg_loss = self.segmenter.train_segment_loss(embeddings.squeeze(), cur_edu_break)
                """ Use this to pass the segmenation loss part: only for debug """
                # seg_loss = 0.0
                total_edu_loss += seg_loss

            # apply dropout
            embeddings = self.nnDropout(embeddings.squeeze(dim=0))
            tmp_average_list = []
            tmp_break_list = [0, ] + [tmp_j + 1 for tmp_j in cur_edu_break]
            for tmp_i in range(len(tmp_break_list) - 1):
                assert tmp_break_list[tmp_i] < tmp_break_list[tmp_i + 1]
                tmp_average_list.append(torch.mean(embeddings[tmp_break_list[tmp_i]:tmp_break_list[tmp_i + 1], :], dim=0, keepdim=True))
            tmp_average_embed = torch.cat(tmp_average_list, dim=0).unsqueeze(dim=0)
            outputs = tmp_average_embed

            """ For averaging the edu level embeddings END """
            if config.document_enc_gru is True:
                outputs, hidden = self.doc_gru_enc(outputs)
                hidden = hidden.view(2, 2, 1, int(self.word_dim / 2))[-1]
                hidden = hidden.transpose(0, 1).view(1, 1, -1).contiguous()

            if config.add_first_and_last is True:
                first_words = []
                last_words = []
                for tmp_i in range(len(tmp_break_list) - 1):
                    first_words.append(embeddings[tmp_break_list[tmp_i]].unsqueeze(dim=0))
                    last_words.append(embeddings[tmp_break_list[tmp_i + 1] - 1].unsqueeze(dim=0))

                outputs = torch.cat((outputs, torch.cat(first_words, dim=0).unsqueeze(dim=0), torch.cat(last_words, dim=0).unsqueeze(dim=0)), dim=2)
                outputs = self.reduce_dim_layer(outputs)

            tem_outputs.append(outputs)
            all_hidden.append(hidden)

        if is_test:
            max_edu_break_num = max([len(tmp_l) for tmp_l in predict_edu_breaks_list])
        for output in tem_outputs:
            cur_break_num = output.size(1)
            all_outputs.append(torch.cat([output, torch.zeros(1, max_edu_break_num - cur_break_num, self.word_dim).cuda()], dim=1))

        res_merged_output = torch.cat(all_outputs, dim=0)
        res_merged_hidden = torch.cat(all_hidden, dim=1)

        return res_merged_output, res_merged_hidden, total_edu_loss, predict_edu_breaks_list

    def GetEDURepresentation(self, input_sentence):
        tmp_max_token_num = len(input_sentence[0])
        bert_token_ids = [self.bert_tokenizer.convert_tokens_to_ids(v) + [5, ] * (tmp_max_token_num - len(v)) for k, v in enumerate(input_sentence)]
        bert_token_ids = torch.LongTensor(bert_token_ids).cuda()
        bert_embeddings = self.language_model(bert_token_ids)

        return bert_embeddings[0]


class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_layers, dropout):
        super(DecoderRNN, self).__init__()

        '''
        Input:
            input: [1,length,input_size]
            initial_hidden_state: [rnn_layer,1,hidden_size]
        Output:
            output: [1,length,input_size]
            hidden_states: [rnn_layer,1,hidden_size]
        '''
        # Define GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers=rnn_layers, batch_first=True, dropout=(0 if rnn_layers == 1 else dropout))

    def forward(self, input_hidden_states, last_hidden):
        # Forward through unidirectional GRU
        outputs, hidden = self.gru(input_hidden_states, last_hidden)

        return outputs, hidden


class PointerAtten(nn.Module):
    def __init__(self, atten_model, hidden_size):
        super(PointerAtten, self).__init__()

        '''       
        Input:
            Encoder_outputs: [length,encoder_hidden_size]
            Current_decoder_output: [decoder_hidden_size] 
            Attention_model: 'Biaffine' or 'Dotproduct' 
            
        Output:
            attention_weights: [1,length]
            log_attention_weights: [1,length]
        '''

        self.atten_model = atten_model
        self.weight1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.weight2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, cur_decoder_output):

        if self.atten_model == 'Biaffine':

            EW1_temp = self.weight1(encoder_outputs)
            EW1 = torch.matmul(EW1_temp, cur_decoder_output).unsqueeze(1)
            EW2 = self.weight2(encoder_outputs)
            bi_affine = EW1 + EW2
            bi_affine = bi_affine.permute(1, 0)

            # Obtain attention weights and logits (to compute loss)
            atten_weights = F.softmax(bi_affine, 0)
            log_atten_weights = F.log_softmax(bi_affine + 1e-6, 0)

        elif self.atten_model == 'Dotproduct':

            dot_prod = torch.matmul(encoder_outputs, cur_decoder_output).unsqueeze(0)
            # Obtain attention weights and logits (to compute loss)
            atten_weights = F.softmax(dot_prod, 1)
            log_atten_weights = F.log_softmax(dot_prod + 1e-6, 1)

        # Return attention weights and log attention weights
        return atten_weights, log_atten_weights


class LabelClassifier(nn.Module):
    def __init__(self, input_size, classifier_hidden_size, classes_label=41,
                 bias=True, dropout=0.5):

        super(LabelClassifier, self).__init__()
        '''
        
        Args:
            input_size: input size
            classifier_hidden_size: project input to classifier space
            classes_label: corresponding to 39 relations we have. 
                           (e.g. Contrast_NN)
            bias: If set to False, the layer will not learn an additive bias.
                Default: True               

        Input:
            input_left: [1,input_size]
            input_right: [1,input_size]
        Output:
            relation_weights: [1,classes_label]
            log_relation_weights: [1,classes_label]
            
        '''
        self.classifier_hidden_size = classifier_hidden_size
        self.labelspace_left = nn.Linear(input_size, classifier_hidden_size, bias=False)
        self.labelspace_right = nn.Linear(input_size, classifier_hidden_size, bias=False)
        self.weight_left = nn.Linear(classifier_hidden_size, classes_label, bias=False)
        self.weight_right = nn.Linear(classifier_hidden_size, classes_label, bias=False)
        self.nnDropout = nn.Dropout(dropout)

        self.classifier_hidden_size = classifier_hidden_size

        if bias:
            self.weight_bilateral = nn.Bilinear(classifier_hidden_size, classifier_hidden_size, classes_label)
        else:
            self.weight_bilateral = nn.Bilinear(classifier_hidden_size, classifier_hidden_size, classes_label, bias=False)

    def forward(self, input_left, input_right):

        left_size = input_left.size()
        right_size = input_right.size()

        labelspace_left = F.elu(self.labelspace_left(input_left))
        labelspace_right = F.elu(self.labelspace_right(input_right))

        # Apply dropout
        union = torch.cat((labelspace_left, labelspace_right), 1)
        union = self.nnDropout(union)
        labelspace_left = union[:, :self.classifier_hidden_size]
        labelspace_right = union[:, self.classifier_hidden_size:]

        output = (self.weight_bilateral(labelspace_left, labelspace_right) +
                  self.weight_left(labelspace_left) + self.weight_right(labelspace_right))

        # Obtain relation weights and log relation weights (for loss) 
        relation_weights = F.softmax(output, 1)
        log_relation_weights = F.log_softmax(output + 1e-6, 1)

        return relation_weights, log_relation_weights


class Segmenter_pointer(nn.Module):

    def __init__(self, hidden_size, atten_model=None, decoder_input_size=None, rnn_layers=None, dropout_d=None):
        super(Segmenter_pointer, self).__init__()

        self.hidden_size = hidden_size
        self.pointer = PointerAtten(atten_model, hidden_size)
        self.encoder = nn.GRU(hidden_size, int(hidden_size / 2), num_layers=1, batch_first=True, dropout=0.2, bidirectional=True)
        self.decoder = DecoderRNN(decoder_input_size, hidden_size, rnn_layers, dropout_d)
        self.loss_function = nn.NLLLoss()

    def forward(self):
        raise RuntimeError('Segmenter does not have forward process.')

    def train_segment_loss(self, word_embeddings, edu_breaks):
        outputs, last_hidden = self.encoder(word_embeddings.unsqueeze(0))
        outputs = outputs.squeeze()
        cur_decoder_hidden = outputs[-1, :].unsqueeze(0).unsqueeze(0)
        edu_breaks = [0] + edu_breaks
        total_loss = torch.FloatTensor([0.0]).cuda()
        for step, start_index in enumerate(edu_breaks[:-1]):
            cur_decoder_output, cur_decoder_hidden = self.decoder(outputs[start_index].unsqueeze(0).unsqueeze(0), last_hidden=cur_decoder_hidden)

            _, log_atten_weights = self.pointer(outputs[start_index:], cur_decoder_output.squeeze(0).squeeze(0))
            cur_ground_index = torch.tensor([edu_breaks[step + 1] - start_index]).cuda()
            total_loss = total_loss + self.loss_function(log_atten_weights, cur_ground_index)

        return total_loss

    def test_segment_loss(self, word_embeddings, edu_breaks):
        outputs, last_hidden = self.encoder(word_embeddings.unsqueeze(0))
        outputs = outputs.squeeze()
        cur_decoder_hidden = outputs[-1, :].unsqueeze(0).unsqueeze(0)
        start_index = 0
        predict_segment = []
        sentence_length = outputs.shape[0]
        while start_index < sentence_length:
            cur_decoder_output, cur_decoder_hidden = self.decoder(outputs[start_index].unsqueeze(0).unsqueeze(0), last_hidden=cur_decoder_hidden)
            atten_weights, log_atten_weights = self.pointer(outputs[start_index:], cur_decoder_output.squeeze(0).squeeze(0))
            _, top_index_seg = atten_weights.topk(1)

            seg_index = int(top_index_seg[0][0]) + start_index
            predict_segment.append(seg_index)
            start_index = seg_index + 1

        if predict_segment[-1] != sentence_length - 1:
            predict_segment.append(sentence_length - 1)

        return predict_segment


class Segmenter(nn.Module):
    def __init__(self, hidden_size):
        super(Segmenter, self).__init__()

        self.hidden_size = hidden_size
        self.drop_out = nn.Dropout(p=0.5)
        self.linear = nn.Linear(hidden_size, 2)
        self.linear_start = nn.Linear(hidden_size, 2)
        self.loss_function = nn.CrossEntropyLoss(weight=torch.Tensor([1.0, 10.0]).cuda())

    def forward(self):
        raise RuntimeError('Segmenter does not have forward process.')

    def train_segment_loss(self, word_embeddings, edu_breaks):
        edu_break_target = [0, ] * word_embeddings.size(0)
        edu_start_target = [0, ] * word_embeddings.size(0)

        for i in edu_breaks:
            edu_break_target[i] = 1
        edu_start_target[0] = 1
        for i in edu_breaks[:-1]:
            edu_start_target[i + 1] = 1

        edu_break_target = torch.LongTensor(edu_break_target).cuda()
        edu_start_target = torch.LongTensor(edu_start_target).cuda()
        outputs = self.linear(self.drop_out(word_embeddings))
        start_outputs = self.linear_start(self.drop_out(word_embeddings))

        if config.if_edu_start_loss:
            total_loss = self.loss_function(outputs, edu_break_target) + self.loss_function(start_outputs, edu_start_target)
        else:
            total_loss = self.loss_function(outputs, edu_break_target)
        return total_loss

    def test_segment_loss(self, word_embeddings):
        outputs = self.linear(self.drop_out(word_embeddings))
        pred = torch.argmax(outputs, dim=1).detach().cpu().numpy().tolist()
        predict_segment = [i for i, k in enumerate(pred) if k == 1]

        if word_embeddings.size(0) - 1 not in predict_segment:
            predict_segment.append(word_embeddings.size(0) - 1)

        return predict_segment
