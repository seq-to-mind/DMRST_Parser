import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from module import EncoderRNN, DecoderRNN, PointerAtten, LabelClassifier, Segmenter
from DataHandler import get_RelationAndNucleus
from random import randint
import config


class ParsingNet(nn.Module):
    def __init__(self, language_model, word_dim=768, hidden_size=768, decoder_input_size=768,
                 atten_model="Dotproduct", classifier_input_size=768, classifier_hidden_size=768, classes_label=42, classifier_bias=True,
                 rnn_layers=1, dropout_e=0.5, dropout_d=0.5, dropout_c=0.5, bert_tokenizer=None):

        super(ParsingNet, self).__init__()
        '''
        Args:
            batch_size: batch size
            word_dim: word embedding dimension 
            hidden_size: hidden size of encoder and decoder 
            decoder_input_size: input dimension of decoder
            atten_model: pointer attention machanisam, 'Dotproduct' or 'Biaffine' 
            device: device that our model is running on 
            classifier_input_size: input dimension of labels classifier 
            classifier_hidden_size: classifier hidden space
            classes_label: relation(label) number, default = 39
            classifier_bias: bilinear bias in classifier, default = True
            rnn_layers: encoder and decoder layer number
            dropout: dropout rate
        '''
        self.word_dim = word_dim
        self.hidden_size = hidden_size
        self.decoder_input_size = decoder_input_size
        self.classifier_input_size = classifier_input_size
        self.classifier_hidden_size = classifier_hidden_size
        self.classes_label = classes_label
        self.classifier_bias = classifier_bias
        self.rnn_layers = rnn_layers
        self.segmenter = Segmenter(hidden_size)
        self.encoder = EncoderRNN(language_model, word_dim, hidden_size, config.enc_rnn_layer_num, dropout_e, bert_tokenizer=bert_tokenizer, segmenter=self.segmenter)
        self.decoder = DecoderRNN(decoder_input_size, hidden_size, rnn_layers, dropout_d)
        self.pointer = PointerAtten(atten_model, hidden_size)
        self.getlabel = LabelClassifier(classifier_input_size, classifier_hidden_size, classes_label, bias=True, dropout=dropout_c)

    def forward(self):
        raise RuntimeError('Parsing Network does not have forward process.')

    def TrainingLoss(self, input_sentence, EDU_breaks, LabelIndex, ParsingIndex, DecoderInputIndex, ParentsIndex, SiblingIndex):

        # Obtain encoder outputs and last hidden states
        EncoderOutputs, Last_Hiddenstates, total_edu_loss, _ = self.encoder(input_sentence, EDU_breaks)

        Label_LossFunction = nn.NLLLoss()
        Span_LossFunction = nn.NLLLoss()

        Loss_label_batch = 0
        Loss_tree_batch = torch.FloatTensor([0.0]).cuda()
        Loop_label_batch = 0
        Loop_tree_batch = 0

        batch_size = len(LabelIndex)
        for i in range(batch_size):

            cur_LabelIndex = LabelIndex[i]
            cur_LabelIndex = torch.tensor(cur_LabelIndex)
            cur_LabelIndex = cur_LabelIndex.cuda()
            cur_ParsingIndex = ParsingIndex[i]
            cur_DecoderInputIndex = DecoderInputIndex[i]
            cur_ParentsIndex = ParentsIndex[i]
            cur_SiblingIndex = SiblingIndex[i]

            if len(EDU_breaks[i]) == 1:

                continue

            elif len(EDU_breaks[i]) == 2:

                # Obtain the encoded representations. The dimension: [2,hidden_size]
                cur_EncoderOutputs = EncoderOutputs[i][:len(EDU_breaks[i])]

                # Use the last hidden state of a span to predict the relation between these two span.
                input_left = cur_EncoderOutputs[0].unsqueeze(0)
                input_right = cur_EncoderOutputs[1].unsqueeze(0)

                _, log_relation_weights = self.getlabel(input_left, input_right)

                Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_LabelIndex)
                Loop_label_batch = Loop_label_batch + 1

            else:
                cur_EncoderOutputs = EncoderOutputs[i][:len(EDU_breaks[i])]
                cur_Last_Hiddenstates = Last_Hiddenstates[:, i, :].unsqueeze(1)
                cur_decoder_hidden = cur_Last_Hiddenstates.contiguous()

                EDU_index = [x for x in range(len(cur_EncoderOutputs))]
                stacks = ['__StackRoot__', EDU_index]

                for j in range(len(cur_DecoderInputIndex)):

                    if stacks[-1] != '__StackRoot__':
                        stack_head = stacks[-1]

                        if len(stack_head) < 3:

                            # Will remove this from stacks after compute the relation between these two EDUS
                            input_left = cur_EncoderOutputs[cur_ParsingIndex[j]].unsqueeze(0)
                            input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)

                            assert cur_ParsingIndex[j] < stack_head[-1]

                            # keep the last hidden state consistent.
                            cur_decoder_input = torch.mean(cur_EncoderOutputs[stack_head], keepdim=True, dim=0).unsqueeze(0)
                            cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input, last_hidden=cur_decoder_hidden)

                            _, log_relation_weights = self.getlabel(input_left, input_right)
                            Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_LabelIndex[j].unsqueeze(0))

                            del stacks[-1]
                            Loop_label_batch = Loop_label_batch + 1

                        else:  # Length of stack_head >= 3

                            # Compute Tree Loss
                            # We don't attend to the last EDU of a span to be parsed
                            cur_decoder_input = torch.mean(cur_EncoderOutputs[stack_head], keepdim=True, dim=0).unsqueeze(0)

                            # Predict the parsing tree break
                            cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input, last_hidden=cur_decoder_hidden)

                            _, log_atten_weights = self.pointer(cur_EncoderOutputs[stack_head[:-1]], cur_decoder_output.squeeze(0).squeeze(0))
                            cur_ground_index = torch.tensor([int(cur_ParsingIndex[j]) - int(stack_head[0])])
                            cur_ground_index = cur_ground_index.cuda()
                            Loss_tree_batch = Loss_tree_batch + Span_LossFunction(log_atten_weights, cur_ground_index)

                            # Compute Classifier Loss
                            """ merge edu level representation for left and right siblings START """
                            if config.average_edu_level is True:
                                input_left = torch.mean(cur_EncoderOutputs[stack_head[0]:cur_ParsingIndex[j] + 1, :], keepdim=True, dim=0)
                                input_right = torch.mean(cur_EncoderOutputs[cur_ParsingIndex[j] + 1: stack_head[-1] + 1, :], keepdim=True, dim=0)
                            else:
                                input_left = cur_EncoderOutputs[cur_ParsingIndex[j]].unsqueeze(0)
                                input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)
                            """ merge edu level representation for left and right siblings END """

                            _, log_relation_weights = self.getlabel(input_left, input_right)
                            Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_LabelIndex[j].unsqueeze(0))

                            # Stacks stuff
                            stack_left = stack_head[:(cur_ParsingIndex[j] - stack_head[0] + 1)]
                            stack_right = stack_head[(cur_ParsingIndex[j] - stack_head[0] + 1):]
                            del stacks[-1]
                            Loop_label_batch = Loop_label_batch + 1
                            Loop_tree_batch = Loop_tree_batch + 1

                            # Remove ONE-EDU part, TWO-EDU span will be removed after classifier in next step
                            if len(stack_right) > 1:
                                stacks.append(stack_right)
                            if len(stack_left) > 1:
                                stacks.append(stack_left)

        Loss_label_batch = Loss_label_batch / Loop_label_batch

        if Loop_tree_batch == 0:
            Loop_tree_batch = 1

        Loss_tree_batch = Loss_tree_batch / Loop_tree_batch

        return Loss_tree_batch, Loss_label_batch, total_edu_loss


    def TestingLoss(self, input_sentence, input_EDU_breaks, LabelIndex, ParsingIndex, GenerateTree, use_pred_segmentation):
        '''
            Input:
                input_sentence: [batch_size, length]
                input_EDU_breaks: e.g. [[2,4,6,9],[2,5,8,10,13],[6,8],[6]]
                LabelIndex: e.g. [[0,3,32],[20,11,14,19],[20],[],]
                ParsingIndex: e.g. [[1,2,0],[3,2,0,1],[0],[]]
            Output: log_atten_weights
                Average loss of tree in a batch
                Average loss of relation in a batch
        '''
        # Obtain encoder outputs and last hidden states
        EncoderOutputs, Last_Hiddenstates, _, predict_edu_breaks = self.encoder(input_sentence, input_EDU_breaks, is_test=use_pred_segmentation)

        if use_pred_segmentation:
            EDU_breaks = predict_edu_breaks
            if LabelIndex is None and ParsingIndex is None:
                LabelIndex = [[0, ] * (len(i) - 1) for i in EDU_breaks]
                ParsingIndex = [[0, ] * (len(i) - 1) for i in EDU_breaks]
        else:
            EDU_breaks = input_EDU_breaks

        Label_LossFunction = nn.NLLLoss()
        Span_LossFunction = nn.NLLLoss()

        Loss_label_batch = torch.FloatTensor([0.0]).cuda()
        Loss_tree_batch = torch.FloatTensor([0.0]).cuda()
        Loop_label_batch = 0
        Loop_tree_batch = 0

        Label_batch = []
        Tree_batch = []

        if GenerateTree:
            SPAN_batch = []

        for i in range(len(EDU_breaks)):

            cur_label = []
            cur_tree = []

            cur_LabelIndex = LabelIndex[i]
            cur_LabelIndex = torch.tensor(cur_LabelIndex)
            cur_LabelIndex = cur_LabelIndex.cuda()
            cur_ParsingIndex = ParsingIndex[i]

            if len(EDU_breaks[i]) == 1:

                # For a sentence containing only ONE EDU, it has no corresponding relation label and parsing tree break.
                Tree_batch.append([])
                Label_batch.append([])

                if GenerateTree:
                    SPAN_batch.append(['NONE'])

            elif len(EDU_breaks[i]) == 2:

                # Obtain the encoded representations, the dimension: [2, hidden_size]
                cur_EncoderOutputs = EncoderOutputs[i][:len(EDU_breaks[i])]

                #  Directly run the classifier to obtain predicted label
                input_left = cur_EncoderOutputs[0].unsqueeze(0)
                input_right = cur_EncoderOutputs[1].unsqueeze(0)
                relation_weights, log_relation_weights = self.getlabel(input_left, input_right)
                _, topindex = relation_weights.topk(1)
                LabelPredict = int(topindex[0][0])
                Tree_batch.append([0])
                Label_batch.append([LabelPredict])

                if use_pred_segmentation is False:
                    Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_LabelIndex)

                Loop_label_batch = Loop_label_batch + 1

                if GenerateTree:
                    # Generate a span structure: e.g. (1:Nucleus=span:8,9:Satellite=Attribution:12)
                    Nuclearity_left, Nuclearity_right, Relation_left, Relation_right = get_RelationAndNucleus(LabelPredict)

                    Span = '(1:' + str(Nuclearity_left) + '=' + str(Relation_left) + \
                           ':1,2:' + str(Nuclearity_right) + '=' + str(Relation_right) + ':2)'
                    SPAN_batch.append([Span])

            else:
                # Obtain the encoded representations, the dimension: [num_EDU, hidden_size]
                cur_EncoderOutputs = EncoderOutputs[i][:len(EDU_breaks[i])]

                EDU_index = [x for x in range(len(cur_EncoderOutputs))]
                stacks = ['__StackRoot__', EDU_index]

                # # Obtain last hidden state
                cur_Last_Hiddenstates = Last_Hiddenstates[:, i, :].unsqueeze(1)
                cur_decoder_hidden = cur_Last_Hiddenstates.contiguous()

                LoopIndex = 0

                if GenerateTree:
                    Span = ''

                tmp_decode_step = -1

                while stacks[-1] != '__StackRoot__':
                    stack_head = stacks[-1]

                    if len(stack_head) < 3:

                        tmp_decode_step += 1
                        # Predict relation label
                        input_left = cur_EncoderOutputs[stack_head[0]].unsqueeze(0)
                        input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)
                        # assert stack_head[0] < stack_head[-1]

                        relation_weights, log_relation_weights = self.getlabel(input_left, input_right)
                        _, topindex = relation_weights.topk(1)
                        LabelPredict = int(topindex[0][0])
                        cur_label.append(LabelPredict)

                        # For 2 EDU case, we directly point the first EDU as the current parsing tree break
                        cur_tree.append(stack_head[0])

                        # keep the last hidden state consistent.
                        cur_decoder_input = torch.mean(cur_EncoderOutputs[stack_head], keepdim=True, dim=0).unsqueeze(0)
                        cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input, last_hidden=cur_decoder_hidden)

                        # Align ground true label
                        if LoopIndex > (len(cur_ParsingIndex) - 1):
                            cur_Label_true = cur_LabelIndex[-1]
                        else:
                            cur_Label_true = cur_LabelIndex[LoopIndex]

                        if use_pred_segmentation is False:
                            Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_Label_true.unsqueeze(0))

                        Loop_label_batch = Loop_label_batch + 1
                        LoopIndex = LoopIndex + 1
                        del stacks[-1]

                        if GenerateTree:
                            # To generate a tree structure
                            Nuclearity_left, Nuclearity_right, Relation_left, Relation_right = get_RelationAndNucleus(LabelPredict)

                            cur_span = '(' + str(stack_head[0] + 1) + ':' + str(Nuclearity_left) + '=' + str(Relation_left) + \
                                       ':' + str(stack_head[0] + 1) + ',' + str(stack_head[-1] + 1) + ':' + str(Nuclearity_right) + '=' + \
                                       str(Relation_right) + ':' + str(stack_head[-1] + 1) + ')'

                            Span = Span + ' ' + cur_span

                    else:  # Length of stack_head >= 3

                        tmp_decode_step += 1

                        # Alternative way is to take the last one as the input. You need to prepare data accordingly for training.
                        cur_decoder_input = torch.mean(cur_EncoderOutputs[stack_head], keepdim=True, dim=0).unsqueeze(0)

                        # Predict the parsing tree break
                        cur_decoder_output, cur_decoder_hidden = self.decoder(cur_decoder_input, last_hidden=cur_decoder_hidden)
                        atten_weights, log_atten_weights = self.pointer(cur_EncoderOutputs[stack_head[:-1]], cur_decoder_output.squeeze(0).squeeze(0))

                        _, topindex_tree = atten_weights.topk(1)
                        TreePredict = int(topindex_tree[0][0]) + stack_head[0]

                        cur_tree.append(TreePredict)

                        """ merge edu level representation for left and right siblings START """
                        if config.average_edu_level is True:
                            input_left = torch.mean(cur_EncoderOutputs[stack_head[0]:TreePredict + 1, :], keepdim=True, dim=0)
                            input_right = torch.mean(cur_EncoderOutputs[TreePredict + 1: stack_head[-1] + 1, :], keepdim=True, dim=0)
                        else:
                            input_left = cur_EncoderOutputs[TreePredict].unsqueeze(0)
                            input_right = cur_EncoderOutputs[stack_head[-1]].unsqueeze(0)
                        """ merge edu level representation for left and right siblings END """

                        relation_weights, log_relation_weights = self.getlabel(input_left, input_right)
                        _, topindex_label = relation_weights.topk(1)
                        LabelPredict = int(topindex_label[0][0])
                        cur_label.append(LabelPredict)

                        # Align ground true label and tree
                        if LoopIndex > (len(cur_ParsingIndex) - 1):
                            cur_Label_true = cur_LabelIndex[-1]
                            cur_Tree_true = cur_ParsingIndex[-1]
                        else:
                            cur_Label_true = cur_LabelIndex[LoopIndex]
                            cur_Tree_true = cur_ParsingIndex[LoopIndex]

                        temp_ground = max(0, (int(cur_Tree_true) - int(stack_head[0])))
                        if temp_ground >= (len(stack_head) - 1):
                            temp_ground = stack_head[-2] - stack_head[0]
                        # Compute Tree Loss
                        cur_ground_index = torch.tensor([temp_ground])
                        cur_ground_index = cur_ground_index.cuda()

                        if use_pred_segmentation is False:
                            Loss_tree_batch = Loss_tree_batch + Span_LossFunction(log_atten_weights, cur_ground_index)
                            Loss_label_batch = Loss_label_batch + Label_LossFunction(log_relation_weights, cur_Label_true.unsqueeze(0))

                        # Stacks stuff
                        stack_left = stack_head[:(TreePredict - stack_head[0] + 1)]
                        stack_right = stack_head[(TreePredict - stack_head[0] + 1):]

                        del stacks[-1]
                        Loop_label_batch = Loop_label_batch + 1
                        Loop_tree_batch = Loop_tree_batch + 1
                        LoopIndex = LoopIndex + 1

                        # Remove ONE-EDU part
                        if len(stack_right) > 1:
                            stacks.append(stack_right)
                        if len(stack_left) > 1:
                            stacks.append(stack_left)

                        if GenerateTree:
                            # Generate a span structure: e.g. (1:Nucleus=span:8,9:Satellite=Attribution:12)
                            Nuclearity_left, Nuclearity_right, Relation_left, Relation_right = \
                                get_RelationAndNucleus(LabelPredict)

                            cur_span = '(' + str(stack_head[0] + 1) + ':' + str(Nuclearity_left) + '=' + str(Relation_left) + \
                                       ':' + str(TreePredict + 1) + ',' + str(TreePredict + 2) + ':' + str(Nuclearity_right) + '=' + \
                                       str(Relation_right) + ':' + str(stack_head[-1] + 1) + ')'
                            Span = Span + ' ' + cur_span

                Tree_batch.append(cur_tree)
                Label_batch.append(cur_label)
                if GenerateTree:
                    SPAN_batch.append([Span.strip()])

        if Loop_label_batch == 0:
            Loop_label_batch = 1

        Loss_label_batch = Loss_label_batch / Loop_label_batch

        if Loop_tree_batch == 0:
            Loop_tree_batch = 1

        Loss_tree_batch = Loss_tree_batch / Loop_tree_batch

        Loss_label_batch = Loss_label_batch.detach().cpu().numpy()
        Loss_tree_batch = Loss_tree_batch.detach().cpu().numpy()

        merged_label_gold = []
        for tmp_i in LabelIndex:
            merged_label_gold.extend(tmp_i)

        merged_label_pred = []
        for tmp_i in Label_batch:
            merged_label_pred.extend(tmp_i)

        # assert len(merged_label_gold) == len(merged_label_pred)
        return Loss_tree_batch, Loss_label_batch, (SPAN_batch if GenerateTree else None), (merged_label_gold, merged_label_pred), EDU_breaks
