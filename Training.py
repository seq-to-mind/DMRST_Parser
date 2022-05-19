import math
import torch.optim as optim
import numpy as np
import torch
import random
import torch.nn as nn
import copy
import os
from Metric import getBatchMeasure, getMicroMeasure, getMacroMeasure
import config
import pickle


def get_batches(data, batch_size):
    batches = []
    for i in range(len(data) // batch_size + bool(len(data) % batch_size)):
        batches.append(data[i * batch_size:(i + 1) * batch_size])

    return batches


def getBatchData_training(InputSentences, EDUBreaks, DecoderInput, RelationLabel,
                          ParsingBreaks, GoldenMetric, ParentsIndex, Sibling, batch_size, batch_idx_list):
    # change them into np.array
    InputSentences = np.array(InputSentences, dtype=object)
    EDUBreaks = np.array(EDUBreaks, dtype=object)
    DecoderInput = np.array(DecoderInput, dtype=object)
    RelationLabel = np.array(RelationLabel, dtype=object)
    ParsingBreaks = np.array(ParsingBreaks, dtype=object)
    GoldenMetric = np.array(GoldenMetric, dtype=object)
    ParentsIndex = np.array(ParentsIndex, dtype=object)
    Sibling = np.array(Sibling, dtype=object)

    if len(DecoderInput) < batch_size:
        batch_size = len(DecoderInput)

    if config.random_with_pre_shuffle:
        IndexSelected = batch_idx_list
    else:
        IndexSelected = random.sample(range(len(DecoderInput)), batch_size)

    # print(IndexSelected)

    # Get batch data
    InputSentences_batch = copy.deepcopy(InputSentences[IndexSelected])
    EDUBreaks_batch = copy.deepcopy(EDUBreaks[IndexSelected])
    DecoderInput_batch = copy.deepcopy(DecoderInput[IndexSelected])
    RelationLabel_batch = copy.deepcopy(RelationLabel[IndexSelected])
    ParsingBreaks_batch = copy.deepcopy(ParsingBreaks[IndexSelected])
    GoldenMetric_batch = copy.deepcopy(GoldenMetric[IndexSelected])
    ParentsIndex_batch = copy.deepcopy(ParentsIndex[IndexSelected])
    Sibling_batch = copy.deepcopy(Sibling[IndexSelected])

    # Get sorted
    Lengths_batch = np.array([len(sent) for sent in InputSentences_batch])
    idx = np.argsort(Lengths_batch)
    idx = idx[::-1]

    # Convert them back to list
    InputSentences_batch = InputSentences_batch[idx].tolist()
    EDUBreaks_batch = EDUBreaks_batch[idx].tolist()
    DecoderInput_batch = DecoderInput_batch[idx].tolist()
    RelationLabel_batch = RelationLabel_batch[idx].tolist()
    ParsingBreaks_batch = ParsingBreaks_batch[idx].tolist()
    GoldenMetric_batch = GoldenMetric_batch[idx].tolist()
    ParentsIndex_batch = ParentsIndex_batch[idx].tolist()
    Sibling_batch = Sibling_batch[idx].tolist()

    return InputSentences_batch, EDUBreaks_batch, DecoderInput_batch, RelationLabel_batch, \
           ParsingBreaks_batch, GoldenMetric_batch, ParentsIndex_batch, Sibling_batch


def getBatchData(InputSentences, EDUBreaks, DecoderInput, RelationLabel,
                 ParsingBreaks, GoldenMetric, batch_size):
    InputSentences = np.array(InputSentences, dtype=object)
    EDUBreaks = np.array(EDUBreaks, dtype=object)
    DecoderInput = np.array(DecoderInput, dtype=object)
    RelationLabel = np.array(RelationLabel, dtype=object)
    ParsingBreaks = np.array(ParsingBreaks, dtype=object)
    GoldenMetric = np.array(GoldenMetric, dtype=object)

    if len(DecoderInput) < batch_size:
        batch_size = len(DecoderInput)

    assert len(DecoderInput) == batch_size

    IndexSelected = random.sample(range(len(DecoderInput)), batch_size)

    # Get batch data
    InputSentences_batch = copy.deepcopy(InputSentences[IndexSelected])
    EDUBreaks_batch = copy.deepcopy(EDUBreaks[IndexSelected])
    DecoderInput_batch = copy.deepcopy(DecoderInput[IndexSelected])
    RelationLabel_batch = copy.deepcopy(RelationLabel[IndexSelected])
    ParsingBreaks_batch = copy.deepcopy(ParsingBreaks[IndexSelected])
    GoldenMetric_batch = copy.deepcopy(GoldenMetric[IndexSelected])

    # Get sorted
    Lengths_batch = np.array([len(sent) for sent in InputSentences_batch])
    idx = np.argsort(Lengths_batch)
    idx = idx[::-1]

    # Convert them back to list
    InputSentences_batch = InputSentences_batch[idx].tolist()
    EDUBreaks_batch = EDUBreaks_batch[idx].tolist()
    DecoderInput_batch = DecoderInput_batch[idx].tolist()
    RelationLabel_batch = RelationLabel_batch[idx].tolist()
    ParsingBreaks_batch = ParsingBreaks_batch[idx].tolist()
    GoldenMetric_batch = GoldenMetric_batch[idx].tolist()

    return InputSentences_batch, EDUBreaks_batch, DecoderInput_batch, RelationLabel_batch, ParsingBreaks_batch, GoldenMetric_batch


class Train(object):
    def __init__(self, model, Tr_Input_sentences, Tr_EDUBreaks, Tr_DecoderInput,
                 Tr_RelationLabel, Tr_ParsingBreaks, Tr_GoldenMetric,
                 Tr_ParentsIndex, Tr_SiblingIndex,
                 Test_InputSentences, Test_EDUBreaks, Test_DecoderInput,
                 Test_RelationLabel, Test_ParsingBreaks, Test_GoldenMetric,
                 batch_size, eval_size, epoch, lr, lr_decay_epoch, weight_decay,
                 save_path):

        self.model = model
        self.Tr_Input_sentences = Tr_Input_sentences
        self.Tr_EDUBreaks = Tr_EDUBreaks
        self.Tr_DecoderInput = Tr_DecoderInput
        self.Tr_RelationLabel = Tr_RelationLabel
        self.Tr_ParsingBreaks = Tr_ParsingBreaks
        self.Tr_GoldenMetric = Tr_GoldenMetric
        self.Tr_ParentsIndex = Tr_ParentsIndex
        self.Tr_SiblingIndex = Tr_SiblingIndex
        self.Test_InputSentences = Test_InputSentences
        self.Test_EDUBreaks = Test_EDUBreaks
        self.Test_DecoderInput = Test_DecoderInput
        self.Test_RelationLabel = Test_RelationLabel
        self.Test_ParsingBreaks = Test_ParsingBreaks
        self.Test_GoldenMetric = Test_GoldenMetric
        self.batch_size = batch_size
        self.eval_size = eval_size
        self.epoch = epoch
        self.lr = lr
        self.lr_decay_epoch = lr_decay_epoch
        self.weight_decay = weight_decay
        self.save_path = save_path

    def getTrainingEval(self):

        # Obtain eval_size samples of training data to evaluate the model in every epoch

        # Convert to np.array
        Tr_Input_sentences = np.array(self.Tr_Input_sentences, dtype=object)
        Tr_EDUBreaks = np.array(self.Tr_EDUBreaks, dtype=object)
        Tr_DecoderInput = np.array(self.Tr_DecoderInput, dtype=object)
        Tr_RelationLabel = np.array(self.Tr_RelationLabel, dtype=object)
        Tr_ParsingBreaks = np.array(self.Tr_ParsingBreaks, dtype=object)
        Tr_GoldenMetric = np.array(self.Tr_GoldenMetric, dtype=object)

        if config.use_dev_set is True:
            IndexSelected = [i for i in range(len(self.Tr_ParsingBreaks))][-config.dev_set_size:]
        else:
            IndexSelected = random.sample(range(len(self.Tr_ParsingBreaks)), self.eval_size)

        DevTr_Input_sentences = Tr_Input_sentences[IndexSelected].tolist()
        DevTr_EDUBreaks = Tr_EDUBreaks[IndexSelected].tolist()
        DevTr_DecoderInput = Tr_DecoderInput[IndexSelected].tolist()
        DevTr_RelationLabel = Tr_RelationLabel[IndexSelected].tolist()
        DevTr_ParsingBreaks = Tr_ParsingBreaks[IndexSelected].tolist()
        DevTr_GoldenMetric = Tr_GoldenMetric[IndexSelected].tolist()

        return DevTr_Input_sentences, DevTr_EDUBreaks, DevTr_DecoderInput, DevTr_RelationLabel, \
               DevTr_ParsingBreaks, DevTr_GoldenMetric

    def getAccuracy(self, Input_sentences, EDUBreaks, DecoderInput, RelationLabel, ParsingBreaks, GoldenMetric,
                    use_pred_segmentation, use_org_Parseval):

        LoopNeeded = int(np.ceil(len(EDUBreaks) / self.batch_size))

        Loss_tree_all = []
        Loss_label_all = []
        correct_span = 0
        correct_relation = 0
        correct_nuclearity = 0
        correct_full = 0
        no_system = 0
        no_golden = 0
        no_gold_seg = 0
        no_pred_seg = 0
        no_correct_seg = 0

        # Macro
        correct_span_list = []
        correct_relation_list = []
        correct_nuclearity_list = []
        no_system_list = []
        no_golden_list = []

        all_label_gold = []
        all_label_pred = []

        for loop in range(LoopNeeded):

            StartPosition = loop * self.batch_size
            EndPosition = (loop + 1) * self.batch_size
            if EndPosition > len(EDUBreaks):
                EndPosition = len(EDUBreaks)

            InputSentences_batch, EDUBreaks_batch, _, RelationLabel_batch, ParsingBreaks_batch, GoldenMetric_batch = \
                getBatchData(Input_sentences[StartPosition:EndPosition],
                             EDUBreaks[StartPosition:EndPosition],
                             DecoderInput[StartPosition:EndPosition],
                             RelationLabel[StartPosition:EndPosition],
                             ParsingBreaks[StartPosition:EndPosition],
                             GoldenMetric[StartPosition:EndPosition], self.batch_size)

            Loss_tree_batch, Loss_label_batch, SPAN_batch, Label_Tuple_batch, predict_EDU_breaks = self.model.TestingLoss(
                InputSentences_batch, EDUBreaks_batch, RelationLabel_batch,
                ParsingBreaks_batch, GenerateTree=True, use_pred_segmentation=use_pred_segmentation)

            all_label_gold.extend(Label_Tuple_batch[0])
            all_label_pred.extend(Label_Tuple_batch[1])

            Loss_tree_all.append(Loss_tree_batch)
            Loss_label_all.append(Loss_label_batch)

            correct_span_batch, correct_relation_batch, correct_nuclearity_batch, correct_full_batch, no_system_batch, no_golden_batch, \
            correct_span_batch_list, correct_relation_batch_list, correct_nuclearity_batch_list, \
            no_system_batch_list, no_golden_batch_list, segment_results_list = getBatchMeasure(SPAN_batch,
                                                                                               GoldenMetric_batch,
                                                                                               predict_EDU_breaks,
                                                                                               EDUBreaks_batch,
                                                                                               use_org_Parseval)

            correct_span = correct_span + correct_span_batch
            correct_relation = correct_relation + correct_relation_batch
            correct_nuclearity = correct_nuclearity + correct_nuclearity_batch
            correct_full = correct_full + correct_full_batch
            no_system = no_system + no_system_batch
            no_golden = no_golden + no_golden_batch
            no_gold_seg += segment_results_list[0]
            no_pred_seg += segment_results_list[1]
            no_correct_seg += segment_results_list[2]

            correct_span_list += correct_span_batch_list
            correct_relation_list += correct_relation_batch_list
            correct_nuclearity_list += correct_nuclearity_batch_list
            no_system_list += no_system_batch_list
            no_golden_list += no_golden_batch_list

        if config.use_micro_F1:
            span_points, relation_points, nuclearity_points, F1_Full, segment_points = getMicroMeasure(correct_span,
                                                                                                       correct_relation,
                                                                                                       correct_nuclearity,
                                                                                                       correct_full,
                                                                                                       no_system,
                                                                                                       no_golden,
                                                                                                       no_gold_seg,
                                                                                                       no_pred_seg,
                                                                                                       no_correct_seg)
        else:
            span_points, relation_points, nuclearity_points = getMacroMeasure(correct_span_list, correct_relation_list,
                                                                              correct_nuclearity_list, no_system_list,
                                                                              no_golden_list)

        return np.mean(Loss_tree_all), np.mean(Loss_label_all), span_points, relation_points, nuclearity_points, F1_Full, segment_points

    def LearningRateAdjust(self, optimizer, epoch, lr_decay, lr_decay_epoch):

        if (epoch % lr_decay_epoch == 0) and (epoch != 0):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_decay

    def train(self):

        if config.different_learning_rate is True:
            bert_parameters_ids = list(map(id, self.model.encoder.language_model.parameters()))
            rest_parameters = filter(lambda p: id(p) not in bert_parameters_ids, self.model.parameters())
            bert_parameters = filter(lambda p: id(p) in bert_parameters_ids, self.model.parameters())

            # optimizer = optim.AdamW([{'params': filter(lambda p: p.requires_grad, bert_parameters), 'lr': self.lr * 0.2,
            #                           "weight_decay": 0.01, "eps": 1e-6},
            #                          {'params': filter(lambda p: p.requires_grad, rest_parameters), 'lr': self.lr,
            #                           "weight_decay": self.weight_decay}], betas=(0.9, 0.999))
            optimizer = optim.AdamW([{'params': filter(lambda p: p.requires_grad, bert_parameters), 'lr': 0.00002},
                                     {'params': filter(lambda p: p.requires_grad, rest_parameters), 'lr': 0.0001}])

        else:
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                    lr=self.lr, betas=(0.9, 0.999), weight_decay=self.weight_decay)

        if config.use_dev_set is True:
            iteration = int(np.ceil((len(self.Tr_ParsingBreaks) - config.dev_set_size) / self.batch_size))
        else:
            iteration = int(np.ceil(len(self.Tr_ParsingBreaks) / self.batch_size))

        try:
            os.mkdir(self.save_path)
        except:
            pass

        best_F_relation = 0
        best_F_span = 0

        label_loss_iter_list = []
        tree_loss_iter_list = []
        edu_loss_iter_list = []

        w_label, w_tree, w_edu = None, None, None
        dwa_T = 2.0

        for one_epoch in range(self.epoch):

            self.LearningRateAdjust(optimizer, one_epoch, 0.9, self.lr_decay_epoch)

            # rebuild shuffle strategy
            if config.use_dev_set is True:
                whole_iter_list = [i for i in range(len(self.Tr_ParsingBreaks))][:-config.dev_set_size]
            else:
                whole_iter_list = [i for i in range(len(self.Tr_ParsingBreaks))]

            if config.random_with_pre_shuffle is True:
                random.shuffle(whole_iter_list)
                print("whole iter list", whole_iter_list[:10], "max:min", max(whole_iter_list), min(whole_iter_list))

            for one_iter in range(iteration):
                if one_iter % config.iter_display_size == 0:
                    print("epoch:%d, iteration:%d" % (one_epoch, one_iter))
                batch_idx_list = whole_iter_list[one_iter * self.batch_size: (one_iter + 1) * self.batch_size]
                assert len(batch_idx_list) > 0

                InputSentences_batch, EDUBreaks_batch, DecoderInput_batch, \
                RelationLabel_batch, ParsingBreaks_batch, _, ParentsIndex_batch, \
                Sibling_batch = getBatchData_training(
                    self.Tr_Input_sentences, self.Tr_EDUBreaks,
                    self.Tr_DecoderInput, self.Tr_RelationLabel,
                    self.Tr_ParsingBreaks, self.Tr_GoldenMetric,
                    self.Tr_ParentsIndex, self.Tr_SiblingIndex, self.batch_size, batch_idx_list)

                self.model.zero_grad()
                Loss_tree_batch, Loss_label_batch, Loss_segment_batch = self.model.TrainingLoss(InputSentences_batch,
                                                                                                EDUBreaks_batch,
                                                                                                RelationLabel_batch,
                                                                                                ParsingBreaks_batch,
                                                                                                DecoderInput_batch,
                                                                                                ParentsIndex_batch,
                                                                                                Sibling_batch)

                if config.use_dwa_loss:
                    if len(label_loss_iter_list) > 2:

                        r_label = label_loss_iter_list[-1] / label_loss_iter_list[-2]
                        r_tree = tree_loss_iter_list[-1] / tree_loss_iter_list[-2]
                        r_edu = edu_loss_iter_list[-1] / edu_loss_iter_list[-2]

                        total_r = math.exp(r_label / dwa_T) + math.exp(r_tree / dwa_T) + math.exp(r_edu / dwa_T)

                        w_label = 3 * math.exp(r_label / dwa_T) / total_r
                        w_tree = 3 * math.exp(r_tree / dwa_T) / total_r
                        w_edu = 3 * math.exp(r_edu / dwa_T) / total_r

                        Loss = w_label * Loss_label_batch + w_tree * Loss_tree_batch + w_edu * Loss_segment_batch

                        label_loss_iter_list.append(Loss_label_batch)
                        tree_loss_iter_list.append(Loss_tree_batch)
                        edu_loss_iter_list.append(Loss_segment_batch)

                    else:
                        Loss = Loss_tree_batch + Loss_label_batch + Loss_segment_batch

                        label_loss_iter_list.append(Loss_label_batch)
                        tree_loss_iter_list.append(Loss_tree_batch)
                        edu_loss_iter_list.append(Loss_segment_batch)

                else:
                    Loss = Loss_tree_batch + Loss_label_batch + Loss_segment_batch

                Loss.backward()

                cur_loss = float(Loss.item())
                if one_iter % config.iter_display_size == 0:
                    print(float(Loss_tree_batch.item()), float(Loss_label_batch.item()), float(Loss_segment_batch.item()))
                    if w_edu:
                        print("lambda:", w_tree, w_label, w_edu)

                # To avoid gradient exploration
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                optimizer.step()

                torch.cuda.empty_cache()

            # Convert model to eval
            self.model.eval()

            # Obtain Training (devolopment) data
            DevTr_Input_sentences, DevTr_EDUBreaks, DevTr_DecoderInput, DevTr_RelationLabel, \
            DevTr_ParsingBreaks, DevTr_GoldenMetric = self.getTrainingEval()

            # Eval on training (devolopment) data
            LossTree_Trdev, LossLabel_Trdev, span_points_Trdev, relation_points_Trdev, \
            nuclearity_points_Trdev, F1_full_Trdev, segment_points_Trdev = self.getAccuracy(DevTr_Input_sentences,
                                                                                            DevTr_EDUBreaks,
                                                                                            DevTr_DecoderInput,
                                                                                            DevTr_RelationLabel,
                                                                                            DevTr_ParsingBreaks,
                                                                                            DevTr_GoldenMetric,
                                                                                            use_pred_segmentation=False,
                                                                                            use_org_Parseval=False)

            # Eval on Testing data
            LossTree_Test, LossLabel_Test, span_points_Test, relation_points_Test, \
            nuclearity_points_Test, F1_full_Test, segment_points_test = self.getAccuracy(self.Test_InputSentences,
                                                                                         self.Test_EDUBreaks,
                                                                                         self.Test_DecoderInput,
                                                                                         self.Test_RelationLabel,
                                                                                         self.Test_ParsingBreaks,
                                                                                         self.Test_GoldenMetric,
                                                                                         use_pred_segmentation=False,
                                                                                         use_org_Parseval=False)

            # Unfold numbers
            # Test
            P_span, R_span, F_span = span_points_Test
            P_relation, R_relation, F_relation = relation_points_Test
            P_nuclearity, R_nuclearity, F_nuclearity = nuclearity_points_Test
            P_segment, R_segment, F_segment = segment_points_test
            # Training (dev)
            _, _, F_span_Trdev = span_points_Trdev
            _, _, F_relation_Trdev = relation_points_Trdev
            _, _, F_nuclearity_Trdev = nuclearity_points_Trdev

            print(
                'Train:', 'F_span:', F_span_Trdev, 'F_relation:', F_relation_Trdev, 'F_nuclearity:', F_nuclearity_Trdev)
            print('Test:', 'F_span:', F_span, 'F_relation:', F_relation, 'F_nuclearity:', F_nuclearity, 'F_segment:',
                  F_segment)

            # Relation will take the priority consideration
            # if F_relation > best_F_relation:
            if F_span > best_F_span:
                best_epoch = one_epoch
                # relation
                best_F_relation = F_relation
                best_P_relation = P_relation
                best_R_relation = R_relation
                # span
                best_F_span = F_span
                best_P_span = P_span
                best_R_span = R_span
                # nuclearity
                best_F_nuclearity = F_nuclearity
                best_P_nuclearity = P_nuclearity
                best_R_nuclearity = R_nuclearity

            # Saving data
            save_data = [one_epoch, LossTree_Trdev, LossLabel_Trdev, F_span_Trdev, F_relation_Trdev, F_nuclearity_Trdev,
                         LossTree_Test, LossLabel_Test, F_span, F_relation, F_nuclearity, F_segment]

            FileName = 'span_bs_{}_es_{}_lr_{}_lrdc_{}_wd_{}.txt'.format(self.batch_size, self.eval_size, self.lr,
                                                                         self.lr_decay_epoch, self.weight_decay)

            with open(os.path.join(self.save_path, FileName), 'a+') as f:
                f.write(','.join(map(str, save_data)) + '\n')

            if config.save_model is True:
                if (one_epoch % 1 == 0 and one_epoch > -1):
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, r'Epoch_%d.torchsave' % (one_epoch)))

            # Convert back to training
            self.model.train()

        return best_epoch, best_F_relation, best_P_relation, best_R_relation, best_F_span, \
               best_P_span, best_R_span, best_F_nuclearity, best_P_nuclearity, best_R_nuclearity
