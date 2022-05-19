import re


def getEvalData(sen, edus):
    b = re.findall(r'\d+', sen)
    b = [str(edus[int(i) - 1]) for i in b]
    cur_new = []
    x = 0
    while x < len(b):
        cur_new.append(b[x] + '-' + b[x + 1])
        x = x + 2
    span = re.split(r' ', sen)
    # print(span)
    dic = {}
    for i in range(len(span)):
        temp = span[i]
        IDK = re.split(r'[:,=]', temp)
        Nuclearity1 = IDK[1]
        relation1 = IDK[2]
        Nuclearity2 = IDK[5]
        relation2 = IDK[6]
        dic[cur_new[2 * i]] = [relation1, Nuclearity1]
        dic[cur_new[2 * i + 1]] = [relation2, Nuclearity2]
    return dic


def getEvalData_parseval(sen, edus):
    span_list = re.split(r' ', sen)
    # print(span)
    dic = {}
    for i in range(len(span_list)):
        temp = span_list[i]
        IDK = re.split(r'[:,=]', temp)
        nuclearity = IDK[1][0] + IDK[5][0]
        relation1 = IDK[2]
        relation2 = IDK[6]
        relation = relation1 if relation1 != 'span' else relation2
        start = str(edus[int(IDK[0].strip('(')) - 1])
        end = str(edus[int(IDK[-1].strip(')')) - 1])
        span = start + '-' + end
        dic[span] = [relation, nuclearity]
    return dic


def getMeasurement(sen1, sen2, sent1_edus, sent2_edus, use_org_Parseval):
    if use_org_Parseval:
        dic1 = getEvalData_parseval(sen1, sent1_edus)
        dic2 = getEvalData_parseval(sen2, sent2_edus)
    else:
        dic1 = getEvalData(sen1, sent1_edus)
        dic2 = getEvalData(sen2, sent2_edus)
    NoNS = 0
    NoRelation = 0
    NoFull = 0

    # no of right spans
    RightSpan = list(set(dic1.keys()).intersection(set(dic2.keys())))
    NoSpans = len(RightSpan)

    # Right Number of relations and nuclearity
    for span in RightSpan:
        if dic1[span][0] == dic2[span][0]:
            NoRelation = NoRelation + 1
        if dic1[span][1] == dic2[span][1]:
            NoNS = NoNS + 1
        if dic1[span][0] == dic2[span][0] and dic1[span][1] == dic2[span][1]:
            NoFull += 1

    # Measurement
    correct_span = NoSpans
    correct_relation = NoRelation
    correct_nuclearity = NoNS
    correct_full = NoFull
    no_system = len(dic1.keys())
    no_golden = len(dic2.keys())

    # return numbers
    return correct_span, correct_relation, correct_nuclearity, correct_full, no_system, no_golden


def getSegMeasure(pred_seg, gold_seg):
    num_gold = len(gold_seg)
    num_pred = len(pred_seg)
    correct = len(set(pred_seg) & set(gold_seg))

    return num_gold, num_pred, correct


def getBatchMeasure(Spans_batch, GoldenMetric_batch, predecit_EDU_breaks, EDUBreaks_batch, use_org_Parseval):
    correct_span = 0
    correct_relation = 0
    correct_nuclearity = 0
    correct_full = 0
    no_system = 0
    no_golden = 0
    no_gold_seg = 0
    no_pred_seg = 0
    no_correct_seg = 0

    correct_span_batch_list = []
    correct_relation_batch_list = []
    correct_nuclearity_batch_list = []
    no_system_batch_list = []
    no_golden_batch_list = []

    for i in range(len(Spans_batch)):

        cur_sent = Spans_batch[i][0]
        cur_golden = GoldenMetric_batch[i][0]
        cur_pred_edus = predecit_EDU_breaks[i]
        cur_gold_edus = EDUBreaks_batch[i]

        cur_spanno = 0
        cur_relationno = 0
        cur_NSno = 0
        cur_sysno = 0
        cur_goldenno = 0

        num_gold_seg, num_pred_seg, num_correct_seg = getSegMeasure(cur_pred_edus, cur_gold_edus)
        no_gold_seg += num_gold_seg
        no_pred_seg += num_pred_seg
        no_correct_seg += num_correct_seg

        if cur_sent != 'NONE' and cur_golden != 'NONE':

            cur_spanno, cur_relationno, cur_NSno, cur_full, cur_sysno, cur_goldenno = getMeasurement(cur_sent, cur_golden, cur_pred_edus, cur_gold_edus, use_org_Parseval)

            correct_span = correct_span + cur_spanno
            correct_relation = correct_relation + cur_relationno
            correct_nuclearity = correct_nuclearity + cur_NSno
            correct_full += cur_full
            no_system = no_system + cur_sysno
            no_golden = no_golden + cur_goldenno

        elif cur_sent != 'NONE' and cur_golden == 'NONE':
            _, _, _, _, cur_sysno, _ = getMeasurement(cur_sent, cur_sent, cur_pred_edus, cur_pred_edus, use_org_Parseval)
            no_system = no_system + cur_sysno

        elif cur_sent == 'NONE' and cur_golden != 'NONE':
            _, _, _, _, _, cur_goldenno = getMeasurement(cur_golden, cur_golden, cur_gold_edus, cur_gold_edus, use_org_Parseval)
            no_golden = no_golden + cur_goldenno

        correct_span_batch_list.append(cur_spanno)
        correct_relation_batch_list.append(cur_relationno)
        correct_nuclearity_batch_list.append(cur_NSno)
        no_system_batch_list.append(cur_sysno)
        no_golden_batch_list.append(cur_goldenno)

    return correct_span, correct_relation, correct_nuclearity, correct_full, no_system, no_golden, \
           correct_span_batch_list, correct_relation_batch_list, correct_nuclearity_batch_list, \
           no_system_batch_list, no_golden_batch_list, (no_gold_seg, no_pred_seg, no_correct_seg)


def getMicroMeasure(correct_span, correct_relation, correct_nuclearity, correct_full, no_system, no_golden, no_gold_seg, no_pred_seg, no_correct_seg):
    if no_system == 0:
        no_system = 1
    # Computer Micro-average measure
    # segmentation
    Precision_seg = no_correct_seg / no_pred_seg
    Recall_seg = no_correct_seg / no_gold_seg
    F1_seg = (2 * no_correct_seg) / (no_gold_seg + no_pred_seg)

    # Span
    Precision_span = correct_span / no_system
    Recall_span = correct_span / no_golden
    F1_span = (2 * correct_span) / (no_golden + no_system)

    # Relation
    Precision_relation = correct_relation / no_system
    Recall_relation = correct_relation / no_golden
    F1_relation = (2 * correct_relation) / (no_golden + no_system)

    # Nuclearity
    Precision_nuclearity = correct_nuclearity / no_system
    Recall_nuclearity = correct_nuclearity / no_golden
    F1_nuclearity = (2 * correct_nuclearity) / (no_golden + no_system)

    # Full
    F1_Full = (2 * correct_full) / (no_golden + no_system)

    return (Precision_span, Recall_span, F1_span), (Precision_relation, Recall_relation, F1_relation), \
           (Precision_nuclearity, Recall_nuclearity, F1_nuclearity), F1_Full, (Precision_seg, Recall_seg, F1_seg)


def getMacroMeasure(correct_span_list, correct_relation_list, correct_nuclearity_list, no_system_list, no_golden_list):
    # Computer Macro-average measure

    F1_span_list = []
    F1_relation_list = []
    F1_nuclearity_list = []

    Precision_span_list = []
    Precision_relation_list = []
    Precision_nuclearity_list = []

    Recall_span_list = []
    Recall_relation_list = []
    Recall_nuclearity_list = []

    for i in range(len(correct_span_list)):
        correct_span = correct_span_list[i]
        correct_relation = correct_relation_list[i]
        correct_nuclearity = correct_nuclearity_list[i]
        no_system = no_system_list[i]
        no_golden = no_golden_list[i]

        # span
        Precision_span = correct_span / no_system
        Recall_span = correct_span / no_golden
        F1_span = (2 * correct_span) / (no_golden + no_system)

        Precision_span_list.append(Precision_span)
        Recall_span_list.append(Recall_span)
        F1_span_list.append(F1_span)

        # Relation
        Precision_relation = correct_relation / no_system
        Recall_relation = correct_relation / no_golden
        F1_relation = (2 * correct_relation) / (no_golden + no_system)

        Precision_relation_list.append(Precision_relation)
        Recall_relation_list.append(Recall_relation)
        F1_relation_list.append(F1_relation)

        # Nuclearity
        Precision_nuclearity = correct_nuclearity / no_system
        Recall_nuclearity = correct_nuclearity / no_golden
        F1_nuclearity = (2 * correct_nuclearity) / (no_golden + no_system)
        Precision_nuclearity_list.append(Precision_nuclearity)
        Recall_nuclearity_list.append(Recall_nuclearity)
        F1_nuclearity_list.append(F1_nuclearity)

    F1_span_avg = sum(F1_span_list) / len(F1_span_list)
    Precision_span_avg = sum(Precision_span_list) / len(Precision_span_list)
    Recall_span_avg = sum(Recall_span_list) / len(Recall_span_list)

    F1_relation_avg = sum(F1_relation_list) / len(F1_relation_list)
    Precision_relation_avg = sum(Precision_relation_list) / len(Precision_relation_list)
    Recall_relation_avg = sum(Recall_relation_list) / len(Recall_relation_list)

    F1_nuclearity_avg = sum(F1_nuclearity_list) / len(F1_nuclearity_list)
    Precision_nuclearity_avg = sum(Precision_nuclearity_list) / len(Precision_nuclearity_list)
    Recall_nuclearity_avg = sum(Recall_nuclearity_list) / len(Recall_nuclearity_list)

    return (Precision_span_avg, Recall_span_avg, F1_span_avg), (Precision_relation_avg, Recall_relation_avg, F1_relation_avg), \
           (Precision_nuclearity_avg, Recall_nuclearity_avg, F1_nuclearity_avg)
