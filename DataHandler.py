import numpy as np
import re


def getLabelOrdered(Original_Order):
    '''
    Get the right order of lable for stacks manner.
    E.g. 
    [8,3,9,2,6,10,1,5,7,11,4] to [8,3,2,1,6,5,4,7,9,10,11]
    '''
    Original_Order = np.array(Original_Order)
    target = []
    stacks = ['root', Original_Order]
    while stacks[-1] != 'root':
        head = stacks[-1]
        if len(head) < 3:
            target.extend(head.tolist())
            del stacks[-1]
        else:
            target.append(head[0])
            temp = np.arange(len(head))
            top = head[temp[head < head[0]]]
            down = head[temp[head > head[0]]]
            del stacks[-1]
            if down.size > 0:
                stacks.append(down)
            if top.size > 0:
                stacks.append(top)

    return [x for x in target]


def get_RelationAndNucleus(label_index):
    RelationTable = ['Attribution_SN', 'Enablement_NS', 'Cause_SN', 'Cause_NN', 'Temporal_SN',
                     'Condition_NN', 'Cause_NS', 'Elaboration_NS', 'Background_NS',
                     'Topic-Comment_SN', 'Elaboration_SN', 'Evaluation_SN', 'Explanation_NN',
                     'TextualOrganization_NN', 'Background_SN', 'Contrast_NN', 'Evaluation_NS',
                     'Topic-Comment_NN', 'Condition_NS', 'Comparison_NS', 'Explanation_SN',
                     'Contrast_NS', 'Comparison_SN', 'Condition_SN', 'Summary_SN', 'Explanation_NS',
                     'Enablement_SN', 'Temporal_NN', 'Temporal_NS', 'Topic-Comment_NS',
                     'Manner-Means_NS', 'Same-Unit_NN', 'Summary_NS', 'Contrast_SN',
                     'Attribution_NS', 'Manner-Means_SN', 'Joint_NN', 'Comparison_NN', 'Evaluation_NN',
                     'Topic-Change_NN', 'Topic-Change_NS', 'Summary_NN', ]

    relation = RelationTable[label_index]
    temp = re.split(r'_', relation)
    sub1 = temp[0]
    sub2 = temp[1]

    if sub2 == 'NN':
        Nuclearity_left = 'Nucleus'
        Nuclearity_right = 'Nucleus'
        Relation_left = sub1
        Relation_right = sub1

    elif sub2 == 'NS':
        Nuclearity_left = 'Nucleus'
        Nuclearity_right = 'Satellite'
        Relation_left = 'span'
        Relation_right = sub1

    elif sub2 == 'SN':
        Nuclearity_left = 'Satellite'
        Nuclearity_right = 'Nucleus'
        Relation_left = sub1
        Relation_right = 'span'

    return Nuclearity_left, Nuclearity_right, Relation_left, Relation_right
