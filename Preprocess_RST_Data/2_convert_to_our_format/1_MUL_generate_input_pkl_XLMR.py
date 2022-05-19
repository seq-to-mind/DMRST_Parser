import pickle
import re
import tqdm
from glob import glob
import os, sys
from nltk.tokenize import word_tokenize
from binary_tree import BinaryTree
from binary_tree import Node
from transformers import XLMRobertaTokenizer

global_bert_tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')


RelationTable = ['Attribution_SN', 'Enablement_NS', 'Cause_SN', 'Cause_NN', 'Temporal_SN',
                 'Condition_NN', 'Cause_NS', 'Elaboration_NS', 'Background_NS',
                 'Topic-Comment_SN', 'Elaboration_SN', 'Evaluation_SN', 'Explanation_NN',
                 'TextualOrganization_NN', 'Background_SN', 'Contrast_NN', 'Evaluation_NS',
                 'Topic-Comment_NN', 'Condition_NS', 'Comparison_NS', 'Explanation_SN',
                 'Contrast_NS', 'Comparison_SN', 'Condition_SN', 'Summary_SN', 'Explanation_NS',
                 'Enablement_SN', 'Temporal_NN', 'Temporal_NS', 'Topic-Comment_NS',
                 'Manner-Means_NS', 'Same-Unit_NN', 'Summary_NS', 'Contrast_SN',
                 'Attribution_NS', 'Manner-Means_SN', 'Joint_NN', 'Comparison_NN', 'Evaluation_NN',
                 'Topic-Change_NN',  'Topic-Change_NS', 'Summary_NN']

relation_dic = {word.lower(): i for i, word in enumerate(RelationTable)}


class ParserInput:
    def __init__(self):
        self.Sentences = []
        self.EDU_Breaks = []
        self.LabelforMetric_list = []
        self.LabelforMetric = ''
        self.ParsingIndex = []
        self.Relation = []
        self.DecoderInputs = []
        self.Parents = []
        self.Siblings = []
        self.Sentence_span = []

def parse_sentence(root_node, edus_list, is_depth_manner):
    '''
    :param node: A node contains the information of a sentence.
    :return:
    '''
    root_node.parent = None
    parser_input = ParserInput()
    if is_depth_manner:
        node_list = get_depth_manner_node_list(root_node)
    else:
        node_list = get_width_manner_node_list(root_node)

    Sentences_list = []
    EDU_breaks_list = []

    edu_start = root_node.span[0]
    for node in node_list:
        if node.edu_id is not None:
            #   Sentences and EDUBreaks.
            # parser_input.Sentences += edus_list[node.edu_id - 1]
            # parser_input.EDU_Breaks.append(len(parser_input.Sentences) - 1)
            Sentences_list.append([node.edu_id, edus_list[node.edu_id - 1]])

        else:
            #   ParsingIndex:
            parser_input.ParsingIndex.append(node.left.span[1] - edu_start)

            #   DecoderInputs:
            parser_input.DecoderInputs.append(node.span[0] - edu_start)

            #   Parents:
            if node.parent is not None:
                parent_index = node.parent.span[1] - edu_start
            else:
                parent_index = 0
            parser_input.Parents.append(parent_index)

            #   Sibling:
            if node.parent is None:
                sibling_index = 99
            else:
                if node == node.parent.left:
                    sibling_index = 99
                else:
                    sibling_index = node.parent.left.span[1] - edu_start
            parser_input.Siblings.append(sibling_index)

            #   LabelforMetric:
            left_child_span = node.left.span
            right_child_span = node.right.span
            nuclearity = node.relation[:2]
            relation = node.relation[3:]
            #   Relation:
            lookup_relation = relation + '_' + nuclearity
            parser_input.Relation.append(relation_dic[lookup_relation.lower()])
            left_nuclearity = 'Nucleus' if nuclearity[0] == 'N' else 'Satellite'
            right_nuclearity = 'Nucleus' if nuclearity[1] == 'N' else 'Satellite'
            if nuclearity == 'NS' or nuclearity == 'SN':
                if nuclearity == 'NS':
                    left_relation = 'span'
                    right_relation = relation
                else:
                    left_relation = relation
                    right_relation = 'span'
            else:
                left_relation = relation
                right_relation = relation
            label_string = '(' + str(left_child_span[0]-edu_start+1) + ':' + left_nuclearity + '=' + left_relation + ':' + str(left_child_span[1]-edu_start+1) + ',' + str(right_child_span[0]-edu_start+1) + ':' + right_nuclearity + '=' + right_relation + ':' + str(right_child_span[1]-edu_start+1) + ')'
            parser_input.LabelforMetric_list.append(label_string)
    parser_input.LabelforMetric = [' '.join(parser_input.LabelforMetric_list)]
    Sentences_list = sorted(Sentences_list, key=lambda x:x[0])

    for i in range(len(Sentences_list)):
        parser_input.Sentences += Sentences_list[i][1]
        parser_input.EDU_Breaks.append(len(parser_input.Sentences) - 1)
    # print(parser_input.Sentences)
    # print(parser_input.EDU_Breaks)
    # print(parser_input.LabelforMetric)
    # print(parser_input.ParsingIndex)
    # print(parser_input.Relation)
    # print(parser_input.DecoderInputs)
    # print(parser_input.Parents)
    # print(parser_input.Siblings)
    # print('\n')
    return parser_input




def get_depth_manner_node_list(root):
    node_list = []
    stack = []
    stack.append(root)
    while len(stack) > 0:
        node = stack.pop()
        node_list.append(node)
        if node.right is not None:
            stack.append(node.right)
        if node.left is not None:
            stack.append(node.left)
    return node_list


def get_width_manner_node_list(root):
    node_list = []
    queue = []
    if root is not None:
        queue.append(root)
    while len(queue) != 0:
        node = queue.pop(0)
        node_list.append(node)
        if node.left is not None:
            queue.append(node.left)
        if node.right is not None:
            queue.append(node.right)
    return node_list

def get_sentence_span_list(sentence_span_dic):
    sentence_list = []
    for key in sentence_span_dic:
        tem_str = key.replace('[','').replace(']','')
        tokens = tem_str.split(',')
        left = int(tokens[0])
        right = int(tokens[1])
        sentence_list.append([left, right])
    return sentence_list



def find_sentence_span(node, edus_list, is_depth_manner):
    '''
    :return: Depth first to find all sentence span.
    '''
    global num_sentence_with_one_edu
    global total_sentences
    if node.is_sentence_span:
        if node.edu_id is not None:
            num_sentence_with_one_edu += 1
            # filter sentences that only have one edu.
            Sentences_list.append(edus_list[node.edu_id - 1])
            EDUBreaks_list.append([len(edus_list[node.edu_id - 1]) -1])
            LableforMetric_list.append(['NONE'])
            ParsingIndex_list.append([])
            Relation_list.append([])
            DecoderInput_list.append([])
            Parents_list.append([])
            Siblings_list.append([])
            return 0
        total_sentences += 1
        parser_input = parse_sentence(node, edus_list, is_depth_manner)
        Sentences_list.append(parser_input.Sentences)
        EDUBreaks_list.append(parser_input.EDU_Breaks)
        LableforMetric_list.append(parser_input.LabelforMetric)
        ParsingIndex_list.append(parser_input.ParsingIndex)
        Relation_list.append(parser_input.Relation)
        DecoderInput_list.append(parser_input.DecoderInputs)
        Parents_list.append(parser_input.Parents)
        Siblings_list.append(parser_input.Siblings)
        return 0
    if node.left is not None:
        find_sentence_span(node.left, edus_list, is_depth_manner)
    if node.right is not None:
        find_sentence_span(node.right, edus_list, is_depth_manner)


def find_document_span(node, edus_list, is_depth_manner, sentence_span_dic):
    '''
    :return: document level span.
    '''
    global total_sentences

    total_sentences += 1
    parser_input = parse_sentence(node, edus_list, is_depth_manner)
    Sentences_list.append(parser_input.Sentences)
    EDUBreaks_list.append(parser_input.EDU_Breaks)
    LableforMetric_list.append(parser_input.LabelforMetric)
    ParsingIndex_list.append(parser_input.ParsingIndex)
    Relation_list.append(parser_input.Relation)
    DecoderInput_list.append(parser_input.DecoderInputs)
    Parents_list.append(parser_input.Parents)
    Siblings_list.append(parser_input.Siblings)
    Sentence_Span_list.append(get_sentence_span_list(sentence_span_dic))
    return 0



def read_edus(edus_path):
    edus_list = []
    with open(edus_path, 'r') as f:
        for line in f:
            tokens = global_bert_tokenizer.tokenize(line.strip())
            # tokens = word_tokenize(line.strip())
            edus_list.append(tokens)
    return edus_list


def generate_input(dmrg_path, text_path, edus_path, is_sentence_level, is_depth_manner):
    tree = BinaryTree(dmrg_path, text_path, edus_path)
    edus_list = read_edus(edus_path)
    if is_sentence_level:
        find_sentence_span(tree.root, edus_list, is_depth_manner)
    else:
        find_document_span(tree.root, edus_list, is_depth_manner, tree.sentence_span)


def save_pickle(obj, file_path):
    file = open(file_path, 'wb')
    pickle.dump(obj, file)
    file.close()


if __name__ == '__main__':

    # build_mode = input("Please select the build mode: depth / breadth :")
    build_mode = "depth"
    if build_mode.strip() == "depth":
        depth_manner = True
    elif build_mode.strip() == "breadth":
        depth_manner = False
    else:
        print("Build mode only depth / breadth ")
        exit()

    input_base_path = "data/translated_data/to_pt/"

    output_base_path = "data/pickle_data/depth/to_pt"
    assert build_mode.strip() in output_base_path



    is_sentence_level = False
    subdirs = os.listdir(input_base_path)
    for subdir in subdirs:
        print(subdir)
        File_Name_list = []
        Sentences_list = []
        EDUBreaks_list = []
        LableforMetric_list = []
        ParsingIndex_list = []
        Relation_list = []
        DecoderInput_list = []
        Parents_list = []
        Siblings_list = []
        Sentence_Span_list = []

        total_sentences = 0
        num_sentence_with_one_edu = 0

        one_language_dmrg_path = input_base_path + subdir + "/"

        if not os.path.isdir(output_base_path + "/" + subdir + "/"):
            os.mkdir(output_base_path + "/" + subdir + "/")
        output_path = output_base_path + "/" + subdir + "/"

        dmrg_paths = sorted(glob(one_language_dmrg_path + '*.dmrg'))
        # dmrg_paths = sorted(glob(All_raw_files_path + '*.dmrg'))

        for dmrg_path in dmrg_paths:
            file_name = dmrg_path.split('/')[-1].split('.')[0]
            File_Name_list.append(file_name)
            edus_file_path = input_base_path + subdir + "/" + file_name + '.edus'
            out_file_path = input_base_path + subdir + "/" + file_name + '.edus'

            if os.path.exists(out_file_path):
                if os.path.exists(edus_file_path):
                    generate_input(dmrg_path, out_file_path, edus_file_path, is_sentence_level, is_depth_manner=depth_manner)
                else:
                    print(edus_file_path)
            else:
                print(out_file_path)

        save_pickle(File_Name_list, output_path + 'FileName.pickle')
        save_pickle(Sentences_list, output_path + 'InputSentences.pickle')
        save_pickle(EDUBreaks_list, output_path + 'EDUBreaks.pickle')
        save_pickle(LableforMetric_list, output_path + 'GoldenLabelforMetric.pickle')
        save_pickle(ParsingIndex_list, output_path + 'ParsingIndex.pickle')
        save_pickle(Relation_list, output_path + 'RelationLabel.pickle')
        save_pickle(DecoderInput_list, output_path + 'DecoderInputs.pickle')
        save_pickle(Parents_list, output_path + 'ParentsIndex.pickle')
        save_pickle(Siblings_list, output_path + 'Sibling.pickle')
        save_pickle(Sentence_Span_list, output_path + 'SentenceSpan.pickle')

        print('Total Number of Sentences:', total_sentences)
        print('Number of only one EDU Sentences', num_sentence_with_one_edu)

