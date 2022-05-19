from glob import glob
import pickle
import numpy as np
import os

import random
random.seed(666)

English_test_samples_list = pickle.load(open('data/pickle_data/English_test_sample_list.pickle', 'rb'))


def save_pickle(obj, file_path):
    file = open(file_path, 'wb')
    pickle.dump(obj, file)
    file.close()


def find_english_test(FileNames):
    indexs = []
    for i, name in enumerate(FileNames):
        if name in English_test_samples_list:
            indexs.append(i)
    assert len(indexs) == 38

    return indexs

def random_split(folder_path, language):
    FileNames = pickle.load(open(os.path.join(folder_path, "FileName.pickle"), "rb"))
    InputSentences = pickle.load(open(os.path.join(folder_path, "InputSentences.pickle"), "rb"))
    EDUBreaks = pickle.load(open(os.path.join(folder_path, "EDUBreaks.pickle"), "rb"))
    DecoderInput = pickle.load(open(os.path.join(folder_path, "DecoderInputs.pickle"), "rb"))
    RelationLabel = pickle.load(open(os.path.join(folder_path, "RelationLabel.pickle"), "rb"))
    ParsingBreaks = pickle.load(open(os.path.join(folder_path, "ParsingIndex.pickle"), "rb"))
    ParentsIndex = pickle.load(open(os.path.join(folder_path, "ParentsIndex.pickle"), "rb"))
    Sibling = pickle.load(open(os.path.join(folder_path, "Sibling.pickle"), "rb"))
    GoldenMetric = pickle.load(open(os.path.join(folder_path, "GoldenLabelforMetric.pickle"), "rb"))

    sample_number = len(FileNames)
    if language == 'en-dt':
        test_indexs = sorted(find_english_test(FileNames))
    else:

        sample_index = [i for i in range(sample_number)]
        random.seed(666)
        test_indexs = sorted(random.sample(sample_index, 38))
    print(language, test_indexs)

    train_indexs = [i for i in range(sample_number) if i not in test_indexs]


    Train_FileNames = [item for i, item in enumerate(FileNames) if i in train_indexs]
    Train_InputSentences = [item for i, item in enumerate(InputSentences) if i in train_indexs]
    Train_EDUBreaks = [item for i, item in enumerate(EDUBreaks) if i in train_indexs]
    Train_DecoderInput = [item for i, item in enumerate(DecoderInput) if i in train_indexs]
    Train_RelationLabel = [item for i, item in enumerate(RelationLabel) if i in train_indexs]
    Train_ParsingBreaks = [item for i, item in enumerate(ParsingBreaks) if i in train_indexs]
    Train_ParentsIndex = [item for i, item in enumerate(ParentsIndex) if i in train_indexs]
    Train_Sibling = [item for i, item in enumerate(Sibling) if i in train_indexs]
    Train_GoldenMetric = [item for i, item in enumerate(GoldenMetric) if i in train_indexs]

    Test_FileNames = [item for i, item in enumerate(FileNames) if i in test_indexs]
    Test_InputSentences = [item for i, item in enumerate(InputSentences) if i in test_indexs]
    Test_EDUBreaks = [item for i, item in enumerate(EDUBreaks) if i in test_indexs]
    Test_DecoderInput = [item for i, item in enumerate(DecoderInput) if i in test_indexs]
    Test_RelationLabel = [item for i, item in enumerate(RelationLabel) if i in test_indexs]
    Test_ParsingBreaks = [item for i, item in enumerate(ParsingBreaks) if i in test_indexs]
    Test_ParentsIndex = [item for i, item in enumerate(ParentsIndex) if i in test_indexs]
    Test_Sibling = [item for i, item in enumerate(Sibling) if i in test_indexs]
    Test_GoldenMetric = [item for i, item in enumerate(GoldenMetric) if i in test_indexs]


    save_pickle(Train_FileNames, os.path.join(folder_path, "Training_FileNames.pickle"))
    save_pickle(Train_InputSentences, os.path.join(folder_path, "Training_InputSentences.pickle"))
    save_pickle(Train_EDUBreaks, os.path.join(folder_path, "Training_EDUBreaks.pickle"))
    save_pickle(Train_DecoderInput, os.path.join(folder_path, "Training_DecoderInputs.pickle"))
    save_pickle(Train_RelationLabel, os.path.join(folder_path, "Training_RelationLabel.pickle"))
    save_pickle(Train_ParsingBreaks, os.path.join(folder_path, "Training_ParsingIndex.pickle"))
    save_pickle(Train_ParentsIndex, os.path.join(folder_path, "Training_ParentsIndex.pickle"))
    save_pickle(Train_Sibling, os.path.join(folder_path, "Training_Sibling.pickle"))
    save_pickle(Train_GoldenMetric, os.path.join(folder_path, "Training_GoldenLabelforMetric.pickle"))

    save_pickle(Test_FileNames, os.path.join(folder_path, "Testing_FileNames.pickle"))
    save_pickle(Test_InputSentences, os.path.join(folder_path, "Testing_InputSentences.pickle"))
    save_pickle(Test_EDUBreaks, os.path.join(folder_path, "Testing_EDUBreaks.pickle"))
    save_pickle(Test_DecoderInput, os.path.join(folder_path, "Testing_DecoderInputs.pickle"))
    save_pickle(Test_RelationLabel, os.path.join(folder_path, "Testing_RelationLabel.pickle"))
    save_pickle(Test_ParsingBreaks, os.path.join(folder_path, "Testing_ParsingIndex.pickle"))
    save_pickle(Test_ParentsIndex, os.path.join(folder_path, "Testing_ParentsIndex.pickle"))
    save_pickle(Test_Sibling, os.path.join(folder_path, "Testing_Sibling.pickle"))
    save_pickle(Test_GoldenMetric, os.path.join(folder_path, "Testing_GoldenLabelforMetric.pickle"))



def split_train_test(base_path):
    translated_folders = sorted(glob(base_path + 'to*'))
    for translated_folder in translated_folders:
        language_folders = sorted(glob(translated_folder+'/*'))
        for language_folder in language_folders:
            language = language_folder.split('/')[-1]
            random_split(language_folder, language)


if __name__ == "__main__":
    base_path = 'data/pickle_data/depth/'
    split_train_test(base_path)







