## Introduction
* One implementation of the paper "DMRST: A Joint Framework for Document-Level Multilingual RST Discourse Segmentation and Parsing". <br>
* Users can apply it to parse the input text from scratch, and get the EDU segmentations and the parsed tree structure. <br>
* The model supports both sentence-level and document-level RST discourse parsing. <br>

## Package Requirements
1. pytorch==1.7.1
2. transformers==4.8.2

## Supported Languages
We trained and evaluated the model with the multilingual collection of RST discourse treebanks, and it natively supports 6 languages: English, Portuguese, Spanish, German, Dutch, Basque. Interested users can also try other languages.

## Data Format
* [Input] `InputSentence`: The input document/sentence, and the raw text will be tokenizaed and encoded by the `xlm-roberta-base` language backbone. '|| ' denotes the EDU boundary positions. <br>
    * Although the report, || which has released || before the stock market opened, || didn't trigger the 190.58 point drop in the Dow Jones Industrial Average, || analysts said || it did play a role in the market's decline. || <br>

* [Output] `EDU_Breaks`: The indices of the EDU boundary tokens, including the last word of the sentence. <br>
    * [2, 5, 10, 22, 24, 33] <br>

* [Output] `tree_parsing_output`: The model outputs of the discourse parsing tree follow this format. <br>
   * (1:Satellite=Contrast:4,5:Nucleus=span:6) (1:Nucleus=Same-Unit:3,4:Nucleus=Same-Unite:4) (5:Satellite=Attribution:5,6:Nucleus=span:6) (1:Satellite=span:1,2:Nucleus=Elaboration:3) (2:Nucleus=span:2,3:Satellite=Temporal:3) <br>

## How to use it for parsing
* Put the text paragraph to the file `./data/text_for_inference.txt`. <br>
* Run the script `MUL_main_Infer.py` to obtain the RST parsing result. See the script for detailed model output. <br>
* We recommend users to run the parser on a GPU-equipped environment. <br>

## Citation
```
@article{liu2021dmrst,
      title={DMRST: A Joint Framework for Document-Level Multilingual RST Discourse Segmentation and Parsing}, 
      author={Zhengyuan Liu and Ke Shi and Nancy F. Chen},
      year={2021},
      eprint={2110.04518},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
```
@inproceedings{liu2020multilingual,
  title={Multilingual Neural RST Discourse Parsing},
  author={Liu, Zhengyuan and Shi, Ke and Chen, Nancy},
  booktitle={Proceedings of the 28th International Conference on Computational Linguistics},
  pages={6730--6738},
  year={2020}
}
```

