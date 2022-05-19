# README #

This directory contains code to pre-process discourse corpora annotated within the RST framework. The input are *rs3*, *dis* or *lisp*/*thiago* files, the output are *dmrg* files representing the discourse trees in a bracketed format (similar to *mrg* for syntax), and *edus* files corresponding to a stand-off annotation of the text of the EDUs.

### Required packages ###
* Python3
* NLTK
* xml.etree.ElementTree 

### Data ###
This code has been used to pre-process the following corpora, see (Braud et al. EACL, 2017) for the references to all these corpora:

* RST Discourse Treebank
* CST-News
* Summ-it
* Rhetalho
* CorpusTCC 
* Spanish RST DT 
* Postdam Commentary Corpus (MAZ)
* Dutch RST DT
* Basque RST DT

Samples of the original and pre-processed data can be found in *preprocess_rst/data/*. The subdirectories names correspond to the names used for the corpora in (Braud et al. EACL, 2017), i.e. en-rst for the RST DT, eu-rst for the Basque RST DT etc.

* data/rstdt-sample/: contain samples of the original data, the format is either dis, rs3 or lisp/thiago
* data/rstdt-sample-dmrg/: contain samples of the pre-processed data, .dmrg files correspond to the trees and .edus to the text of the EDUs (the index of the ids of the EDUs in the dmrg files begin at 1)

### Pre-processing data ###

Use the following line to pre-process a corpus:

        bash path_to/preprocess_rst/code/read_corpus.sh PATH_TO_SRC_DIR PATH_TO_INPUT_DATA PATH_TO_OUTPUT_DATA

Test the code with the samples in *data/*, for example with the English RST DT:

        bash path_to/preprocess_rst/code/read_corpus.sh path_to/preprocess_rst/code/src/ path_to/preprocess_rst/data/rstdt-sample/en-rst/ path_to/preprocess_rst/data/rstdt-sample-dmrg/en-rst/ 

By default, this script does a mapping of the relation names to the 18 coarse grained classes, as defined in (Carlson et al. 2001) for the English RST DT, and as proposed in (Braud et al., 2017) for the others.
Remove the --mapping option in the bash script to keep the original labels.
You can also modify the dictionnary in *code/src/relationSet.py* to use another mapping (all the relations from all the corpora cited above are in this dictionary).

Use the --draw option to produce *ps* files representing the trees.

### Contact
Thanks the code from chloe.braud@gmail.com

