#!/bin/bash

# Read a corpus annotated within the RST framework
# and output bracketed trees (.dmrg) + standoff annotation of the EDUs (.edus)
# all the relations are mapped to the same relation set (Carlson et al., 2001)

# Exit immediately if a command exits with a non-zero status.
set -e


SRC=$1 #path_to_src_dir/ containing dt_reader.py
CORPUS=$2 #Input path containing the RST corpus, 
#  Engl RST DT: rst_discourse_treebank/data/
OUTPATH=$3 #Output directory
FORMAT=$4 #Type of discourse files (Default=dis)
#   Engl RST DT: dis, Rhetalho: thiago, Basque: rs3 ...


# -- English (RST DT)

# Mapping to the 18 coarse grained classes defined in (Carlson et al. 2001)
python ${SRC}/dt_reader.py --treebank ${CORPUS} --format ${FORMAT} --outpath ${OUTPATH} --mapping

# # If you want to keep the original labels of the relations
# python ${SRC}/dt_reader.py --treebank ${CORPUS} --format ${FORMAT} --outpath ${OUTPATH}

# # If you want to draw the trees in ps files
# python ${SRC}/dt_reader.py --treebank ${CORPUS} --format ${FORMAT} --outpath ${OUTPATH} --mapping --draw



# If the data are organized as in data/rstdt-sample/, pre-process all the corpora
# by uncommenting the following lines:
# # -- DIS
# for DSET in en-rst
# do
#     IN=${CORPUS}${DSET}
#     FORMAT=dis
#     OUT=${OUTPATH}${DSET}
# 
#     python ${SRC}/dt_reader.py --treebank ${IN} --format ${FORMAT} --outpath ${OUT} --mapping
# done
# # -- RS3
# for DSET in eu-rst de-rst du-rst es-rst
# do
#     IN=${CORPUS}${DSET}
#     FORMAT=rs3
#     OUT=${OUTPATH}${DSET}
# 
#     python ${SRC}/dt_reader.py --treebank ${IN} --format ${FORMAT} --outpath ${OUT} --mapping
# done 
# #-- For Brazilian Portuguese, merge all the corpora
# CORPUS=${CORPUS}pt-rst/
# for DSET in cst summit
# do
#     IN=${CORPUS}${DSET}
#     echo ${IN}
#     FORMAT=rs3
#     OUT=${OUTPATH}pt-rst
# 
#     python ${SRC}/dt_reader.py --treebank ${IN} --format ${FORMAT} --outpath ${OUT} --mapping
# done
# for DSET in rhetalho tcc
# do
#     IN=${CORPUS}${DSET}
#     FORMAT=thiago
#     OUT=${OUTPATH}pt-rst
# 
#     python ${SRC}/dt_reader.py --treebank ${IN} --format ${FORMAT} --outpath ${OUT} --mapping
# done


