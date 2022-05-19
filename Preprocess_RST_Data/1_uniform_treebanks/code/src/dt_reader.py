#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
Reader for the corpora annotated within the RST framework
:author: Chloe Braud
'''

import argparse, os, sys, shutil

import data

def main( ):
    parser = argparse.ArgumentParser(
            description='Read discourse corpora (.dis, .rs3, .lisp(thiago)) and \
                    output desired files (discourse mrg files and edu files).')
    parser.add_argument('--treebank',
            default='./DataSets/RST/RST_multilingual/gum/rst/rstweb',
            dest='treebank',
            action='store',
            help='Input directory to read (RST files)')
    parser.add_argument('--outpath',
            default='./DataSets/RST/RST_multilingual/gum_transferred',
            dest='outpath',
            action='store',
            help='Output directory')
    parser.add_argument('--format',
            dest='format',
            action='store',
            choices=["rs3", "dis", "thiago"],
            default="rs3",
            help='Format (Default=dis)')
    parser.add_argument('--mapping',
            default=True,
            dest='mapping',
            action='store_true',
            help='If True, map the relations using the mapping defined in \
                    relationSet.py, i.e. the 18 coarse grained classes as\
                    proposed in Carlson et al. 2001, taking into account the\
                    modifications proposed for the other corpora defined in\
                    EACL17 paper. (Default=True)')
    parser.add_argument('--draw',
            dest='draw',
            action='store_true',
            help='Draw a ps file for each tree (Default=True)')
    args = parser.parse_args()

    if not os.path.isdir( args.outpath ):
        os.mkdir( args.outpath )

    read( args.treebank, args.outpath, mapping=args.mapping, draw=args.draw,
            format=args.format )



def read( tbpath, outpath, mapping=True, draw=True, format="dis" ):
    # Keep the training/test split if existing (i.e. for the engl RST DT)
    if os.path.isdir( os.path.join( tbpath, 'TRAINING' ) ):
        for dataset in ['TRAINING', 'TEST']:
            corpus = data.Corpus( os.path.join( tbpath, dataset ), mapping=mapping,
                    datatype=format, draw=draw )
            print( "\nCorpus:", corpus.__str__(), file=sys.stderr )
            corpus.read( )
            outpath_ = os.path.join( outpath, dataset )
            if not os.path.isdir( outpath_ ):
                os.mkdir( outpath_ )
            corpus.write( outpath_ )
    else:
        corpus = data.Corpus( tbpath, mapping=mapping, datatype=format, draw=draw )
        print( "\nCorpus:", corpus.__str__(), file=sys.stderr )
        corpus.read( )
        outpath_ = outpath
        if not os.path.isdir( outpath_ ):
            os.mkdir( outpath_ )
        corpus.write( outpath_ )
    corpus.printLabels()



if __name__ == '__main__':
    main()

