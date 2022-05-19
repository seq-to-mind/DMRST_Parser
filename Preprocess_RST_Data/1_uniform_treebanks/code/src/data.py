#!/usr/bin/python
# -*- coding: utf-8 -*-

#from __future__ import print_function
import os, sys, codecs
from nltk import Tree

from common import *
from utils_rs3 import *
from utils_dis_thiago import *
from relationSet import *


'''
TODO:
    - for now, read the entire corpus before writing, do both at the same time
    - still issues with the ps output (warning + not really pretty print)
'''



class Corpus:
    def __init__( self, tbpath, datatype="dis", mapping=True, draw=True ):
        self.path = tbpath
        self.datatype = datatype
        self.draw = draw #draw a ps file for each tree
        self.files = []
        self.edufiles = []
        self.documents = []
        self.validDocuments = []#document for which a valid tree has been built
        self.outputExt = ".dmrg" #Extension of the output tree files
        self.mapping=mapping
        #Keep track of the original/final relation set
        self.originLabels, self.finalLabels = set(), set()

    def read( self ):
        self.getDocuments( )
        for doc in self.documents:
            print( "Reading:", os.path.basename(doc.path), file=sys.stderr )
            doc.read()
            addLabels( doc.tree, self.originLabels )
            if self.mapping == True:
                doc.mapRelation( 'mapping' )
            addLabels( doc.tree, self.finalLabels )
        self.validDocuments = [d for d in self.documents if d.tree != None]
        #self.validDocuments = self.documents
        self.pb_files = [d.path for d in self.documents if d.tree == None]
        print( "\t#Files read:", len( self.files ),
                "#Tree built:", len( self.validDocuments ), file=sys.stderr )

    def write( self, outpath ):
        if not os.path.isdir( outpath ):
            os.mkdir( outpath )
        for doc in self.validDocuments:
            print( "Writing:", os.path.basename(doc.path), file=sys.stderr )
            doc.writeTree( outpath, self.outputExt )
            doc.writeEdu( outpath )
            if self.draw:#create a picture representing the tree
                doc.drawTree( outpath, '.'+doc.datatype, '.ps' )
        # Write the list of documents for which we couldnt build a tree
        if len( self.pb_files ) != 0:
            with open( os.path.join( outpath, "pb_files" ), 'w' ) as f:
                f.write( '\n'.join( self.pb_files ) )

    def getDocuments( self ):
        if self.datatype == "dis":
            # retrieve tree files DisDocument(
            self.files = getFiles( self.path, ".dis" )
            # retrieve edu files
            self.edufiles = getFiles( self.path, ".edus" )
            # Associate each tree with the corresponding edu file
            self.documents = associate_tree_edus( self.files, self.edufiles )
        elif self.datatype == "rs3":
            self.files = getFiles( self.path, ".rs3" )
            self.documents = [Rs3Document( f ) for f in self.files]
        elif self.datatype == "thiago":
            self.files = getFiles( self.path, ".txt.lisp.thiago" )
            self.documents = [ThiagoDocument( f ) for f in self.files]
        else:
            sys.exit( "Unknown data type "+datatype )

    def printLabels( self ):
        ''' The label sets record tuples (relation, nuclearity)  '''
        # -- Originaly
        labels = np.unique( [l for (l,n) in self.originLabels] )
        print( "\n#Original Labels:"+str(len( labels) ) )
        print( ', '.join( sorted( labels ) ) )
        # -- Finaly/mapped
        labels = np.unique( [l for (l,n) in self.finalLabels] )
        print( "\n#Final Labels:"+str(len( labels) ) )
        print( ', '.join( sorted( labels ) ) )

    def __str__( self ):
        return ' '.join( [self.path, "Type:", self.datatype] )



# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class Document:
    def __init__( self, dpath ):#? parse=parse, raw=raw
        self.path = dpath
        self.datatype = None
        self.tree = None
        self.tokendict = None # Token dict: id token in the document -> token form
        self.eduIds = []
        self.edudict = None # EDU dict: id EDU -> list of id tokens
        self.outbasename = os.path.basename( self.path ) #Name of the output file, can be modified for the RST DT
        self.statistics = {} # statistics for one document

    def read( self ):
        raise NotImplementedError

    def writeTree( self, outpath, outExt ):
        '''
        Write the bracketed tree into a file
        Remove the original extension, keep only .outExt as extension
        '''
        fileout = os.path.join( outpath,
                self.outbasename.replace( '.out', '' ).replace('.txt.lisp','').replace( '.'+self.datatype, '' ) )+outExt
        with open( fileout, 'w' ) as fout:
            fout.write( self.tree.__str__().strip() )

    def drawTree(self, outpath, ext, outExt, docno=-1 ):
        '''Draw RST tree into a file'''
        if not self.outbasename.endswith( ext ):#For modified file names
            fileout = os.path.join( outpath, self.outbasename+outExt )
        else:
            fileout = os.path.join( outpath, self.outbasename.replace( ext, outExt ) )
        cf = CanvasFrame()
        tc = TreeWidget(cf.canvas(), self.tree)
        cf.add_widget(tc,10,10) # (10,10) offsets
        cf.print_to_file( fileout )
        cf.destroy()

    def mapRelation( self, mappingRel ):
        if self.tree == None:
            return
        if os.path.isfile( mappingRel ):
            sys.exit( "Mapping RS3 from file not implemented yet" )
        else:
            if mappingRel == 'mapping':#Default general mapping
                performMapping( self.tree, mapping )
            elif mappingRel == 'basque_labels':
                performMapping( self.tree, basque_labels )
            elif mappingRel == 'brazilianCst_labels':
                performMapping( self.tree, brazilianCst_labels )
            elif mappingRel == 'brazilianSum_labels':
                performMapping( self.tree, brazilianSum_labels )
            elif mappingRel == 'germanPcc_labels':
                performMapping( self.tree, germanPcc_labels )
            elif mappingRel == 'spanish_labels':
                performMapping( self.tree, spanish_labels )
            elif mappingRel == 'rstdt_mapping18':
                performMapping( self.tree, rstdt_mapping18 )
            elif mappingRel == 'dutch_labels':
                performMapping( self.tree, dutch_labels )
            elif mappingRel == 'brazilianTCC_labels':
                performMapping( self.tree, brazilianTCC_labels )
            else:
                print( "Unknown mapping: "+mappingRel )

# ----------------------------------------------------------------------------------
class Rs3Document( Document ):
    '''
    Class for a document encoded in rs3 format.
    - XML format
    - the relation list in the header gives the nuclearity of the relations
    - EDU id are not always continuous: EDU are renamed
    - For some corpora/languages, the binarization using right branching is not enough,
    a more general strategy is used
    - An EDU file is created
    '''
    def __init__( self, dpath ):
        Document.__init__(self, dpath)
        self.datatype = "rs3"
        self.nuclearity_relations = {}

    def read( self ):
        '''
        Create a binarized (NLTK) Tree, self.tree, from the rs3 file
        Fill self.tokendict and self.edudict
        '''
        doc_root, rs3_xml_tree = parseXML( self.path )
        # Retrieve the relations in the header (used to find multinuc rel)
        self.nuclearity_relations = getRelationsType( rs3_xml_tree )
        # Get info for each node
        eduList, groupList, root = readRS3Annotation( doc_root )
        # Build nodes, rename DU, tree=SpanNode instance
        tree = buildNodes( eduList, groupList, root, self.nuclearity_relations )
        # Can t be retrieved from the tree for now, some EDU have children
        eduIds = [e["id"] for e in eduList]
        # Order span list for each node
        orderSpanList( tree, eduIds )
        # Clean the tree: deal with DU with only one child + same unit cases
        cleanTree( tree, eduIds, self.nuclearity_relations, self )
        # Retrieve info about the text of the EDUs
        self.tokendict, self.edudict = retrieveEdu( tree, eduIds )
        non_bin_tree = tree
        # Binarize the tree
        binarizeTreeGeneral(tree, self, nucRelations=self.nuclearity_relations)
        tree = backprop(tree, self) # Backprop info
        self.tree = Tree.fromstring( parse(tree) )# Build an nltk tree
        validTree = checkTree( self.tree, non_bin_tree, self )
        if not validTree:
            self.tree = None

    def writeEdu( self, outpath ):
        writeEdus( self, ".rs3", outpath ) #Write edu files similar to the ones used in the RST DT


# ----------------------------------------------------------------------------------
class DisDocument( Document ):
    def __init__( self, dpath, epath ):
        Document.__init__(self, dpath)
        self.datatype = "dis"
        self.eduPath = epath

    def read( self ):#, eduFiles
        basename = os.path.basename( self.path )
        for e in ['.out','.dis','.txt','.edus']:
            basename = basename.replace( e, '' )
        if basename in file_mapping:#Modify the name of some specific files in the RST DT
            self.outbasename = file_mapping[basename]
        tree, self.eduIds = buildTree(open( self.path ).read()) # Build RST Tree
        tree = binarizeTreeRight(tree)# Binarize it
        doc = readEduDoc( self.eduPath, self )# Retrieve info on EDUs
        tree = backprop(tree, self)
        str_tree = parse(tree)#Get nltk tree
        self.tree = Tree.fromstring( str_tree )

    def writeEdu( self, outpath, docno=-1 ):
        # copy the EDU file, possibly rename it using the file mapping
        if self.outbasename != os.path.basename( self.path.split('.')[0] ):
            shutil.copy( self.eduPath, os.path.join( outpath,
                self.outbasename.replace( '.out', '' ).replace( '.dis', '' )+'.edus' ) )
        else:
            shutil.copy( self.eduPath.replace( '.out', '' ).replace( '.dis', '' ), outpath )


# ----------------------------------------------------------------------------------
class ThiagoDocument( Document ):
    def __init__( self, dpath ):
        Document.__init__(self, dpath)
        self.datatype = "thiago"
        self.eduPath = None

    def read( self ):
        tree, self.eduIds, allnodes, self.edudict = buildTreeThiago(
                open( self.path, encoding="windows-1252" ).read() )
        tree = bTree( allnodes, self.path )
        tree = binarizeTreeRightThiago(tree)
        tree = backprop(tree, self) # Backprop info
        str_tree = parse(tree)
        self.tree = Tree.fromstring( parse(tree) )

    def writeEdu( self, outpath ):
        writeEdusFile( self, ".txt.lisp.thiago", outpath )


# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
class SpanNode:
    """
    RST tree node (from DPLP, by Yangfeng Ji)
    """
    def __init__(self, prop):
        """
        Initialization of SpanNode
        :type text: string
        :param text: text of this span
        """
        self.text, self.relation = None, None # Text of this span / Discourse relation
        self.eduspan, self.nucspan = None, None # EDU span / Nucleus span (begin, end) index id EDU
        self.nucedu = None # Nucleus single EDU (itself id for an EDU)s
        self.prop = prop # Property: Nucleus/Satellite/Roots
        self.lnode, self.rnode = None, None # Children nodes (for binary RST tree only)
        self.pnode = None # Parent node
        self.nodelist = [] # Node list (for general RST tree only)
        self.form = None # Relation form: NN, NS, SN
        self.eduCovered = [] # Id of the EDUS covered by a CDU (CHLOE Added)
        self._id = None # Id (int) of a DU, only from rs3 files (CHLOE Added)

    def __str__( self ):
        return self._info()+"\n"+"\n".join( "\t"+n._info() for n in self.nodelist )

    def _info( self ):
        return "eduspan: "+str(self.eduspan)



# ----------------------------------------------------------------------------------
def associate_tree_edus( treeFiles, eduFiles ):
    ''' Retrieve the EDU file associated to a tree for the dis format '''
    documents = []
    for treePath in treeFiles:
        basename = os.path.basename( treePath )
        for e in ['.out','.dis','.txt','.edus']:
            basename = basename.replace( e, '' )
        eduPath = findFile( eduFiles, basename )# Retrieve EDUs file
        if eduPath == None:
            sys.exit( "Edus file not found: "+basename )
        documents.append( DisDocument( treePath, eduPath ) )
    return documents

def getFiles( tbpath, ext ):
    files = []
    for p, dirs, _files in os.walk( tbpath ):
        subdir = os.path.basename(p)
        dirs[:] = [d for d in dirs if not d[0] == '.']
        for file in _files:
            if not file.startswith('.') and file.endswith(ext):
                #and not 'ANOTADOR_B' in os.path.join( p,file ) TODO
                files.append( os.path.join( p, file ) )
    return files

