from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.ldamodel import LdaModel
from nltk.stem import SnowballStemmer
from six import iteritems
#from pattern.es import parsetree
import gensim
import string
import io
import os
import pprint

import config

#import nltk
#nltk.download('wordnet')

logger = config.config_logger(__name__, 10)


def pdfparser(data):

    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data = retstr.getvalue()

    return data


def convert_pdf_to_txt(pdf_path, txt_path):
    logger.info('Stage 0: .pdf to .txt')
    logger.info('Pdf path: {0}'.format(pdf_path))
    pdf_list = os.listdir(pdf_path)
    all_txt = []

    for index, pdf_name in enumerate(pdf_list):
        data = pdfparser(pdf_path + pdf_name)
        all_txt.append(data)
        temp_name = os.path.splitext(pdf_name)[0]
        temp = open(txt_path + '{0}.txt'.format(temp_name), 'w')
        temp.write(data)
        temp.close()
        print('{0}. {1} - len: {2}\n{3}'.format(index, temp_name, len(data), data[:100]))
    return all_txt


def clean_data(doc):
    stop = set(stopwords.words('spanish'))
    exclude = set(string.punctuation)
    exclude.update(['«', '¿', '»', '¡', '…', '“', '—', '_', '‘', '–', '’'])
    #lemma = WordNetLemmatizer()
    stemmer = SnowballStemmer('spanish')
    #parsetree("buena", lemmata=True)
    stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = ' '.join(stemmer.stem(word) for word in punc_free.split())
    return normalized


def list_to_matrix(doc):
    dictionary = gensim.corpora.Dictionary(doc)
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(once_ids)
    dictionary.compactify()

    logger.info('Unique words: {0}'.format(len(dictionary.keys())))
    doc_term_matrix = [dictionary.doc2bow(my_doc) for my_doc in doc]
    return doc_term_matrix, dictionary


def lda_model(dictionary, matrix, n_topics):
    ldamodel = LdaModel(matrix, num_topics=n_topics, id2word=dictionary, passes=50)
    return ldamodel

