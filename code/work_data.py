from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from nltk.corpus import stopwords
from nltk.tokenize import BlanklineTokenizer, TreebankWordTokenizer, LineTokenizer
from gensim.models.ldamodel import LdaModel
from nltk.stem import SnowballStemmer
from six import iteritems
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import gensim
import string
import io
import os

import config


logger = config.config_logger(__name__, 10)


class Testimony:
    def __init__(self, name, text, bow):
        self.name = name
        self.text = text
        self.bow = bow

    def __str__(self):
        if self.bow is not None:
            return '{0} - Words: {0}'.format(self.name, len(self.bow))
        else:
            return '{0} - Words: {0}'.format(self.name, len(self.text.split()))

    def clean_data(self):
        doc = self.text
        stop = set(stopwords.words('spanish'))
        exclude = set(string.punctuation)
        exclude.update(['«', '¿', '»', '¡', '…', '“', '—', '_', '‘', '–', '’'])
        stop.update(['señor', 'señora'])
        stemmer = SnowballStemmer('spanish')
        stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = ' '.join(stemmer.stem(word) for word in punc_free.split())
        return Testimony(name=self.name, text=normalized, bow=normalized.split())

    def filter_protocol(self):
        token_parag = BlanklineTokenizer()
        paragraphs = token_parag.tokenize(self.text)
        output = []
        veto = ['AUDIENCIAS', 'tomar asiento', 'pueden empezar', 'Caso número', 'testimonio de', 'ponerse de pie',
                'brindar su testimonio', 'prometen solemnemente', 'coloquen de pie', 'promesa respectiva',
                'Testimonio de', 'gracias  por  este testimonio', 'SESIÓN', 'A.M.', 'P.M', 'ENERO', 'FEBRERO',
                'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE',
                'DICIEMBRE', 'por tu testimonio', 'gracias por su valiente testimonio',
                'miembros de la Comisión de la Verdad', 'decir solo la verdad', 'Asiento']
        veto_first = ['Señora', 'Doctor', 'Bienvenidas', 'Bienvenidos', 'Sí.', 'Sí', 'Sí,' 'Padre',
                      'Gracias,', 'Gracias.', 'Muchas gracias,' ' Muchas gracias.', 'Ingeniero',' Monseñor']
        for parag in paragraphs:
            token_word = TreebankWordTokenizer()
            words = token_word.tokenize(parag)
            if any(word in parag for word in veto):
                #print(parag)
                #print([word in parag for word in veto])
                continue
            elif words[0] in veto_first:
                #print([words[0] in veto_first])
                #print(words)
                continue
            else:
                output.append(parag)
        output = '\n'.join(parag for parag in output)
        return output

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
        all_txt.append(Testimony(name=pdf_name, text=data, bow=None))
        temp_name = os.path.splitext(pdf_name)[0]
        temp = open(txt_path + '{0}.txt'.format(temp_name), 'w')
        temp.write(data)
        temp.close()
        print('{0}. {1} - len: {2}\n{3}'.format(index, temp_name, len(data), data[:20]))
    return all_txt


def open_testimonies(txt_path):
    txt_list = os.listdir(txt_path)
    all_txt = []
    for txt_file in txt_list:
        file = open(txt_path + txt_file, 'r')
        data = file.read()
        temp_name = os.path.splitext(txt_file)[0]
        all_txt.append(Testimony(name=temp_name, text=data, bow=None))
        file.close()
    return all_txt


def extract_quechua(docs):
    quechua_words = ['ñachik', 'ñachu', 'ñam', 'ñataq', 'ñawi', 'ñañ', 'ñuq', 'ñuqachik', 'ñuqahin', 'ñuqall',
                     'ñuqallay', 'ñuqallayq', 'ñuqam', 'ñuqanchik', 'ñuqap', 'ñuqapam', 'ñuqapaq', 'ñuqapaqq',
                     'ñuqaq', 'ñuqat', 'ñuqatap', 'ñuqataq', 'ñuqay', 'ñuqayku', 'ñuqaykum', 'ñuqaykun',
                     'ñuqaykup', 'ñuqaykupaq', 'ñuqaykuq', 'ñuqaykut', 'ñuqaykutam', 'ñuqaykuñataq', 'ñuqañataq']
    spanish, quechua = [], []
    for testimony in docs:
        if any(word in quechua_words for word in testimony.bow):
            quechua.append(testimony)
        else:
            spanish.append(testimony)
    return spanish, quechua


def list_to_matrix(testimonials):
    docs = [x.bow for x in testimonials]
    dictionary = gensim.corpora.Dictionary(docs)
    once_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq == 1]
    dictionary.filter_tokens(once_ids)
    dictionary.compactify()

    logger.info('Unique words: {0}'.format(len(dictionary.keys())))
    doc_term_matrix = [dictionary.doc2bow(my_doc) for my_doc in docs]
    return doc_term_matrix, dictionary


def lda_model(dictionary, matrix, n_topics):
    ldamodel = LdaModel(matrix, num_topics=n_topics, id2word=dictionary, passes=50)
    return ldamodel


def generate_wordcloud(testimonials, output_path):
    test_string = ' '.join(x.text for x in testimonials)
    my_stopwords = set(stopwords.words('spanish'))
    my_stopwords.update(['Entonces', 'ahí', 'así', 'dijo', 'dije', 'después', 'señor', 'señora', 'sí', 'si', 'no'])
    my_mask = np.array(Image.open('./data/graphs/gun.jpg'))
    wordcloud = WordCloud(background_color='white',
                          stopwords=my_stopwords,
                          width=1200,
                          height=1000).generate(test_string)
    plt.imshow(wordcloud)
    plt.axis('off')
    wordcloud.to_file(output_path + 'graphs/' + 'wordcloud.jpg')
    plt.close()

