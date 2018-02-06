from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from nltk.corpus import stopwords
from nltk.tokenize import BlanklineTokenizer, TreebankWordTokenizer, LineTokenizer, word_tokenize
from gensim.models.ldamodel import LdaModel
from nltk.stem import SnowballStemmer
from six import iteritems
from wordcloud import WordCloud
from PIL import Image
import indicoio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
import gensim
import string
import io
import os
import re

import config


logger = config.config_logger(__name__, 10)
nltk.download('stopwords')
nltk.download('punkt')


class Paragraph:
    def __init__(self, name, text, bow, region, sentiment, agent):
        self.name = name
        self.text = text
        self.bow = bow
        self.region = region
        self.sentiment = sentiment
        self.agent = agent

    def __str__(self):
        if self.bow is not None:
            return '{0} - {1} - Words: {2} - {3} - {4}'.format(
                self.name, self.region, len(self.bow), self.text[:15], self.sentiment)
        else:
            return '{0} - {2} - Words: {2} - {3} - {4}'.format(
                self.name, self.region, len(self.text.split()), self.text[:15], self.sentiment)

    def clean_data(self):
        doc = self.text
        stop = set(stopwords.words('spanish'))
        exclude = set(string.punctuation)
        exclude.update(['«', '¿', '»', '¡', '…', '“', '—', '_', '‘', '–', '’'])
        stop.update(['señor', 'señora', 'ahí', 'jorge', 'castilla'])
        stemmer = SnowballStemmer('spanish')
        stop_free = ' '.join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = ' '.join(stemmer.stem(word) for word in punc_free.split())
        return Paragraph(name=self.name, text=normalized, bow=normalized.split(),
                         region=self.region, sentiment=self.sentiment, agent=self.agent)


class Testimony:
    def __init__(self, name, text, bow, region, gen_region=False):
        self.name = name
        self.text = text
        self.bow = bow
        self.region = region

        if gen_region:
            lines = LineTokenizer().tokenize(self.text)
            self.region = lines[0].split('EN ')[-1]

    def __str__(self):
        if self.bow is not None:
            return '{0} - {1} - Words: {2} - {3}'.format(self.name, self.region, len(self.bow), self.text[:15])
        else:
            return '{0} - {2} - Words: {2} - {3}'.format(self.name, self.region, len(self.text.split()), self.text[:15])

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
        return Testimony(name=self.name, text=normalized, bow=normalized.split(), region=self.region)

    def filter_protocol(self):
        my_text = self.text.replace('    ', '\n\n')
        output = []
        veto = ['AUDIENCIAS', 'tomar asiento', 'pueden empezar', 'Caso número', 'testimonio de', 'ponerse de pie',
                'brindar su testimonio', 'prometen solemnemente', 'coloquen de pie', 'promesa respectiva',
                'Testimonio de', 'gracias  por  este testimonio', 'SESIÓN', 'A.M.', 'P.M', 'ENERO', 'FEBRERO',
                'MARZO', 'ABRIL', 'MAYO', 'JUNIO', 'JULIO', 'AGOSTO', 'SEPTIEMBRE', 'OCTUBRE', 'NOVIEMBRE',
                'DICIEMBRE', 'por tu testimonio', 'gracias por su valiente testimonio',
                'miembros de la Comisión de la Verdad', 'decir solo la verdad', 'Asiento']
        veto_first = ['Señora', 'Doctor', 'Bienvenidas', 'Bienvenidos', 'Sí.', 'Sí', 'Sí,' 'Padre',
                      'Gracias,', 'Gracias.', 'Muchas gracias,' ' Muchas gracias.', 'Ingeniero',' Monseñor',
                      'Testimonio', 'Señora', 'Vamos', 'Buenos', 'Bienvenidas,', 'Bienvenidos,', 'Caso']
        #paragraphs = BlanklineTokenizer().tokenize(self.text)
        paragraphs = re.split(r'\s*\n\s*\n\s*', my_text)
        parag_filter = []
        for parag in paragraphs:
            filter_lines = []
            lines = parag.split('\n')
            for line in lines:
                words = line.split(' ')
                if any(word in words for word in veto):
                    continue
                elif words[0] in veto_first:
                    continue
                elif len(words) < 2:
                    continue
                else:
                    filter_lines.append(line)
            if len(filter_lines) != 0:
                filter_lines = ' '.join(line for line in filter_lines)
                parag_filter.append(Paragraph(
                    name=self.name,
                    text=filter_lines,
                    bow=filter_lines.split(),
                    region=self.region,
                    sentiment='none',
                    agent='none'))
        return parag_filter


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
        data = pdfparser(pdf_path + pdf_name, )
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
        all_txt.append(Testimony(name=temp_name, text=data, bow=None, gen_region=True, region='none'))
        file.close()
    return all_txt


def extract_quechua(docs):
    quechua_words = ['ñachik', 'ñachu', 'ñam', 'ñataq', 'ñawi', 'ñañ', 'ñuq', 'ñuqachik', 'ñuqahin', 'ñuqall',
                     'ñuqallay', 'ñuqallayq', 'ñuqam', 'ñuqanchik', 'ñuqap', 'ñuqapam', 'ñuqapaq', 'ñuqapaqq',
                     'ñuqaq', 'ñuqat', 'ñuqatap', 'ñuqataq', 'ñuqay', 'ñuqayku', 'ñuqaykum', 'ñuqaykun',
                     'ñuqaykup', 'ñuqaykupaq', 'ñuqaykuq', 'ñuqaykut', 'ñuqaykutam', 'ñuqaykuñataq', 'ñuqañataq']
    spanish, quechua = [], []
    for testimony in docs:
        counter = 0
        for parag in testimony:
            if any(word in quechua_words for word in parag.bow):
                counter += 1
        if counter > 0:
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
    my_stopwords.update(['Entonces', 'ahí', 'así', 'dijo', 'dije', 'después', 'señor', 'señora', 'sí', 'si', 'no'
                         'Lerner', 'Salomón', 'Doctor', 'Febres', 'Salomón', 'AUDIENCIAS', 'PÚBLICAS',
                         'CASOS', 'Comisión', 'Reconciliación', 'Verdad', 'dice', 'acá', 'sido', 'chay', 'pues'])
    my_mask = np.array(Image.open('./data/graphs/gun.jpg'))
    wordcloud = WordCloud(background_color='white',
                          stopwords=my_stopwords,
                          width=1200,
                          height=1000).generate(test_string)
    frequencies = pd.DataFrame.from_dict(wordcloud.words_, orient='index').reset_index()
    frequencies.columns = ['word', 'freq']
    frequencies = frequencies.sort_values('freq', ascending=False)
    plt.imshow(wordcloud)
    plt.axis('off')
    wordcloud.to_file(output_path + 'graphs/' + 'wordcloud.jpg')
    plt.close()
    return frequencies


def input_sentiment(doc, my_dict):
    for index, my_dict in my_dict.iterrows():
        if doc[0].name == my_dict['name']:
            #print(doc[0].name)
            #print('----------' + my_dict['start'])
            for parag in doc:
                len_key = len(my_dict['start'])
                check_text = ' '.join(parag.text.split())
                #print(check_text[:len_key] + '---' + my_dict['start'] + str(len_key))
                if check_text[:len_key] == my_dict['start']:
                    setattr(parag, 'sentiment', my_dict['sentiment'])
                    #print(parag)
    return doc


def input_agent(doc, agent_dict):
    for parag in doc:
        freqs = nltk.FreqDist(parag.bow)
        agent_dict['count'] = agent_dict.apply(lambda x: count_agent(x, freqs), axis=1)
        agents = agent_dict.groupby('esfera').sum()
        number_max = list(agents['count']).count(max(agents['count']))
        if number_max == 1:
            target = agents[agents['count'] == max(agents['count'])].index.values[0]
            setattr(parag, 'agent', str(target))
    return doc


def count_agent(row, freq):
    return freq[row['label']]


def train_sentiment(docs):
    train_set = []
    test_set = []
    for doc in docs:
        for parag in doc:
            if parag.sentiment != 'none':
                train_set.append(parag)
            else:
                test_set.append(parag)
    all_words = set(word.lower() for passage in train_set for word in word_tokenize(passage.text))
    t = [({word: (word in word_tokenize(x.text)) for word in all_words}, x.sentiment) for x in train_set]
    classifier = nltk.NaiveBayesClassifier.train(t)
    for parag in test_set:
        test_features = {word.lower(): (word in word_tokenize(parag.text.lower())) for word in all_words}
        prediction = classifier.classify(test_features)
        setattr(parag, 'sentiment', prediction)
    train_set = train_set + test_set
    output = pd.DataFrame([])
    for obs in train_set:
        d = {'name': obs.name, 'region': obs.region, 'agent': obs.agent, 'sentiment': obs.sentiment}
        output = output.append(pd.DataFrame([d], columns=d.keys()))
    output = output.reset_index(drop=True)
    return output


def input_sentiment_posneg(docs):
    indicoio.config.api_key = 'd0fcf8a01ea1ee9d1de238a8c89b8b83'
    train_set = []
    for doc in docs:
        for parag in doc:
            try:
                prediction = indicoio.sentiment(parag.text)
                setattr(parag, 'sentiment', prediction)
                train_set.append(parag)
            except:
                continue
    output = pd.DataFrame([])
    for obs in train_set:
        d = {'name': obs.name, 'region': obs.region, 'agent': obs.agent, 'sentiment': obs.sentiment}
        output = output.append(pd.DataFrame([d], columns=d.keys()))
    output = output.reset_index(drop=True)
    return output



