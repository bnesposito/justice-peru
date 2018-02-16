import numpy as np
import pandas as pd
import time
import logging
import pprint

import work_data
import config


def main():
    logger = config.config_logger(__name__, 10)
    t0 = time.time()

    pdf_path = './data/pdf/'
    txt_path = './data/txt/'
    dict_path = './data/dict/'
    output_path = './output/'

    convert_files = False

    logger.info('Begin execution')
    if convert_files:
        logger.info('Coonvert files: {0}'.format(convert_files))
        logging.getLogger().setLevel(30)
        all_docs = work_data.convert_pdf_to_txt(pdf_path, txt_path)
        logging.getLogger().setLevel(10)
    else:
        logger.info('Import testimonials')
        all_docs = work_data.open_testimonies(txt_path)

    logger.info('Create wordcloud')
    wordcloud_words = work_data.generate_wordcloud(all_docs, output_path)
    wordcloud_words.to_csv('./output/word_count.csv')

    logger.info('Remove protocol paragraphs')
    filter_docs = [doc.filter_protocol() for doc in all_docs]
    filter_docs = [doc for doc in filter_docs if len(doc) > 5]

    logger.info('Load dictionaries')
    dict1 = pd.read_csv(dict_path + 'dict_ale.csv', index_col=0, header=0)
    dict2 = pd.read_csv(dict_path + 'dict_erika.csv', index_col=0, header=0)
    dict3 = pd.read_csv(dict_path + 'dict_macla.csv', index_col=0, header=0)
    dict_agents = pd.read_csv(dict_path + 'dict_agentes.csv', index_col=0, header=0)

    filter_docs = [work_data.input_sentiment(doc, dict1) for doc in filter_docs]
    filter_docs = [work_data.input_sentiment(doc, dict2) for doc in filter_docs]
    filter_docs = [work_data.input_sentiment(doc, dict3) for doc in filter_docs]
    #clean_docs = [work_data.input_agent(doc, dict_agents) for doc in clean_docs]

    tagged = not_tagged = 0
    for i in filter_docs:
        for par in i:
            if par.sentiment != 'none':
                tagged += 1
            else:
                not_tagged += 1
    logger.info('Paragraphs tagged {0} - not tagged {1}'.format(tagged, not_tagged))

    logger.info('Number of testimonials: {0}'.format(len(all_docs)))
    logger.info('Clean testimonials')
    clean_docs = [[parag.clean_data() for parag in doc] for doc in filter_docs]

    clean_docs, quechua = work_data.extract_quechua(clean_docs)
    logger.info('Testimonials in spanish {0} - quechua {1}'.format(len(clean_docs), len(quechua)))
    print([doc[0].name for doc in quechua])

    tagged = not_tagged = 0
    for i in clean_docs:
        for par in i:
            if par.sentiment != 'none':
                tagged += 1
            else:
                not_tagged += 1
    logger.info('Paragraphs tagged {0} - not tagged {1}'.format(tagged, not_tagged))

    logger.info('Train model')
    parag_trained1 = work_data.input_sentiment_posneg(filter_docs)
    parag_trained2 = work_data.train_sentiment(clean_docs)

    logger.info('Save preditions')
    parag_trained1.to_csv(output_path + 'reg_database1.csv')
    parag_trained2.to_csv(output_path + 'reg_database2.csv')

    # LDA implementation
    mat_docs, dictionary = work_data.list_to_matrix(clean_docs)
    print(mat_docs[0])
    pprint.pprint(dictionary.dfs)

    lda_model = work_data.lda_model(dictionary, mat_docs, 10)
    print(lda_model)
    pprint.pprint(lda_model.print_topics(num_topics=10, num_words=10))

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()