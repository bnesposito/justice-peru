import numpy as np
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

    logger.info('Remove protocol paragraphs')
    clean_docs = [doc.filter_protocol() for doc in all_docs]

    logger.info('Create wordcloud')
    work_data.generate_wordcloud(all_docs, output_path)

    logger.info('Number of testimonials: {0}'.format(len(all_docs)))
    logger.info('Clean testimonies')
    clean_docs = [doc.clean_data() for doc in clean_docs]
    print(clean_docs[0])
    print(clean_docs[0].text)

    clean_docs, quechua = work_data.extract_quechua(clean_docs)
    print(len(clean_docs), len(quechua))
    print([x.name for x in quechua])
    hi

    mat_docs, dictionary = work_data.list_to_matrix(clean_docs)
    print(mat_docs[0])
    pprint.pprint(dictionary.dfs)
    hi

    lda_model = work_data.lda_model(dictionary, mat_docs, 10)
    print(lda_model)
    pprint.pprint(lda_model.print_topics(num_topics=10, num_words=10))

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()