import numpy as np
import pickle
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
    pkl_path = './data/pickle/'

    convert_files = False

    logger.info('Beginning execution')
    if convert_files:
        logger.info('Coonverting files: {0}'.format(convert_files))
        logging.getLogger().setLevel(30)
        all_docs = work_data.convert_pdf_to_txt(pdf_path, txt_path)
        logging.getLogger().setLevel(10)
        testimonies = open(pkl_path + 'testimonies.pkl', 'wb')
        pickle.dump(all_docs, testimonies)
        testimonies.close()
    else:
        testimonies = open(pkl_path + 'testimonies.pkl', 'rb')
        all_docs = pickle.load(testimonies)
        testimonies.close()

    logger.info('Number of testimonies: {0}'.format(len(all_docs)))
    # TODO filter data before cleaning (first ~5 lines are pointless)
    clean_docs = [work_data.clean_data(doc).split() for doc in all_docs]
    print(clean_docs[0])

    mat_docs, dictionary = work_data.list_to_matrix(clean_docs)
    print(mat_docs[0])
    pprint.pprint(dictionary.token2id)
    hi

    lda_model = work_data.lda_model(dictionary, mat_docs)
    print(lda_model)
    pprint.pprint(lda_model.print_topics(num_topics=3, num_words=10))

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()