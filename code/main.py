import numpy as np
import os
import time

import work_data
import config


def main():
    logger = config.config_logger(__name__, 30)
    t0 = time.time()

    pdf_path = './data/pdf/'
    txt_path = './data/txt/'

    logger.info('Beginning execution')
    logger.info('Stage 0: .pdf to .txt')
    logger.info('Pdf path: {0}'.format(pdf_path))
    pdf_list = os.listdir(pdf_path)

    for index, pdf_name in enumerate(pdf_list):
        data = work_data.pdfparser(pdf_path + pdf_name)
        temp_name = os.path.splitext(pdf_name)[0]
        temp = open(txt_path + '{0}.txt'.format(temp_name), 'w')
        temp.write(data)
        temp.close()
        print('{0}. {1} - len: {2}\n{3}'.format(index, temp_name, len(data), data[:100]))

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()