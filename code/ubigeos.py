import pandas as pd
import numpy as np
import time

import config


def add_zero_left(number, length):
    gap = length - len(str(number))
    output = str(number)
    for i in range(gap):
        output = '0' + output
    return output


def main():
    # Objective: create a csv file with the ubigeo code of all centro poblados in Peru.
    logger = config.config_logger(__name__, 10)
    t0 = time.time()

    logger.info('Beginning execuation')
    raw = pd.read_csv('./data/ubigeo/Ubigeo_mod.csv', header=0)
    districts = raw.loc[raw['district'].apply(np.isnan)]
    print(districts.shape)
    departments = raw.loc[~raw['region'].apply(np.isnan)]
    n_centro_pob = sum(departments['centro_pob'])
    logger.info('There are {0:,} centros poblados in Peru'.format(n_centro_pob))

    output = []
    for index, row in districts.iterrows():
        temp_centros = row['centro_pob']
        temp_code = row['code']
        for centro in range(1, temp_centros+1):
            centro_code = add_zero_left(centro, 4)
            centro_code = str(temp_code) + centro_code
            centro_code = add_zero_left(centro_code, 10)
            output.append(centro_code)

    print(len(output))  # This shoul be equal to the number of centros poblados.

    print(raw.head().to_string())
    print(districts.head().to_string())
    print(departments.head().to_string())

    output = pd.DataFrame({'ubigeo': output})
    output.to_csv('./data/ubigeo/ubigeo_final.csv')

    config.time_taken_display(t0)


if __name__ == '__main__':
    main()
