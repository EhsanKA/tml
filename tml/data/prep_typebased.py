import numpy as np


def prep_typebased(path_full, cols):
    print("Start Loading Data!")

    dataset = iter_loadtxt(path_full, usecols=cols, dtype='S20')
    names = iter_loadtxt(path_full, usecols=range(2), dtype='S20')

    print("Loading Data Done!")
    print("Start Preparing Data!")

    snp_unq_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == b'SNP-Unq']
    snp_hm_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == b'SNP-Hm']
    snp_ht_h_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == b'SNP-Ht-H']
    snp_ht_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == b'SNP-Ht']
    snp_sm_ind = [index for index, data_type in enumerate(dataset[:, 0]) if data_type == b'SNP-Somatic']

    neg_ind = snp_unq_ind
    pos_ind = snp_ht_h_ind + snp_ht_ind
    test_ind = pos_ind + neg_ind
    pos_ind = np.sort(pos_ind)
    test_ind = np.sort(test_ind)
    rest_pos_ind = snp_sm_ind + snp_hm_ind
    hpos_ind = snp_ht_h_ind + snp_hm_ind

    # Replacing the names with labels
    dataset[pos_ind, 0] = 1
    dataset[neg_ind, 0] = 0
    dataset[rest_pos_ind, 0] = 1

    dataset = np.float64(dataset)

    print("Data Preparation Done!")

    output_dict = {
        "all_set": dataset,
        "test_ind": test_ind,
        "neg_ind": neg_ind,
        "pos_ind": pos_ind,
        "hpos_ind": hpos_ind,
        "names": names
        }
    
    return output_dict

# Helper function to load large CSV files iteratively
def iter_loadtxt(filename, usecols=None, delimiter=',', skiprows=0, dtype=np.float32):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                if usecols is not None:
                    line = [line[i] for i in usecols]
                for item in line:
                    yield item
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data