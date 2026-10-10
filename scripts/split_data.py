from ml.data_set_splitter import DataSetSplitter
import pickle

if __name__ == '__main__':
    dataset = 'ctg-public-xml'

    with open(f'output/{dataset}.ctdmc', 'rb') as f:
        ctdmc = pickle.load(f)

    path = DataSetSplitter.dump_split_data_set(
        ctdmc,
        f'output/{dataset}.sds',
        frequency_threshold=0
    )

    print('Dataset split saved:', path)