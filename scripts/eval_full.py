import pickle
from ml.mse_tester import MseTester
from ml.average_precision_k_tester import AveragePrecisionKTester
from ml.data_set_splitter import DataSetSplitter

with open('./output/model-0.001_1000.0_0.01_0.1_60.pickle', 'rb') as f:
    model = pickle.load(f)

TEST = DataSetSplitter.get_test_utility_matrix('./output/ctg-public-xml.sds')
print("TEST docs:", TEST.shape[1])
print("MSE, MSER:", MseTester(model)(TEST))
print("APTK, APRK:", AveragePrecisionKTester(model)(TEST))