from Model_BiLSTM import Model_BiLSTM
from Model_CNN import Model_CNN


def Model_CNN_BiLSTM(Train_Data, Train_Target, Test_Data, Test_Target, sol=None):
    if sol is None:
        sol = [5, 5, 50, 5, 5, 50]
    eval, pred_cnn = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
    eval, pred_bilstm = Model_BiLSTM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
    Pred = (pred_bilstm+pred_cnn)/2
    return eval, Pred