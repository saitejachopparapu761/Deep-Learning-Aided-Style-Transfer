from Evaluation import evaluation, net_evaluation
from Global_Vars import Global_Vars
import cv2 as cv
import numpy as np
from scipy.ndimage import variance
from scipy.stats import entropy

from Model_BiLSTM import Model_BiLSTM
from Model_CNN import Model_CNN
from Model_DBN import Model_DBN
from OTSU import Hist, threshold, get_threshold
from FCM import FCM


def Obj_fun_FCM(Soln):
    Images = Global_Vars.Images

    dimension = len(Soln.shape)
    if dimension == 1:
        fitness = np.zeros((Soln.shape[0], 1))
        for iter in range(Soln.shape[0]):
            sol = Soln[iter, :]
            Fitn = []
            for iteration in range(len(Images)):
                image = Images[iteration]
                output = np.zeros(image.shape, dtype=np.uint8)
                h = Hist(image)
                values = cv.threshold(h)
                thresh_value = get_threshold(values)
                ret, thresh = cv.threshold(image, thresh_value, sol[3], cv.THRESH_BINARY_INV)
                index = np.where(thresh == 255)
                thresh[index[0], index[1]] = image[index[0], index[1]]
                cluster = FCM(thresh, image_bit=8, n_clusters=5, m=round(sol[0]), epsilon=sol[1],
                              max_iter=round(sol[2]))
                cluster.form_clusters()
                result = cluster.result.astype('uint8') * 50
                uniq, counts = np.unique(result, return_counts=True)
                count_sort = np.argsort(counts)
                indices = np.where(result == uniq[count_sort[1]])
                output[indices[0], indices[1]] = 255
                kernel = np.ones((5, 5), np.uint8)
                closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)
                connect_comp = cv.connectedComponentsWithStats(closing, cv.CV_32S)
                (numLabels, labels, stats, centroids) = connect_comp
                cc_uniq, cc_count = np.unique(labels, return_counts=True)
                cc_sort = np.argsort(cc_count)
                index_0 = np.where(labels == cc_sort[cc_sort.shape[0] - 2])
                labels[index_0[0], index_0[1]] = 0
                index_1 = np.where(labels != 0)
                labels[index_1[0], index_1[1]] = 255
                labels = labels.astype('uint8')
                Eval = net_evaluation(labels, output)
                Fitn.append(Eval[4] + Eval[6])
            Fitn = np.asarray(Fitn)
            fitness[iter] = np.mean(Fitn)
        return fitness
    else:
        sol = Soln
        Fitn = []
        for iteration in range(len(Images)):
            image = Images[iteration]
            output = np.zeros(image.shape, dtype=np.uint8)
            h = Hist(image)
            values = threshold(h)
            thresh_value = get_threshold(values)
            ret, thresh = cv.threshold(image, thresh_value, sol[3], cv.THRESH_BINARY_INV)
            index = np.where(thresh == 255)
            thresh[index[0], index[1]] = image[index[0], index[1]]
            cluster = FCM(thresh, image_bit=8, n_clusters=5, m=round(sol[0]), epsilon=sol[1], max_iter=round(sol[2]))
            cluster.form_clusters()
            result = cluster.result.astype('uint8') * 50
            uniq, counts = np.unique(result, return_counts=True)
            count_sort = np.argsort(counts)
            indices = np.where(result == uniq[count_sort[1]])
            output[indices[0], indices[1]] = 255
            kernel = np.ones((5, 5), np.uint8)
            closing = cv.morphologyEx(output, cv.MORPH_CLOSE, kernel)
            connect_comp = cv.connectedComponentsWithStats(closing, cv.CV_32S)
            (numLabels, labels, stats, centroids) = connect_comp
            cc_uniq, cc_count = np.unique(labels, return_counts=True)
            cc_sort = np.argsort(cc_count)
            index_0 = np.where(labels == cc_sort[cc_sort.shape[0] - 2])
            labels[index_0[0], index_0[1]] = 0
            index_1 = np.where(labels != 0)
            labels[index_1[0], index_1[1]] = 255
            labels = labels.astype('uint8')
            Eval = net_evaluation(labels, output)
            Fitn.append(Eval[4] + Eval[6])
        Fitn = np.asarray(Fitn)
        Fitness = np.mean(Fitn)
        return Fitness

def objfun_cls(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Tar = np.reshape(Tar, (-1, 1))
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred_lstm = Model_BiLSTM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval, pred_cnn = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            predict = (pred_lstm + pred_cnn) / 2
            Eval = evaluation(predict, Test_Target)
            Fitn[i] = (1 / Eval[4]) + Eval[8] + Eval[11]
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred_lstm = Model_BiLSTM(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval, pred_cnn = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        predict = (pred_lstm + pred_cnn) / 2
        Eval = evaluation(predict, Test_Target)
        Fitn = (1 / Eval[4]) + Eval[8] + Eval[11]
        return Fitn


def objfun(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Tar = np.reshape(Tar, (-1, 1))
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Feat[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Feat[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, predict = Model_DBN(Train_Data, Train_Target, Test_Target, sol)
            Eval = evaluation(predict, Test_Target)
            Fitn[i] = (1 / Eval[4]) + Eval[8] + Eval[11]
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Feat[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Feat[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, predict = Model_DBN(Train_Data, Train_Target, Test_Target, sol)
        Eval = evaluation(predict, Test_Target)
        Fitn = (1 / Eval[4]) + Eval[8] + Eval[11]
        return Fitn

