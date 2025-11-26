from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, confusion_matrix
import cv2 as cv


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'TSO', 'LSO', 'HHOA', 'EEO', 'HEEO']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for i in range(1):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='TSO-AWDBN ')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='LSO-AWDBN ')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='HHSO-AWDBN ')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='EO-AWDBN ')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='EEO-AWDBN ')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['LSTM', 'CNN', 'Bi-LSTM', 'DBN', 'CNN+Bi-LSTM+DBN', 'EEO-AWDBN ']
    for a in range(1):  # For 5 Datasets
        # Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        Actual = np.load('Target.npy', allow_pickle=True)

        colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score.npy', allow_pickle=True)[a][i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_results_kfold():
    eval1 = np.load('Eval_all_KFold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 3, 4, 6]
    Algorithm = ['TERMS', 'TSO', 'LSO', 'HHOA', 'EEO', 'PROPOSED']
    Classifier = ['TERMS', 'LSTM', 'CNN', 'BiLSTM', 'DBN', 'CNN + Bi-LSTM + DBN', 'PROPOSED']
    for i in range(eval1.shape[0]):
        value1 = eval1[i, 4, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, :])
        print('-------------------------------------------------- KFOLD - 5 FOLD-Dataset', i + 1,
              'Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, :])
        print('-------------------------------------------------- KFOLD - 5 FOLD-Dataset', i + 1,
              'Classifier Comparison',
              '--------------------------------------------------')
        print(Table)

    learnper = [1, 2, 3, 4, 5]
    for i in range(eval1.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros((eval1.shape[1], eval1.shape[2]))
            for k in range(eval1.shape[1]):
                for l in range(eval1.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval1[i, k, l, Graph_Terms[j] + 4] * 100

            plt.plot(learnper, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="TSO-AWDBN ")
            plt.plot(learnper, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="LSO-AWDBN ")
            plt.plot(learnper, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="HHSO-AWDBN ")
            plt.plot(learnper, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                     label="EO-AWDBN ")
            plt.plot(learnper, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                     label="UEEO-AIDLNet")
            plt.xlabel('KFOLD')
            plt.xticks(learnper, ('1', '2', '3', '4', '5'))
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_line_kfold.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="LSTM")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="CNN")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="Bi-LSTM")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="DBN")
            ax.bar(X + 0.40, Graph[:, 9], color='y', width=0.10, label="CNN+Bi-LSTM+DBN")
            ax.bar(X + 0.50, Graph[:, 10], color='k', width=0.10, label="UEEO-AIDLNet")
            plt.xticks(X + 0.10, ('1', '2', '3', '4', '5'))
            plt.xlabel('KFOLD')
            plt.ylabel(Terms[Graph_Terms[j]])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_bar_kfold.png" % (i + 1, Terms[Graph_Terms[j]])
            plt.savefig(path1)
            plt.show()


import seaborn as sns


def Plot_Confusion():
    Actual = np.load('Actual.npy', allow_pickle=True)
    Predict = np.load('Predict.npy', allow_pickle=True)
    no_of_Dataset = 1
    for n in range(no_of_Dataset):
        ax = plt.subplot()
        cm = confusion_matrix(np.asarray(Actual[n]).argmax(axis=1), np.asarray(Predict[n]).argmax(axis=1))
        sns.heatmap(cm, annot=True, fmt='g',
                    ax=ax)
        path = "./Results/Confusion_%s.png" % (n + 1)
        plt.title('Accuracy')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig(path)
        plt.show()


def Sample_images():
    Plant = ['Apple', 'Cherry', 'Citrus', 'Corn', 'Grape', 'Peach', 'Pepper', 'Potato', 'Strawberry', 'Tomato']
    for n in range((1)):
        Images = np.load('Img_2.npy', allow_pickle=True)
        Image = [8, 9, 10, 11, 12, 13]
        # for i in range(len(Image)):
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("Sample Images from Lung Dataset", fontsize=20)
        plt.subplot(2, 3, 1)
        plt.title('Image-1')
        plt.imshow(Images[Image[0]])
        plt.subplot(2, 3, 2)
        plt.title('Image-2')
        plt.imshow(Images[Image[1]])
        plt.subplot(2, 3, 3)
        plt.title('Image-3')
        plt.imshow(Images[Image[2]])
        plt.subplot(2, 3, 4)
        plt.title('Image-4')
        plt.imshow(Images[Image[3]])
        plt.subplot(2, 3, 5)
        plt.title('Image-5')
        plt.imshow(Images[Image[4]])
        plt.subplot(2, 3, 6)
        plt.title('Image-6')
        plt.imshow(Images[Image[5]])
        # path1 = "./Results_1/Class5/Dataset_%simage.png" % (i + 1)
        # plt.savefig(path1)
        plt.show()
        cv.imwrite('./Results/Sample_Images/Image-1.png', Images[Image[0]])
        cv.imwrite('./Results/Sample_Images/Image-2.png', Images[Image[1]])
        cv.imwrite('./Results/Sample_Images/Image-3.png', Images[Image[2]])
        cv.imwrite('./Results/Sample_Images/Image-4.png', Images[Image[3]])
        cv.imwrite('./Results/Sample_Images/Image-5.png', Images[Image[4]])
        # cv.imwrite('./Results/Image_Results/Plant' + str(n + 1) + '-Seg-Abnormal' + str(i + 1) + '.png',
        #            SegImg1[Image[i]])


def plot_results():
    Eval_all = np.load('Eval_all.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD', 'VARIANCE']
    Algorithm = ['TERMS', 'TSO', 'LSO', 'HHOA', 'EEO', 'PROPOSED']
    Methods = ['TERMS', 'FCM', 'REGION-GROWING', 'FCN+REGION-GROWING', 'PROPOSED']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 0, :], color='r', width=0.10, label="TSO-FCM-Region growing")
            ax.bar(X + 0.10, stats[i, 1, :], color='g', width=0.10, label="LSO-FCM-Region growing")
            ax.bar(X + 0.20, stats[i, 2, :], color='b', width=0.10, label="HHSO-FCM-Region growing")
            ax.bar(X + 0.30, stats[i, 3, :], color='m', width=0.10, label="EO-FCM-Region growing")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="EEO-FCM+Region growing")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
            #            ncol=2, fancybox=True, shadow=True)
            plt.legend(loc=10)
            path1 = "./Results/Dataset_%s_%s_alg.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 1, :], color='c', width=0.10, label="CGAN")
            ax.bar(X + 0.10, stats[i, 5, :], color='r', width=0.10, label="FCM")
            ax.bar(X + 0.20, stats[i, 6, :], color='g', width=0.10, label="Region growing")
            ax.bar(X + 0.30, stats[i, 7, :], color='m', width=0.10, label="FCM+Region growing")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="EEO-FCM+Region growing")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            #            ncol=3, fancybox=True, shadow=True)
            plt.legend(loc=10)
            path1 = "./Results/Dataset_%s_%s_met.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()



def plot_results_kfold1():
    for a in range(1):
        Eval =np.load('Evaluate_all.npy',allow_pickle=True)[a]

        Terms = ['PSNR','SNR','MSE']
        for b in range(len(Terms)):
            learnper = [1, 2, 3, 4]

            X = np.arange(5)
            plt.plot(learnper, Eval[:, 0,b], color='#aaff32', linewidth=3, marker='o', markerfacecolor='#aaff32', markersize=14,
                     label="TSO-AIDLNet")
            plt.plot(learnper, Eval[:, 1,b], color='#ad03de', linewidth=3, marker='o', markerfacecolor='#ad03de', markersize=14,
                     label="LSO-AIDLNet")
            plt.plot(learnper, Eval[:, 2,b], color='#8c564b', linewidth=3, marker='o', markerfacecolor='#8c564b', markersize=14,
                     label="HHSO-AIDLNet")
            plt.plot(learnper, Eval[:, 3,b], color='#ff000d', linewidth=3, marker='o', markerfacecolor='#ff000d', markersize=14,
                     label="EO-AIDLNet")
            plt.plot(learnper, Eval[:, 4,b], color='k', linewidth=3, marker='o', markerfacecolor='k', markersize=14,
                     label="UEEO-AIDLNet")

            labels = ['gaussian\nnoise', 'speckle\nnoise', 'salt and pepper\nnoise', 'poisson\nnoise']
            plt.xticks(learnper, labels)

            # plt.xlabel('BLOCK SIZE')
            plt.ylabel(Terms[b])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_noise_line.png" % (a + 1, Terms[b])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(4)
            ax.bar(X + 0.00, Eval[:, 5,b], color='#aaff32', width=0.10, label="LSTM")
            ax.bar(X + 0.10, Eval[:, 6,b], color='#ad03de', width=0.10, label="CNN")
            ax.bar(X + 0.20, Eval[:, 7,b], color='#8c564b', width=0.10, label="Bi-LSTM")
            ax.bar(X + 0.30, Eval[:, 8,b], color='#ff000d', width=0.10, label="DBN")
            ax.bar(X + 0.40, Eval[:, 2,b], color='c', width=0.10, label="CNN+Bi-LSTM+DBN")
            ax.bar(X + 0.50, Eval[:, 9,b], color='k', width=0.10, label="UEEO-AIDLNet")
            # plt.xticks(X + 0.25, ('5', '10', '15', '20', '25'))


            labels = ['gaussian\nnoise', 'speckle\nnoise', 'salt and pepper\nnoise', 'poisson\nnoise']
            plt.xticks(X, labels)
            # plt.xlabel('BLOCK SIZE')
            plt.ylabel(Terms[b])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset_%s_%s_noise_bar.png" % (a + 1, Terms[b])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    plot_results_kfold()
    Plot_ROC_Curve()
    plotConvResults()
    Plot_Confusion()
    Sample_images()
    plot_results()
    plot_results_kfold1()
