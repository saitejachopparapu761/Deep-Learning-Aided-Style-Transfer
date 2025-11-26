#from keras.utils import to_categorical
import numpy as np
from Evaluation import evaluation
from dbn.tensorflow import SupervisedDBNClassification


def DBN(Train_Data, Train_Target):
    classifier = SupervisedDBNClassification(hidden_layers_structure=[0.5, 0.5],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=1,
                                             n_iter_backprop=2,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    for i in range(Train_Target.shape[1]):
        print(i)
        classifier.fit(Train_Data, Train_Target[:, 0])
    weights = classifier.layers[0].get_weights()
    return weights[0]


def Model_DBN(Data, Target, weight_1, soln=None):
    sol = [soln, soln]
    Weight = DBN(Data, Target)
    w = Weight + Weight * (sol[0])
    classifier = SupervisedDBNClassification(hidden_layers_structure=[0.5, 0.5],
                                             learning_rate_rbm=0.05,
                                             learning_rate=0.1,
                                             n_epochs_rbm=1,
                                             n_iter_backprop=2,
                                             batch_size=32,
                                             activation_function='relu',
                                             dropout_p=0.2)
    classifier.layers[0] = w
    model = classifier

    pred = model.predict(w)
    pred = np.asarray(pred)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, Target)
    return Eval, pred
