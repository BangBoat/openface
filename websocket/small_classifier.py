import pickle
import time
import numpy as np
import sys
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import cv2
import openface
import os
from operator import itemgetter
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class TestClassifier:
    imgDim = 96
    not_found_code = 404

    def __init__(self, verbose):
        print(os.getcwd())
        fileDir = os.path.dirname(os.path.realpath(__file__))
        modelDir = os.path.join(fileDir, '..', 'models')
        dlibModelDir = os.path.join(modelDir, 'dlib')
        openfaceModelDir = os.path.join(modelDir, 'openface')
        self.verbose = verbose
        self.align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
        self.net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
                                           , imgDim=TestClassifier.imgDim
                                           , cuda=False)

    def getRep(self, path_img, multiple=False):
        total_time = float()
        initial_time = time.time()
        start = time.time()
        bgrImg = cv2.imread(path_img)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(path_img))

        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.verbose:
            print("  + Original size: {}".format(rgbImg.shape))
        if self.verbose:
            print("Loading the image took {} seconds.".format(time.time() - start))

        start = time.time()

        if multiple:
            bbs = self.align.getAllFaceBoundingBoxes(rgbImg)
        else:
            bb1 = self.align.getLargestFaceBoundingBox(rgbImg)
            bbs = [bb1]
        if len(bbs) == 0 or (not multiple and bb1 is None):
            return TestClassifier.not_found_code, float(), "Unable to find a face: {}".format(os.path.basename(path_img))
            # raise Exception("Unable to find a face: {}".format(path_img))
        if self.verbose:
            print("Face detection took {} seconds.".format(time.time() - start))

        reps = []
        for bb in bbs:
            start = time.time()
            alignedFace = self.align.align(
                TestClassifier.imgDim,
                rgbImg,
                bb,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if alignedFace is None:
                raise Exception("Unable to align image: {}".format(path_img))
            if self.verbose:
                print("Alignment took {} seconds.".format(time.time() - start))
                print("This bbox is centered at {}, {}".format(bb.center().x, bb.center().y))

            start = time.time()
            rep = self.net.forward(alignedFace)
            if self.verbose:
                print("Neural network forward pass took {} seconds.".format(
                    time.time() - start))
            reps.append((bb.center().x, rep))
        sreps = sorted(reps, key=lambda x: x[0])
        total_time = time.time() - initial_time
        return sreps, total_time, ""

    def infer(self, le, clf, img, multiple=False):
        # for img in img:
        #     print("\n=== {} ===".format(img))
        #     reps = self.getRep(img, multiple)
        print("\n=== {} ===".format(img))
        reps, total_time, error = self.getRep(img, multiple)
        if type(reps) is int:
            if reps == TestClassifier.not_found_code:
                return {"error": error}
        dict_output = dict()
        if len(reps) > 1:
            dict_output.update({"multiple": "List of faces in image from left to right"})
        number = int()
        total_prediction_time = float()

        for r in reps:
            number += 1
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform(maxI)
            confidence = predictions[maxI]
            dict_time = dict()
            if self.verbose:
                prediction_time = time.time() - start
                total_prediction_time = prediction_time + total_prediction_time
                print("Prediction took {} seconds.".format(prediction_time))
                dict_time = {"prediction_time": "{0:.5f}".format(prediction_time)}
            if multiple:
                print("Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,
                                                                         confidence))
                dict_result = {"result": "Predict {} @ x={} with {:.2f} confidence.".format(person, bbx,
                                                                                            confidence)}
            else:
                print("Predict {} with {:.2f} confidence.".format(person, confidence))
                dict_result = {"result": "Predict {} with {:.2f} confidence.".format(person, confidence)}
            if isinstance(clf, GaussianMixture):
                dist = np.linalg.norm(rep - clf.means_[maxI])
                print("  + Distance from the mean: {}".format(dist))
            if len(dict_time) == 1:
                dict_result.update(dict_time)
            dict_output.update({number: dict_result})
        if total_prediction_time > 0:
            total_time += total_prediction_time
        dict_output.update({"total_time": "{0:.5f}".format(total_time)})
        return dict_output

    def load_model(self, path_model):
        with open(path_model, 'rb') as f:
            if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
            else:
                (le, clf) = pickle.load(f, encoding='latin1')
        return le, clf

    def train(self, clf, dir_work, classifier="LinearSvm", dim_lda=-1):
        print("Loading embeddings.")
        fname = "{}/labels.csv".format(dir_work)
        labels = pd.read_csv(fname, header=None).as_matrix()[:, 1]
        labels = list(map(itemgetter(1), map(os.path.split, map(os.path.dirname, labels))))  # Get the directory.
        fname = "{}/reps.csv".format(dir_work)
        embeddings = pd.read_csv(fname, header=None).as_matrix()
        le = LabelEncoder().fit(labels)
        labelsNum = le.transform(labels)
        nClasses = len(le.classes_)
        print("Training for {} classes.".format(nClasses))
    
        if classifier == 'LinearSvm':
            clf = SVC(C=1, kernel='linear', probability=True)
        elif classifier == 'GridSearchSvm':
            print("""
            Warning: In our experiences, using a grid search over SVM hyper-parameters only
            gives marginally better performance than a linear SVM with C=1 and
            is not worth the extra computations of performing a grid search.
            """)
            param_grid = [
                {'C': [1, 10, 100, 1000],
                 'kernel': ['linear']},
                {'C': [1, 10, 100, 1000],
                 'gamma': [0.001, 0.0001],
                 'kernel': ['rbf']}
            ]
            clf = GridSearchCV(SVC(C=1, probability=True), param_grid, cv=5)
        elif classifier == 'GMM':  # Doesn't work best
            clf = GaussianMixture(n_components=nClasses)
    
        # ref:
        # http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#example-classification-plot-classifier-comparison-py
        elif classifier == 'RadialSvm':  # Radial Basis Function kernel
            # works better with C = 1 and gamma = 2
            clf = SVC(C=1, kernel='rbf', probability=True, gamma=2)
        elif classifier == 'DecisionTree':  # Doesn't work best
            clf = DecisionTreeClassifier(max_depth=20)
        elif classifier == 'GaussianNB':
            clf = GaussianNB()
    
        # ref: https://jessesw.com/Deep-Learning/
        elif classifier == 'DBN':
            from nolearn.dbn import DBN
            clf = DBN([embeddings.shape[1], 500, labelsNum[-1:][0] + 1],  # i/p nodes, hidden nodes, o/p nodes
                      learn_rates=0.3,
                      # Smaller steps mean a possibly more accurate result, but the
                      # training will take longer
                      learn_rate_decays=0.9,
                      # a factor the initial learning rate will be multiplied by
                      # after each iteration of the training
                      epochs=300,  # no of iternation
                      # dropouts = 0.25, # Express the percentage of nodes that
                      # will be randomly dropped as a decimal.
                      verbose=1)
    
        if dim_lda > 0:
            clf_final = clf
            clf = Pipeline([('lda', LDA(n_components=dim_lda)),
                            ('clf', clf_final)])
    
        clf.fit(embeddings, labelsNum)
    
        fName = "{}/classifier.pkl".format(dir_work)
        print("Saving classifier to '{}'".format(fName))
        with open(fName, 'wb') as f:
            pickle.dump((le, clf), f)
        return "training complete"
