import pickle
import time
import numpy as np
import sys
from sklearn.mixture import GaussianMixture
import cv2
import openface
import os


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

    def getRep(self, imgPath, multiple=False):
        total_time = float()
        initial_time = time.time()
        start = time.time()
        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))

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
            return TestClassifier.not_found_code, float(), "Unable to find a face: {}".format(imgPath)
            # raise Exception("Unable to find a face: {}".format(imgPath))
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
                raise Exception("Unable to align image: {}".format(imgPath))
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

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)
            else:
                (le, clf) = pickle.load(f, encoding='latin1')
        return le, clf
