import openface
import os
import cv2
from openface.data import iterImgs
import random
import shutil
# import openface.helper


class TestAlign:

    def __init__(self, verbose, dir_output, landmark):
        print(os.getcwd())
        fileDir = os.path.dirname(os.path.realpath(__file__))
        modelDir = os.path.join(fileDir, '..', 'models')
        dlibModelDir = os.path.join(modelDir, 'dlib')
        openfaceModelDir = os.path.join(modelDir, 'openface')
        self.verbose = verbose
        self.dir_output = dir_output
        self.landmark = landmark
        self.dlib_face_predictor = os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat")

    def alignMain(self, path_img, skipMulti=False, size=96, fallbackLfw=None):
        if os.path.isfile(self.dir_output+"/cache.t7"):
            os.remove(self.dir_output+"/cache.t7")
        openface.helper.mkdirP(self.dir_output)

        imgs = list(iterImgs(path_img))

        # Shuffle so multiple versions can be run at once.
        random.shuffle(imgs)

        landmarkMap = {
            'outerEyesAndNose': openface.AlignDlib.OUTER_EYES_AND_NOSE,
            'innerEyesAndBottomLip': openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP
        }
        if self.landmark not in landmarkMap:
            raise Exception("Landmarks unrecognized: {}".format(self.landmark))

        landmarkIndices = landmarkMap[self.landmark]

        align = openface.AlignDlib(self.dlib_face_predictor)

        nFallbacks = 0
        for imgObject in imgs:
            print("=== {} ===".format(imgObject.path))
            outDir = os.path.join(self.dir_output, imgObject.cls)
            openface.helper.mkdirP(outDir)
            outputPrefix = os.path.join(outDir, imgObject.name)
            imgName = outputPrefix + ".png"

            if os.path.isfile(imgName):
                if self.verbose:
                    print("  + Already found, skipping.")
            else:
                rgb = imgObject.getRGB()
                if rgb is None:
                    if self.verbose:
                        print("  + Unable to load.")
                    outRgb = None
                else:
                    outRgb = align.align(size, rgb,
                                         landmarkIndices=landmarkIndices,
                                         skipMulti=skipMulti)
                    if outRgb is None and self.verbose:
                        print("  + Unable to align.")

                if fallbackLfw and outRgb is None:
                    nFallbacks += 1
                    deepFunneled = "{}/{}.jpg".format(os.path.join(fallbackLfw,
                                                                   imgObject.cls),
                                                      imgObject.name)
                    shutil.copy(deepFunneled, "{}/{}.jpg".format(os.path.join(self.dir_output,
                                                                              imgObject.cls),
                                                                 imgObject.name))

                if outRgb is not None:
                    if self.verbose:
                        print("  + Writing aligned file to disk.")
                    outBgr = cv2.cvtColor(outRgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(imgName, outBgr)

        if fallbackLfw:
            print('nFallbacks:', nFallbacks)