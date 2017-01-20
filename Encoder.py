#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2



import openface


class encoder:
    def __init__(self, imgDim=96):
        self.imgDim = imgDim
        a = os.path.realpath(__file__)
        self.fileDir = os.path.dirname(os.path.realpath(__file__))
        self.modelDir = os.path.join(self.fileDir, '..', '..', 'models')
        self.openfaceModelDir = os.path.join(self.modelDir, 'openface')
        self.networkModel = os.path.join(self.openfaceModelDir, 'nn4.small2.v1.t7')
        self.net = openface.TorchNeuralNet(self.networkModel, self.imgDim)


    def get_rep_preprocessed(self, imgPath=None):
        rep = self.net.forward(cv2.imread(imgPath))
        return rep

