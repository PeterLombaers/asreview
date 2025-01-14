# Copyright 2019-2022 The ASReview Authors. All Rights Reserved.
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

__all__ = ["LogisticClassifier"]

from sklearn.linear_model import LogisticRegression

from asreview.models.classifiers.base import BaseTrainClassifier


class LogisticClassifier(BaseTrainClassifier):
    """
    Logistic regression classifier (``logistic``).

    The Logistic regressions classifier is an implementation based
    on the sklearn Logistic regressions classifier.

    Parameters
    ----------
    C: float
        Parameter inverse to the regularization strength of the model.
    class_weight: float
        Class weight of the inclusions.
    random_state: int, asreview.utils.SeededRandomState
        Random state for the model.
    """

    name = "logistic"
    label = "Logistic regression"

    def __init__(self, C=1.0, solver="liblinear", random_state=None):
        super().__init__()
        self.C = C

        self._model = LogisticRegression(
            C=C,
            solver=solver,
            random_state=random_state,
        )
