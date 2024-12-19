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

__all__ = ["RatioLogisticClassifier"]

from sklearn.utils import compute_sample_weight
from asreview.models.classifiers.logistic import LogisticClassifier


class RatioLogisticClassifier(LogisticClassifier):
    name = "ratio"

    def __init__(self, ratio=1.0, *args, **kwargs):
        self.ratio = ratio
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        weights = compute_sample_weight(
            {1: 1.0, 0: sum(y == 1) / (self.ratio * sum(y == 0))}, y=y
        )
        sample_weight = weights * (len(y) / sum(weights))
        return self._model.fit(X, y, sample_weight=sample_weight)
