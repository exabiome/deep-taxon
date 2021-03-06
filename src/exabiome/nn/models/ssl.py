from argparse import Namespace
import warnings

import torch
import torch.nn as nn


from . import model, AbstractLit
from .resnet_feat import ResNetFeat


@model("resnet_clf")
class ResNetClassifier(AbstractLit):
    """A model that adds a classification layer to a pre-trained ResNet feature model"""

    def __init__(self, hparams):
        self.hparams = self.check_hparams(hparams)
        self.hparams.classify = True
        super().__init__(self.hparams)

        self.set_inference(False)
        self.lr = getattr(hparams, 'lr', None)

        if hasattr(self.hparams, 'features_checkpoint') and self.hparams.features_checkpoint is not None:
            breakpoint()
            self._setup_features(self.hparams.features_checkpoint)
            self._setup_classifier()

    def _setup_classifier(self):
        n_inputs = 512 * self.features.layer4[-1].expansion
        n_outputs = self.hparams.n_outputs
        sizes = [n_inputs]
        sizes.extend(getattr(self.hparams, 'hidden_layers', list()))
        sizes.append(n_outputs)
        layers = list()
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(sizes[i-1], sizes[i]))
        self.classifier = nn.Sequential(*layers)

    def _setup_features(self, features_checkpoint):
        from . import _models

        feat_model_hparams = torch.load(features_checkpoint, map_location=torch.device('cpu'))['hyper_parameters']
        feat_model_cls = _models[feat_model_hparams['model']]
        self.hparams.feat_model_hparams  = feat_model_hparams
        if not issubclass(feat_model_cls, ResNetFeat):
            raise ValueError("features_checkpoint must be a checkpoint for a ResNetFeat model")
        self.features = feat_model_cls.load_from_checkpoint(features_checkpoint, hparams=feat_model_hparams)

    def forward(self, x):
        feats = self.features(x)
        outputs = self.classifier(feats)
        return outputs

    def on_load_checkpoint(self, checkpoint):
        self.hparams.features_checkpoint = checkpoint['hyper_parameters']['features_checkpoint']
        self._setup_features(self.hparams.features_checkpoint)
        self._setup_classifier()
