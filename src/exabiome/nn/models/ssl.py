import torch.nn as nn

from . import model, AbstractLit

from .resnet_feat import ResNetFeat


@model("resnet_clf")
class ResNetClassifier(AbstractLit):
    """A model that adds a classification layer to a pre-trained ResNet feature model"""

    def __init__(self, hparams=None, features=None):
        self.hparams = self.check_hparams(hparams)
        self.hparams.classify = True
        super().__init__(self.hparams)

        self.set_inference(False)
        self.lr = getattr(hparams, 'lr', None)

        if features is not None:
            self.features = features
            if not isinstance(self.features, ResNetFeat):
                raise ValueError("features must be an instance of ResNetFeat")
            self.setup_classifier()

    def setup_classifier(self):
        n_inputs = 512 * self.features.layer4[-1].expansion
        n_outputs = self.hparams.n_outputs
        sizes = [n_inputs]
        sizes.extend(getattr(self.hparams, 'hidden_layers', list()))
        sizes.append(n_outputs)
        layers = list()
        for i in range(1, len(sizes)):
            layers.append(nn.Linear(sizes[i-1], sizes[i]))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        feats = self.features(x)
        outputs = self.classifier(feats)
        return outputs

    def on_load_checkpoint(self, checkpoint):
        from . import _models
        from argparse import Namespace

        hparams = Namespace(**checkpoint['hyper_parameters'])
        feat_model_cls = _models[hparams.model]
        self.features = feat_model_cls(hparams)
        self.setup_classifier()
