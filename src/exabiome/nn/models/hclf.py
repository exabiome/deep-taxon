import torch
import torch.nn as nn

class HierarchicalClassifier(nn.Module):

    def __init__(self, in_features, taxa_table):
        super().__init__()

        in_feats = in_features
        out_feats = len(taxa_table.phylum_elements)
        self.phylum_fc = nn.Linear(in_feats, out_feats)

        in_feats = in_features + out_feats
        out_feats = len(taxa_table.class_elements)
        self.class_fc = nn.Linear(in_feats, out_feats)

        in_feats = in_features + out_feats
        out_feats = len(taxa_table.order_elements)
        self.order_fc = nn.Linear(in_feats, out_feats)

        in_feats = in_features + out_feats
        out_feats = len(taxa_table.family_elements)
        self.family_fc = nn.Linear(in_feats, out_feats)

        in_feats = in_features + out_feats
        out_feats = len(taxa_table.genus_elements)
        self.genus_fc = nn.Linear(in_feats, out_feats)

        in_feats = in_features + out_feats
        out_feats = len(taxa_table.species)
        self.species_fc = nn.Linear(in_feats, out_feats)

    def forward(self, x):

        feats = x

        ret = list()

        out = self.phylum_fc(x)
        x = torch.cat([out, feats], dim=1)
        ret.append(out)

        out = self.class_fc(x)
        x = torch.cat([out, feats], dim=1)
        ret.append(out)

        out = self.order_fc(x)
        x = torch.cat([out, feats], dim=1)
        ret.append(out)

        out = self.family_fc(x)
        x = torch.cat([out, feats], dim=1)
        ret.append(out)

        out = self.genus_fc(x)
        x = torch.cat([out, feats], dim=1)
        ret.append(out)

        out = self.species_fc(x)
        ret.append(out)

        out = torch.cat(ret, dim=1)
        return out

class HierarchicalLoss(nn.Module):

    def __init__(self, taxa_table):
        super().__init__()

        ar = [ len(taxa_table.phylum_elements),
               len(taxa_table.class_elements),
               len(taxa_table.order_elements),
               len(taxa_table.family_elements),
               len(taxa_table.genus_elements),
               len(taxa_table.species) ]
        self.idx = torch.Tensor(ar).cumsum(0).int()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, labels):
        s = 0
        loss = 0
        for e in ar:
            tmp_outputs = x[:, s:e]
            tmp_labels = labels[s:e]
            loss += self.loss(tmp_outputs, tmp_labels)
        return loss
