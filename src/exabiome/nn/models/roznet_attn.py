import math
import torch.nn as nn
import torch
import torch.nn.functional as F

from . import model, AbstractLit

class SelfAttention(nn.Module):
    """ Self attention Layer """
    def __init__(self, in_dim):
        super().__init__()
        self.in_channel = in_dim

        self.query = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size= 1)
        self.key = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size= 1)
        self.value = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps(B X C X W)
            returns :
                out : self attention value + input feature (B x C x W)
                attention: softmax scores (B X W X W)
            steps:
            1. create query, key, value 1D conv layers
            2. matmul transposed query (B x W x C/8) by key (B x C/8 x W) = scores (B x W x W)
            3. normalize to sum to 1 with softmax
            4. matmul value (B x C x W) by scores (B x W x W) = out (B x C x W)
        """
        batch_size, conv, width = x.size()
        proj_query = self.query(x).view(batch_size, -1, width).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width)
        scores = torch.bmm(proj_query, proj_key)
        attention = self.softmax(scores) 
        proj_value = self.value(x).view(batch_size, -1, width)
        out = torch.bmm(proj_value, attention)
        out = out.view(batch_size, conv, width)

        out = self.gamma*out + x
        return out, attention

class MultiHeadAttention(AbstractLit):
    """ multi-headed attention Layer """
    def __init__(self, in_dim, activation, num_heads, hparams):
        super().__init__(hparams)
        self.in_channel = in_dim
        self.activation = activation
        self.num_attention_heads = num_heads
        assert in_dim % self.num_attention_heads == 0, "The input size is not a multiple of the number of attention heads"
        self.attention_head_size = int(in_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        self.conv_out = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X W)
            returns :
                out : self attention value + input feature (B x C x W)
                attention: softmax scores (B X W X W)
            steps:

        """
        batch_size, conv, width = x.size()

        proj_query = self.query(x).permute(0, 2, 1)
        proj_key = self.key(x).permute(0, 2, 1)
        proj_value = self.value(x).permute(0, 2, 1)

        query_layer_size = proj_query.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        proj_query = proj_query.view(query_layer_size)
        query_layer = proj_query.permute(0, 2, 1, 3)
        key_layer_size = proj_key.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        proj_key = proj_key.view(key_layer_size)
        key_layer = proj_key.permute(0, 2, 1, 3)
        value_layer_size = proj_value.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        proj_value = proj_value.view(value_layer_size)
        value_layer = proj_value.permute(0, 2, 1, 3)

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        context_layer = context_layer.view(batch_size, conv, width)
        out = self.conv_out(context_layer)
        return out


@model('roznet_attn')
class RozNetAttn(AbstractLit):
    '''
    A 1D CNN with 5 convolutional layers, followed by 3 fully-connected layers

    Args:
        input_nc (int):  the input number of channels
    '''

    def __init__(self, hparams):
        super().__init__(hparams)
        hparams = self.check_hparams(hparams)
        input_nc = getattr(hparams, 'input_nc', None)
        n_outputs = getattr(hparams, 'n_outputs', 2)
        first_kernel_size = getattr(hparams, 'first_kernel_size', 7)
        maxpool = getattr(hparams, 'maxpool', True)
        emb_dim = 8
        self.embedding = nn.Embedding(input_nc, emb_dim)
        self.features = nn.Sequential(
            nn.Conv1d(emb_dim, 64, kernel_size=first_kernel_size, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )
        pool_size = 24

        self.pool = nn.AdaptiveMaxPool1d(pool_size)

        self.classifier = nn.Sequential(
            nn.Dropout(hparams.dropout_rate),
            nn.Linear(256*pool_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(hparams.dropout_rate),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_outputs),
            #nn.BatchNorm1d(n_outputs)
        )

        self.attention = SelfAttention(256)

    def forward(self, x, **kwargs):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.features(x)
        x = self.attention(x)[0]
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



if __name__ == '__main__':
    from .train import parse_args, run, check_model, load_dataset

    args = parse_args("Train CNN with Spatial Pyramidal Pooling")
                      #[['-E', '--emb_nc'], dict(type=int, default=0, help='the number of embedding channels. default is no embedding')])

    input_nc = 5
    if args['protein']:
        input_nc = 26

    if args['sanity']:
        input_nc = 5


    dataset, io = load_dataset(path=args['input'], **args)

    if args['classify']:
        n_outputs = len(dataset.difile.taxa_table)
    else:
        n_outputs = dataset.difile.n_emb_components

    model = check_model(RozNet(input_nc, n_outputs=n_outputs), **args)

    args['pad'] = True

    run(dataset, model, **args)

    io.close()
