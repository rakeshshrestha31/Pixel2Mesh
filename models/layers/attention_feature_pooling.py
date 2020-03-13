##
#  @author Rakesh Shrestha, rakeshs@sfu.ca

import torch
import torch.nn as nn
from models.layers.multihead_attention import MultiheadAttention

class AttentionFeaturePooling(nn.Module):
    def __init__(self, input_features_dim, output_features_dim,
                 num_heads, max_views=3, bias=True):
        super(AttentionFeaturePooling, self).__init__()
        self.max_views = max_views
        self.attention = MultiheadAttention(
            output_features_dim, num_heads, bias=bias
        )

        # linear modules ot transform input features to key/value/query
        self.query_linear = nn.Linear(
            input_features_dim, output_features_dim, bias=bias
        )
        self.key_linear = nn.Linear(
            input_features_dim, output_features_dim, bias=bias
        )
        self.value_linear = nn.Linear(
            input_features_dim, output_features_dim, bias=bias
        )

    ##
    #  @param features tensor of dimensions views x batch x features
    def forward(self, features):
        num_views, batch_size, num_input_features = features.size()
        assert(num_views <= self.max_views)

        # pad views to self.max_views
        if num_views < self.max_views:
            features = torch.cat((
                features, torch.zeros_like(features[0:1])
                               .repeat(self.max_views - num_views, 1, 1)
            ), dim=0)

        flattened_features = features.view(-1, features.size(-1))

        # query
        query = self.query_linear(flattened_features)
        query = query.view(num_views, batch_size, -1)
        # the first dim of query should be target sequence size (here 1)
        # taking mean is non-standard (TODO: find a better way)
        query = query.mean(dim=0, keepdim=True)

        # key/value
        key = self.key_linear(flattened_features)
        value = self.value_linear(flattened_features)

        attn_output, attn_weights = self.attention(query, key, value)
        return attn_output, attn_weights

