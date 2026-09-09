import torch
import torch.nn as nn

class DimensionWiseAttentionRatingPrediction(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        embedding_dim=64
    ):
        super(DimensionWiseAttentionRatingPrediction, self).__init__()
        self.embedding_dim = embedding_dim
        self.attention_projection = nn.Linear(
            2 * embedding_dim,
            embedding_dim
        )
        self.user_bias = nn.Parameter(
            torch.zeros(num_users)
        )
        self.item_bias = nn.Parameter(
            torch.zeros(num_items)
        )
        self.global_bias = nn.Parameter(
            torch.tensor(0.0)
        )
    def forward(
        self,
        user_embeddings,
        item_embeddings,
        user_indices,
        item_indices
    ):
        pair_representation = torch.cat(
            [user_embeddings, item_embeddings],
            dim=1
        )
        attention = torch.sigmoid(
            self.attention_projection(
                pair_representation
            )
        )
        interaction = (
            user_embeddings
            * item_embeddings
        )
        weighted_interaction = (
            attention
            * interaction
        )
        interaction_score = weighted_interaction.sum(
            dim=1
        )
        user_bias = self.user_bias[user_indices]
        item_bias = self.item_bias[item_indices]

        score = (
            interaction_score
            + user_bias
            + item_bias
            + self.global_bias
        )
        prediction = torch.sigmoid(score)
        return prediction, attention, weighted_interaction