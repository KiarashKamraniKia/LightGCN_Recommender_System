import torch
import torch.nn as nn

class AdaptiveGraphMessagePassing(nn.Module):
  
    def __init__(
        self,
        embedding_dim=64,
        num_layers=3,
        num_prototypes=5
    ):
        super(AdaptiveGraphMessagePassing, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_prototypes = num_prototypes

        # Learnable edge propagation weight
        self.edge_projection = nn.Linear(
            2 * num_prototypes,
            1
        )

    def _compute_propagation_coefficients(
        self,
        assignments
    ):
        num_nodes = assignments.size(0)

        # Build source and target prototype assignments
        source = assignments.unsqueeze(1).expand(
            num_nodes,
            num_nodes,
            -1
        )

        target = assignments.unsqueeze(0).expand(
            num_nodes,
            num_nodes,
            -1
        )

        # Combine assignments for each node pair
        pair_assignments = torch.cat(
            [source, target],
            dim=-1
        )

        # Compute edge propagation coefficients
        alpha = self.edge_projection(
            pair_assignments
        ).squeeze(-1)

        # Map coefficients to [0, 1]
        alpha = torch.sigmoid(alpha)

        return alpha

    def forward(
        self,
        user_embeddings,
        item_embeddings,
        user_assignments,
        item_assignments,
        refined_adj
    ):
        """Perform adaptive message propagation."""

        # Combine user and item embeddings
        all_embeddings = torch.cat(
            [user_embeddings, item_embeddings],
            dim=0
        )

        # Combine prototype assignments
        all_assignments = torch.cat(
            [user_assignments, item_assignments],
            dim=0
        )

        # Compute adaptive propagation weights
        alpha = self._compute_propagation_coefficients(
            all_assignments
        )

        # Keep weights only on existing graph edges
        alpha = alpha * (refined_adj != 0).to(alpha.dtype)

        # NOTE: Degree is computed from the weights of the refined graph
        degrees = torch.sum(
            refined_adj,
            dim=1
        )

        # Avoid division by zero
        degrees = torch.clamp(
            degrees,
            min=1.0
        )

        # Keep embeddings from each propagation depth
        embeddings = [all_embeddings]

        # Propagate messages through the refined graph
        for _ in range(self.num_layers):

            # Compute degree normalization
            normalization = torch.sqrt(
                degrees.unsqueeze(1)
                * degrees.unsqueeze(0)
            )

            # Apply adaptive edge weights and normalization
            normalized_weights = (
                refined_adj * alpha
            ) / normalization

            # Aggregate neighbor messages
            all_embeddings = torch.matmul(
                normalized_weights,
                all_embeddings
            )

            embeddings.append(all_embeddings)

        # Aggregate embeddings from all depths
        embeddings = torch.stack(
            embeddings,
            dim=1
        )

        final_embeddings = torch.mean(
            embeddings,
            dim=1
        )

        return final_embeddings, alpha