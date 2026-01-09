import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def create_graph_data(X, y, k=5):
    """
    X : numpy array of shape (N, D)
    y : numpy array of shape (N,)
    k : number of nearest neighbors
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    edges = nbrs.kneighbors_graph(X).tocoo()

    # Edge index for PyG
    edge_index = torch.from_numpy(
        np.vstack((edges.row, edges.col))
    ).long()

    # Add self-loops
    num_nodes = X.shape[0]
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_loops = torch.stack([self_loops, self_loops])
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    x = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    return Data(x=x, edge_index=edge_index, y=y_tensor)

def train_gnn(graph_data, input_dim, num_classes, hidden_dim=64, epochs=50, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNN(input_dim, hidden_dim, num_classes).to(device)
    graph_data = graph_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data)
        loss = F.nll_loss(out, graph_data.y)
        loss.backward()
        optimizer.step()

    return model
