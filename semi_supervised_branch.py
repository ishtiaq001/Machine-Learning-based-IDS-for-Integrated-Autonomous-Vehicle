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
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    edges = nbrs.kneighbors_graph(X).tocoo()
    edge_index = torch.tensor([edges.row, edges.col], dtype=torch.long)
    x = torch.tensor(X, dtype=torch.float)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y_tensor)

def train_gnn(graph_data, input_dim, num_classes, epochs=20, lr=0.01):
    model = GNN(input_dim, 64, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(graph_data)
        loss = F.nll_loss(out, graph_data.y)
        loss.backward()
        optimizer.step()
    return model
