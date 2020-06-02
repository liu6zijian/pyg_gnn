# -*- coding: utf-8 -*-
"""
@Time   : 2020/6/2

@Author : Zijian Liu
"""
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn
import argparse

# parameters setting
parser = argparse.ArgumentParser()
parser.add_argument("--path", default="best_{}.pth", type=str, help="model save path")
parser.add_argument("--model", default="GraphUnet", type=str, help="select model (gat, gcn, graphsage, sgcn, GraphUnet)")
parser.add_argument("--interval", default=50.0, type=float, help="display and test interval")
parser.add_argument("--max_epoch", default=500, type=int, help="max epoch")
parser.add_argument("--lr", default=1e-2, type=float, help="learning rate")
parser.add_argument("--train",default=False, type=bool, help="train (True) or test (False) mode")
parser.add_argument("--gpu",default="0", type=str, help="determine gpu device number")

args = parser.parse_args()
print(args)

# load dataset
def get_data(folder="node_classify/cora", data_name="cora"):
    dataset = Planetoid(root=folder, name=data_name)
    return dataset

# create the graph cnn model
class GraphCNN(nn.Module):
    def __init__(self, in_c, out_c, hid_c=64):
        super(GraphCNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_c, out_channels=hid_c)
        self.conv2 = pyg_nn.GCNConv(in_channels=hid_c, out_channels=out_c)

    def forward(self, data):
        # data.x data.edge_index
        x = data.x  # [N, C]
        edge_index = data.edge_index  # [2 ,E]
        x = F.dropout(x, training=self.training)

        hid = self.conv1(x=x, edge_index=edge_index)  # [N, D]
        hid = F.dropout(hid, training=self.training)
        hid = F.relu(hid)

        out = self.conv2(x=hid, edge_index=edge_index)  # [N, out_c]

        out = F.log_softmax(out, dim=1)  # [N, out_c]

        return out

# todo list
class GAT_GCN(nn.Module):
    def __init__(self, in_c, out_c, hid_c=64, K=8):
        super(GAT_GCN, self).__init__()
        self.conv1 = pyg_nn.GATConv(in_c, hid_c, heads=K, dropout=.5, concat=True)
        self.conv2 = pyg_nn.GATConv(hid_c*K, out_c, heads=1, dropout=.5, concat=False)

    def forward(self,data):
        x = data.x
        edge_idx = data.edge_index

        x = F.dropout(x, training=self.training)

        hid = self.conv1(x, edge_idx)
        hid = F.dropout(hid, training=self.training)
        hid = F.elu(hid)

        out = self.conv2(hid, edge_idx)
        out = F.log_softmax(out, dim=1)

        return out

class graphSAGE(nn.Module):
    def __init__(self, in_c, out_c, hid_c=64):
        super(graphSAGE,self).__init__()
        self.conv1 = pyg_nn.SAGEConv(in_c,hid_c)
        self.conv2 = pyg_nn.SAGEConv(hid_c,out_c)
    def forward(self, data):
        x = data.x
        edge_idx = data.edge_index
        x = F.dropout(x, training=self.training)

        hid = self.conv1(x, edge_idx)
        hid = F.dropout(hid, training=self.training)
        hid = F.relu(hid)

        out = self.conv2(hid, edge_idx)
        out = F.log_softmax(out, dim=1)
        return out 

class fastGCN(nn.Module):
    def __init__(self, in_c, out_c, hid_c = 64):
        super(fastGCN, self).__init__()
        self.conv1 = pyg_nn.SGConv(in_c, hid_c)
        self.conv2 = pyg_nn.SGConv(hid_c, out_c)

    def forward(self, data):
        x = data.x
        edge_idx = data.edge_index
        x = F.dropout(x, training=self.training)

        hid = self.conv1(x, edge_idx)
        hid = F.dropout(hid, training=self.training)
        hid = F.relu(hid)

        out = self.conv2(hid, edge_idx)
        out = F.log_softmax(out, dim=1)

        return out

class GraphUnet(nn.Module):
    def __init__(self, in_c, out_c, hid_c=64, depth=3):
        super(GraphUnet, self).__init__()
        self.conv1 = pyg_nn.GraphUNet(in_c, hid_c, out_c, depth)

    def forward(self, data):
        x = data.x
        edge_idx = data.edge_index
        x = F.dropout(x, training=self.training)

        out = self.conv1(x, edge_idx)
        out = F.log_softmax(out, dim=1)

        return out

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cora_dataset = get_data()
    if args.model == "gat":
        # todo list
        my_net = GAT_GCN(cora_dataset.num_features,cora_dataset.num_classes)
    elif args.model == "gcn":
        my_net = GraphCNN(cora_dataset.num_features,cora_dataset.num_classes)
    elif args.model == "graphsage":
        my_net = graphSAGE(cora_dataset.num_features,cora_dataset.num_classes)
    elif args.model == "sgcn":
        my_net = fastGCN(cora_dataset.num_features,cora_dataset.num_classes)
    elif args.model == "GraphUnet":
        my_net = GraphUnet(cora_dataset.num_features,cora_dataset.num_classes)

    args.path = args.path.format(args.model)
    
    # load params for model test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = torch.optim.SGD(my_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-5)
    total_loss, best_acc, epoch_start = .0, .0, 0
    try:
        params = torch.load(args.path)
        # my_net_dict = my_net.state_dict()
        # params = {k:v for k,v in params.items() if k in my_net_dict}
        my_net.load_state_dict(params["model"])
        optimizer.load_state_dict(params["optimizer"])
        epoch_start = params["epoch"]
        best_acc = params["acc"]
        print("Model load success!")
    except:
        print("Model load error!")
        args.train = True

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.max_epoch,eta_min=1e-4,last_epoch=-1)
    my_net = my_net.to(device)
    data = cora_dataset[0].to(device)
    # model train
    if args.train:
        
        for epoch in range(epoch_start,args.max_epoch):
            my_net.train()
            optimizer.zero_grad()
            # scheduler.step()
            output = my_net(data)
            loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # model validation or test
            if epoch % args.interval == (args.interval-1):
                print("Epoch: {} Loss: {:.6f} ".format(epoch+1, total_loss / args.interval))
                total_loss = 0.0 # reset total loss

                my_net.eval()
                with torch.no_grad():
                    _, prediction = my_net(data).max(dim=1)

                    target = data.y
                    val_idx = data.test_mask # or data.val_mask
                        
                    test_correct = prediction[val_idx].eq(target[val_idx]).sum().item()
                    test_number = val_idx.sum().item()
                    acc = test_correct / test_number
                    print("Accuracy of Val Samples: {:.3f}".format(acc) )

                if best_acc < acc:
                    best_acc = acc
                    
                    state = {"model": my_net.state_dict(), "optimizer": optimizer.state_dict(), 
                                "epoch": epoch, "acc": best_acc}
                    torch.save(state,args.path)
                    print("Best accuracy {:.4f} at epoch {}".format(best_acc, epoch+1) )

                # torch.save(my_net,"current.pth")
    # # model test
    else:
        my_net.eval()
        _, prediction = my_net(data).max(dim=1)

        target = data.y

        test_correct = prediction[data.test_mask].eq(target[data.test_mask]).sum().item()
        test_number = data.test_mask.sum().item()
        # 82+, not bad hhh OvO
        # Test accuracy on Cora dataset with GAT, GCN, GraphSAGE, SimpleGCN, GraphUnet
        # Net       Acc
        # GAT       83%
        # GCN       82%
        # GraphSAGE 81.2%
        # SimpleGCN 82.5%
        # GraphUnet 81.4%
        print("Accuracy of Test Samples with {} is: {:.3f}".format(args.model, test_correct / test_number) )


if __name__ == '__main__':
    main()
