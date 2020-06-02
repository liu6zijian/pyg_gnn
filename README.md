# pyg_gnn
 Some gnn methods (GAT, GCN, GraphSAGE, simpleGCN, GraphUnet) with pytorch geometric framework.
## files
 These {}.pth files are saved model parameters.
 `node_classify.py` is the main function for node classification on <cora>.
 
 The train code is: ```python node_classify.py --train True --model gat --gpu 0 --path best_{}.pth``` 
 
 The test code is: ```python node_classify.py --train False --model gat --gpu 0 --path best_{}.pth``` 
 
 * _--train_ is the train and test mode flag - _True_ for train and _False_ for test
 * _--model_ can select different gnn methods - here we provide five common approaches
 * _--gpu_ can determine the number of gpu device
 * _--path_ gives the model save path
 
 Net       GAT       GCN       GraphSAGE    SimpleGCN      GraphUnet 
 
 Acc       83%       82%       81.2%        82.5%          81.4%
 
       
 
 
 
 
 

 
 
