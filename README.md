# pyg_gnn
 Some gnn methods (GAT, GCN, GraphSAGE, simpleGCN, GraphUnet) with pytorch geometric framework.
## files
 These {}.pth files are saved model parameters with highest test accuracy.
 `node_classify.py` is the main function for node classification on <cora>.
 `test_env.py` is the environment test function for pyG framework.
## dependency
 pytorch 1.4.0 (here we don't have cuda 10 or 9.2, we choose cpu version)
 
 torchvision 0.5.0
 
 https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
 
 `${CUDA} = cu92, cu100 or cpu`
 
 ```pip install torch-scatter==latest+${CUDA}``` -f https://pytorch-geometric.com/whl/torch-1.4.0.html
 
 ```pip install torch-sparse==latest+${CUDA}``` -f https://pytorch-geometric.com/whl/torch-1.4.0.html
 
 ```pip install torch-cluster==latest+${CUDA}``` -f https://pytorch-geometric.com/whl/torch-1.4.0.html
 
 ```pip install torch-spline-conv==latest+${CUDA}``` -f https://pytorch-geometric.com/whl/torch-1.4.0.html
 
 ```python setup.py install``` or ```pip install torch-geometric```

 The train code is: ```python node_classify.py --train True --model gat --gpu 0 --path best_{}.pth``` 
 
 The test code is: ```python node_classify.py --train False --model gat --gpu 0 --path best_{}.pth``` 
 
 * _--train_ is the train and test mode flag - _True_ for train and _False_ for test
 * _--model_ can select different gnn methods - here we provide five common approaches
 * _--gpu_ can determine the number of gpu device
 * _--path_ gives the model save path
 
 
       
 
