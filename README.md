## Impact-of-heterophilic-designs

The relevant codes for "**What Contributes More to the Robustness of Heterophilic Graph Neural Networks?**", [*Under Review*].




- ### Requirements

  - Our codes are built based on the project [GraphWar (Now GreatX)](https://github.com/EdisonLeeeee/GreatX/tree/graphwar) , so you may need to install PyTorch, PyTorch Geometric, and GraphWar first. Please see the 'requirements.txt' for details. 

    

- ### Code illustrations

  - **examples/attack/targeted**: Main demo folder

    - for_real_gnn_evaluate.py: main codes for conducting attacks on realistic GNNs, such as H2GCN and UGCN.
  - sim_heterophilic_attack.py: main codes for conducting attacks on baseline GCN models integrating with different heterophilic designs. 
  
- **graphwar/heter_gnn**: Main algorithm folder
  
  - basic_gcn.py
    - h2gcn.py
  - ugcn.py
  
- **heterophily_dataseets_matlab**: Folder of the corresponding heterophilic graph data.
  
  
  
- ### Run the demo

  ```
  # baseline GCN model integrating with different heterophilic designs
  python sim_heterophilic_attack.py
  
  # realistic GNN model
  python for_real_gnn_evaluate.py
  ```

  

