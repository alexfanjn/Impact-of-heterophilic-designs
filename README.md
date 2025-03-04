## Impact-of-heterophilic-designs

The relevant codes for "**What Contributes More to the Robustness of Heterophilic Graph Neural Networks?**", [*IEEE TSMC*].




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

  

- ### Cite

  If you find this work helpful, please cite our paper, Thank you.

  ```
   @article{fang2025what,
      title={What Contributes More to the Robustness of Heterophilic Graph Neural Networks?},
      author={Fang, Junyuan and Ynag, Han and Wu, Jiajing and Zheng, Zibin and Tse, Chi K},
      journal={IEEE Transactions on Systems, Man, and Cybernetics: Systems},
      year={2025},
      publisher={IEEE}
    }
  ```
