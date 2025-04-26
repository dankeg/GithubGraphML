# GithubGraphML
Data Set: [gitSED](https://zenodo.org/records/5021329)

# Abstract
We perform an analysis on GitSED, or the GitHub Socially Enhanced Dataset. In particular, we study the network topology of collaboration networks of repositories predominantly written in specific programming languages via classical network analysis techniques. Additionally, we evaluate the dataset in the context of recommendation systems, exploring its potential for tasks such as developer and repository recommendation. Our results suggest that while modeling Github collaboration as a social network enables insights to be derived using traditional community detection and Graph Neural Network techniques, re-expressions of these relationships may provide better performance depending on the task. 


# How to Use:
Download the gitSED dataset, unzip, and place the contents of the directory within the root directory of this repo.

From there, you can run the implementations of the analysis within the python files, such as the GNN code. Loading is designed such that on the first load, it performs the parsing of the data from csv into graph-tools, and stores it as a pickle. Further loading utilizes this pickle file, significantly speeding up data loading. 

## Modules:

### GithubGraphML
Top level module, containing primary functions for data parsing, analysis, eval, and visualization

### Parsing
Contains code for loading and parsing graphs, performing initial loading and processing from the gitSED format and providing the capability to reload natively into Graph-Tools. Is leveraged by the other analysis as a utility library. 

### Analysis
Contains code for performing analysis on graphs, such as generating community metrics. Start-to-end analysis leveraging this code, including visualization capabilities discussed below, are implemented within the jupyter notebooks, such as `test_community.ipynb`.

### Visualization 
Generates visuals of graphs, with property maps, and is used in the implementation of analyses such as community detection. 

### Neural Net
Contains the code for defining Graph Neural Networks, performing associated data processing and training, and evaluating their performance. Can be ran by running any of the individual GNN python implementations, such as `GithubGraphML/neural_net/node_classification.py`, which will perform start to end loading, training, and eval. 