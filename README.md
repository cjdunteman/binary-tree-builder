Implement a program that builds a binary decision tree for numerical attributes,
and binary classification tasks. Each node will have a selected attribute and an associated threshold value. Instances (aka examples) that have an attribute value less than or equal to the threshold belong to the left subtree of a node, and instances with an attribute value greater than the threshold belong to the right subtree of a node. 

To run the program: 
```
java HW3 <train file> <test file> <maximum instances per leaf> <maximum depth>
```
Maximum instances and depth need to be positive integers.

The dataset can be found here: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29.