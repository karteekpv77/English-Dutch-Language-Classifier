English-Dutch Language Classifier:
Classify the lines in a file to English or Dutch using the Decision tree algorithm and 
also boosting the accuracy of the results using Adaboost algorithm


Usage:
The command line argument for Decision Tree:
Training: $python3 Decision.py train train.txt hypothesis_dt.out dt
Predict: $python3 Decision.py predict hypothesis_dt.out test.txt

The command line argument for Adaboost:
Training: $python3 Decision.py train train.txt hypothesis_ada.out ada
Predict: $python3 Decision.py predict hypothesis_ada.out test.txt
