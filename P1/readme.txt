- testes.py
Treina e testa Regressão Linear, Ridge e Lasso.
Utiliza cross-validation (através de kfold).
São testados vários alpha para os modelos Ridge e Lasso de modo a determinar qual minimiza o SSE, calculado no fim.

- final.py
Treina o modelo e produz o ficheiro Ytest.npy, utilizando ridge com alpha=1.78 obtido através do ficheiro testes.py, sendo este o valor que produziu o SSE mais baixo.
