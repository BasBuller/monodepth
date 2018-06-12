import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_pickle('../results/results.pickle')

index_pruning = ['second_test','third_test']
index_squeeze = []
index_remove  = []

remove_params = [31.6e6,19.8e6,8.0e6 ]
#remove_abs =


plt.plot(,label='Pruning')
plt.plot([1,2,3],[4,5,6],label='Squeezenet')
plt.plot([1,2,3],[4,5,6],label='Remove layers')
plt.xlabel('Number of parameters [-]')
plt.ylabel('Absolute  relative error [-]')
plt.legend()
plt.show()
