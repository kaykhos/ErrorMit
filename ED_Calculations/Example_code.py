from Entropy_Patch import patch_entropies
from matplotlib import pyplot as plt

# Super simple code to calculate the purity dynamics of the transverese field Ising model.

N = 8 #Chain length
J = -1 #Ising coupling strength
hx = -0.5 #transverese field strength
t_max = 5 #max time
Patches = [[0],[0,1,2,3],[4,5,6],[0,1,2,3,4,5,6,7]] #set ofpatches

times, results = patch_entropies(N,J,hx,t_max,Patches)   # returns the times and the purities for the patches.

plt.plot(times, results, label = 'free')
plt.show()
