import subprocess
import numpy as np
import pickle

N_scripts = 50

#for i in range(N_scripts):
#    subprocess.run(["python", "main.py", str(i)])



## Computing the weight for each beta:
ESS_list = []
beta_weights = []
all_weights = []
for i in range(44):
    with open("B3DCMB/data/simulated_beta_NSIDE_512_" + str(i), "rb") as f:
        res_current = pickle.load(f)

    log_weights = res_current["log_weights"]
    w = np.exp(log_weights - np.max(log_weights))
    weights = (w/np.sum(w))
    beta_weights.append(np.sum(weights))
    all_weights += list(weights)
    ess = (np.sum(weights)**2)/np.sum(weights**2)
    ESS_list.append(ess)
    print(ess)
    #pickle.dump({"simulated_points": all_sample, "sampled_beta": sampled_beta, "log_weights": log_weights}, f)

print(beta_weights)
beta_ess = (np.sum(beta_weights)**2)/np.sum(np.array(beta_weights)**2)
print("beta_ess:")
print(len(beta_weights))
print(beta_ess)
print("All ess")
all_ess = (np.sum(all_weights)**2)/np.sum(np.array(all_weights)**2)
print(len(all_weights))
print(all_ess)