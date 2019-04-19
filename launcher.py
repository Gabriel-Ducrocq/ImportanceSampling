import subprocess
import numpy as np
import pickle
from main import main

N_scripts = 100

#for i in range(N_scripts):
#    subprocess.run(["python", "main.py", str(i)])

likelihood_evals = []
points = []
for i in np.linspace(start = 0.5, stop = 25, num = 50, endpoint = False):
    lik_eval = main(512, i, i)
    points.append(i)
    likelihood_evals.append(lik_eval)



d = {"y": likelihood_evals, "x":points}
with open("B3DCMB/flatness", "wb") as f:
    pickle.dump(d, f)


"""
## Computing the weight for each beta:
ESS_list = []
beta_weights = []
all_weights = []
for i in range(86):
    with open("B3DCMB/data/simulated_beta_NSIDE_512_" + str(i), "rb") as f:
        res_current = pickle.load(f)

    log_weights = res_current["log_weights"]
    w = np.exp(log_weights - np.max(log_weights))
    weights = (w/np.sum(w))
    beta_weights.append(log_weights)
    ess = (np.sum(weights)**2)/np.sum(weights**2)
    print(ess)
    #pickle.dump({"simulated_points": all_sample, "sampled_beta": sampled_beta, "log_weights": log_weights}, f)

log_w = np.array([np.max(arr) + np.log(np.mean(np.exp(arr - np.max(arr)))) for arr in beta_weights])
w = np.exp(log_w - np.max(log_w))
auto_norm_w = w/np.sum(w)
ess = (np.sum(auto_norm_w)**2)/np.sum(auto_norm_w**2)
print("Ess for beta")
print(ess)
print("Ess/N")
print(ess/44)
print(auto_norm_w)
print(log_w)
print(w)
print(log_w - np.max(log_w))
"""