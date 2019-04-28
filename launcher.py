import subprocess
import numpy as np
import pickle
import matplotlib.pyplot as plt

N_scripts = 100

#for i in range(N_scripts):
#    subprocess.run(["python", "main.py", str(i)])

#likelihood_evals = []
#points = []
#for i, As in enumerate(np.linspace(start = 0.5, stop = 25, num = 50, endpoint = False)):
#    subprocess.run(["python", "main.py", str(i), str(As)])

lik_evals = []
var = []
points = []
for i, As in enumerate(np.linspace(start = 0.5, stop = 25, num = 50, endpoint = False)):
    with open("B3DCMB/flatness_bis" +str(i), "rb") as f:
        d = pickle.load(f)
        log_weights = d["log_weights"]
        log_approx = np.max(log_weights) + np.log(np.mean(np.exp(log_weights - np.max(log_weights))))
        #log_var = 2*np.max(log_weights) + np.log(np.mean((np.exp(log_weights - np.max(log_weights)) - np.exp(log_approx - np.max(log_weights)))**2))
        log_var = np.log(np.mean(np.exp(log_weights - np.max(log_weights))**2) - (np.mean(np.exp(log_weights - np.max(log_weights)))**2)) + 2*np.max(log_weights)
        lik_evals.append(log_approx)
        upper_bound = log_var*np.log(1+(1.96/np.sqrt(len(log_weights)))*np.exp(log_approx - log_var))
        lower_bound = log_approx * np.log(1 - (1.96 / np.sqrt(len(log_weights))) * np.exp(log_var - log_approx))
        print("bounds")
        print(upper_bound)
        print(log_approx)
        print(log_var)
        print(lower_bound)
        ess = (np.sum(np.exp(log_weights - np.max(log_weights)))**2)/np.sum(np.exp(log_weights - np.max(log_weights))**2)
        print("ess")
        print(ess)
        print("As")
        print(As)
        if ess < 10:
            print(log_weights)
        print("\n")


points = [As for As in np.linspace(start = 0.5, stop = 25, num = 50, endpoint = False)]
print(lik_evals)
plt.plot(points, np.exp(lik_evals - np.max(lik_evals))/np.sum(np.exp(lik_evals - np.max(lik_evals))))
plt.savefig("log_likelihood_As_bis.png")
#print(vals)

with open("B3DCMB/data/reference_data_As_NSIDE_512_bis", "rb") as f:
    reference_data = pickle.load(f)

print(reference_data["cosmo_params"])
#d = {"y": likelihood_evals, "x":points}
#with open("B3DCMB/flatness", "wb") as f:
#    pickle.dump(d, f)
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