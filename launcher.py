import subprocess
import numpy as np
import pickle
import matplotlib.pyplot as plt

N_scripts = 100

#for i in range(N_scripts):
#    subprocess.run(["python", "main.py", str(i)])

#likelihood_evals = []
#points = []
#for i, As in enumerate(np.linspace(start = 0.01, stop = 13, num = 50, endpoint = False)):
#    subprocess.run(["python", "main.py", str(i), str(As)])


lik_evals = []
upper_bounds = []
lower_bounds = []
var = []
ratios = []
all_ess = []
points = []
for i, As in enumerate(np.linspace(start = 0.01, stop = 13, num = 50, endpoint = False)):
    with open("B3DCMB/flatness_" +str(i), "rb") as f:
        d = pickle.load(f)
        log_weights = d["log_weights"]
        log_approx = np.max(log_weights) + np.log(np.mean(np.exp(log_weights - np.max(log_weights))))
        log_var = np.log(np.mean(np.exp(log_weights - np.max(log_weights))**2) - (np.mean(np.exp(log_weights - np.max(log_weights)))**2)) + 2*np.max(log_weights)
        lik_evals.append(log_approx)
        upper_bound = log_approx + np.log(1 + (1.96/np.sqrt(len(log_weights)))*np.exp((1/2)*log_var - log_approx))
        lower_bound = log_approx + np.log(1 - (1.96/np.sqrt(len(log_weights)))*np.exp((1/2)*log_var - log_approx))
        upper_bounds.append(upper_bound)
        lower_bounds.append(lower_bound)
        print("bounds")
        print(upper_bound)
        print(log_approx)
        print(np.exp(log_approx - (1/2)*log_var))
        print(lower_bound)
        ess = (np.sum(np.exp(log_weights - np.max(log_weights)))**2)/np.sum(np.exp(log_weights - np.max(log_weights))**2)
        all_ess.append(ess)
        print("ess")
        print(ess)
        print("As")
        print(As)
        ratios.append(upper_bound*lower_bound/(log_approx**2))
        var.append(log_var)
        points.append(As)
        print("\n")


cut = 10
fig, axes = plt.subplots(2, 2, figsize = (10, 10))
fig.subplots_adjust(wspace = 0.7, hspace = 0.7)
fig.suptitle("log likelihood in As with noise for 'true value' of As = 3.034")
axes[0,0].plot(points, lik_evals, "blue")
axes[0, 0].set_title("log likelihood")
axes[0, 0].set_xlabel("As")
axes[0, 1].plot(points, all_ess)
axes[0, 1].set_title("Ess of approximation for each point")
axes[0, 1].set_xlabel("As")
axes[1, 0].plot(points[0:cut], lik_evals[0:cut], "blue")
axes[1, 0].plot(points[0:cut], upper_bounds[0:cut], "red")
axes[1, 0].plot(points[0:cut], lower_bounds[0:cut], "red")
axes[1, 0].set_title("95% confidence intervals for each point", pad = 20)
axes[1, 0].set_xlabel("As")
fig.savefig("log_lik_As_with_noise_3.png")
plt.close()
points = [As for As in np.linspace(start = 0.5, stop = 25, num = 50, endpoint = False)]
print(lik_evals)
plt.plot(points[:12], lik_evals[0:12], "blue")
plt.plot(points[0:12], upper_bounds[0:12], "red")
plt.plot(points[0:12], lower_bounds[0:12], "red")
#plt.plot(points, lower_bounds, "red")
plt.savefig("log_likelihood_As_with_noise_3.png")
plt.close()
#plt.plot(points, all_ess)
#plt.plot(points, lower_bounds, "red")
#plt.savefig("log_likelihood_ess_no_noise.png")
#print(vals)

with open("B3DCMB/data/reference_data_As_NSIDE_512_", "rb") as f:
    reference_data = pickle.load(f)

print(reference_data["cosmo_params"])
#print(upper_bounds[29] - lik_evals[29])
#print(lik_evals[29] - lower_bounds[29])
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