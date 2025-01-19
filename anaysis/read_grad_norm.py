import numpy as np


loaded_data = np.load("/home/sorlova/repos/NewStart/VideoMAE/logs/check_things/grad_norm/grad_norms/gradnorm_ep0.npz")


# Convert to dictionary if needed
data_dict = {key: loaded_data[key] for key in loaded_data}

print("")