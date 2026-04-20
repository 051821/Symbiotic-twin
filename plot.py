import matplotlib.pyplot as plt
import numpy as np

# ---- YOUR FINAL VALUES ----
accuracy = 95        # %
latency = 65000      # ms
energy = 210         # J

# ---- NORMALIZATION (FIXED) ----
# Accuracy → higher is better
acc_norm = accuracy / 100  

# Latency & Energy → lower is better (invert)
lat_norm = 1 - (latency / 100000)   # adjust max if needed
eng_norm = 1 - (energy / 500)       # adjust max if needed

values = [acc_norm, lat_norm, eng_norm]
labels = ["Accuracy", "Latency", "Energy"]

plt.figure()

bars = plt.bar(labels, values)

# Add values on top
for i, bar in enumerate(bars):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height(),
             f"{values[i]:.2f}",
             ha='center', va='bottom')

plt.ylabel("Value (Normalized)")
plt.title("Symbiotic-Twin")

plt.ylim(0, 1)

plt.show()