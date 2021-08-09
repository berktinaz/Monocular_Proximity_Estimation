import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

dists_0 = np.load('/data/btinaz/mono_depth/final_videos/distances_0_5x5_finetune.npy')
dists_1 = np.load('/data/btinaz/mono_depth/final_videos/distances_1_5x5_finetune.npy')
dists_2 = np.load('/data/btinaz/mono_depth/final_videos/distances_2_5x5_finetune.npy')

dists_0 = signal.medfilt(dists_0, 9)
dists_1 = signal.medfilt(dists_1, 9)
dists_2 = signal.medfilt(dists_2, 9)

plt.hist(dists_0[dists_0!=-1])
plt.xlabel('meters (m)')
plt.ylabel('count')
plt.title('Distribution of the distance in session_0')
plt.savefig('hist_dist_0.png')

plt.figure()
plt.hist(dists_1[dists_1!=-1])
plt.xlabel('meters (m)')
plt.ylabel('count')
plt.title('Distribution of the distance in session_1')
plt.savefig('hist_dist_1.png')

plt.figure()
plt.hist(dists_2[dists_2!=-1])
plt.xlabel('meters (m)')
plt.ylabel('count')
plt.title('Distribution of the distance in session_2')
plt.savefig('hist_dist_2.png')

plt.figure()
plt.plot(dists_0[dists_0!=-1])
plt.xlabel('frame number')
plt.ylabel('meters (m)')
plt.title('Proximity vs time in session_0')
plt.savefig('time_dist_0.png')

plt.figure()
plt.plot(dists_1[dists_1!=-1])
plt.xlabel('frame number')
plt.ylabel('meters (m)')
plt.title('Proximity vs time in session_1')
plt.savefig('time_dist_1.png')

plt.figure()
plt.plot(dists_2[dists_2!=-1])
plt.xlabel('frame number')
plt.ylabel('meters (m)')
plt.title('Proximity vs time in session_2')
plt.savefig('time_dist_2.png')