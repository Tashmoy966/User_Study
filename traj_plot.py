import json
# Importing matplotlib for plotting
import matplotlib.pyplot as plt
import numpy as np

# 3D Trajectory Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

trajectory_path='/home/tashmoy/iisc/HIRO/GPT/Manipulator-20241207T074856Z-001/traj_list/trajectory_6.json'

with open(trajectory_path, 'r') as file:
        data = json.load(file)
processed_data=np.array(data["trajectory"])
# Plotting the trajectory
ax.plot(processed_data[:,0], processed_data[:,1], processed_data[:,2], label="Trajectory Path", color='b')
ax.scatter(processed_data[:,0], processed_data[:,1], processed_data[:,2], c='r', marker='o', label="Key Points")

# Adding labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Trajectory Plot')
ax.legend()

# Show the plot
plt.show()