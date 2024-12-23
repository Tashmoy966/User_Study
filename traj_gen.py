import numpy as np
import json
import matplotlib.pyplot as plt

def generate_linear_trajectory(num_points, x_min, x_max, z_min, z_max):
    """Generate a linear trajectory."""
    x = np.linspace(x_min, x_max, num_points)
    y = np.zeros(num_points)
    z = np.linspace(z_min, z_max, num_points)
    return x, y, z


def generate_sinusoidal_trajectory(num_points, x_min, x_max, y_min, y_max, z_min, z_max):
    """Generate a sinusoidal trajectory."""
    t = np.linspace(0, 2 * np.pi, num_points)
    x = np.linspace(x_min, x_max, num_points)
    y = (y_max - y_min) / 2 * np.sin(2 * t) + (y_min + y_max) / 2
    z = np.linspace(z_max, z_min, num_points)
    return x, y, z


def generate_circular_trajectory(num_points, radius, z_min, z_max):
    """Generate a circular trajectory in the XY plane."""
    t = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(z_min, z_max, num_points)
    return x, y, z


def generate_spiral_trajectory(num_points, radius_start, radius_end, z_min, z_max):
    """Generate a 3D spiral trajectory."""
    t = np.linspace(0, 4 * np.pi, num_points)  # Spiral spans 2 full turns
    radius = np.linspace(radius_start, radius_end, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = np.linspace(z_min, z_max, num_points)
    return x, y, z


def calculate_velocity(x, y, z):
    """Calculate velocity based on the derivatives of position."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    velocity = np.sqrt(dx**2 + dy**2 + dz**2)
    return velocity


def save_to_json(x, y, z, velocity, output_file):
    """Save trajectory data to a JSON file."""
    trajectory_data = {
        "trajectory": [
            {"x": float(xi), "y": float(yi), "z": float(zi), "velocity": float(vi)}
            for xi, yi, zi, vi in zip(x, y, z, velocity)
        ]
    }

    with open(output_file, "w") as f:
        json.dump([trajectory_data], f, indent=4)

    print(f"Trajectory data saved to {output_file}")

def plot_traj(x,y,z,x_min,x_max,y_min,y_max,z_min,z_max):
    # Plot the trajectory
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label="Sinusoidal Trajectory")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.legend()
    plt.show()
# Main script to generate and save a trajectory
if __name__ == "__main__":
    num_points = 1000
    output_file = "/home/tashmoy/iisc/HIRO/GPT/sample/trajectory_2.json"

    # Example: Generate a sinusoidal trajectory
    # x, y, z = generate_sinusoidal_trajectory(num_points, x_min=0.4, x_max=0.8, y_min=-0.4, y_max=0.4, z_min=0.5, z_max=0.8)
    # x, y, z = generate_linear_trajectory(num_points, x_min=0.4, x_max=0.8, z_min=0.5, z_max=0.8)
    # x, y, z = generate_circular_trajectory(num_points, radius=0.5, z_min=0.5, z_max=0.8)
    # x, y, z = generate_spiral_trajectory(num_points, radius_start=0.1, radius_end=0.5, z_min=0.5, z_max=0.8)

    # Calculate velocity
    velocity = calculate_velocity(x, y, z)

    # Save trajectory to JSON
    save_to_json(x, y, z, velocity, output_file)
    plot_traj(x, y, z, x_min=0.4, x_max=0.8, y_min=-0.4, y_max=0.4, z_min=0.5, z_max=0.8)

    
