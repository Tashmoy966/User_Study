from flask import Flask, render_template, request, redirect, url_for
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import io
import base64

app = Flask(__name__)

# Sample 3D trajectory data (replace this with your dataset)
original_trajectory = np.cumsum(np.random.randn(100, 3), axis=0)
method1_trajectory = original_trajectory + np.random.normal(0, 0.5, original_trajectory.shape)
method2_trajectory = original_trajectory + np.random.normal(0, 1.0, original_trajectory.shape)

# Initialize DataFrame to store responses
response_data = pd.DataFrame(columns=["Method", "Rating", "Comparison"])

# Function to create a plot and return its base64 encoding
def create_3d_plot():
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(original_trajectory[:, 0], original_trajectory[:, 1], original_trajectory[:, 2], label="Original Trajectory", color='blue', lw=2)
    ax.plot(method1_trajectory[:, 0], method1_trajectory[:, 1], method1_trajectory[:, 2], label="Method 1", color='green', lw=2)
    ax.plot(method2_trajectory[:, 0], method2_trajectory[:, 1], method2_trajectory[:, 2], label="Method 2", color='red', lw=2)

    ax.set_title("3D Trajectory Comparison")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.legend()

    # Convert the plot to a PNG image and encode it to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return image_base64

@app.route('/')
def index():
    # Display the plot and the survey form
    plot_image = create_3d_plot()
    return render_template('index.html', plot_image=plot_image)

@app.route('/submit', methods=['POST'])
def submit():
    method1_rating = request.form.get('method1_rating')
    method2_rating = request.form.get('method2_rating')
    comparison = request.form.get('comparison')

    if method1_rating and method2_rating and comparison:
        global response_data
        response_data.loc[len(response_data)] = ["Method 1", method1_rating, ""]
        response_data.loc[len(response_data)] = ["Method 2", method2_rating, ""]
        response_data.loc[len(response_data)] = ["Comparison", "", comparison]
        return redirect(url_for('thanks'))
    else:
        return "Please provide all ratings and comparison."

@app.route('/thanks')
def thanks():
    return "Thank you for your submission!"

if __name__ == '__main__':
    app.run(debug=True)
