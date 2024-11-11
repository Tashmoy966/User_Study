# from flask import Flask, render_template, request, redirect, url_for
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import pandas as pd
# import numpy as np
# import io
# import base64

# app = Flask(__name__)

# # Sample 3D trajectory data (replace this with your dataset)
# original_trajectory = np.cumsum(np.random.randn(100, 3), axis=0)
# method1_trajectory = original_trajectory + np.random.normal(0, 0.5, original_trajectory.shape)
# method2_trajectory = original_trajectory + np.random.normal(0, 1.0, original_trajectory.shape)

# # Initialize DataFrame to store responses
# response_data = pd.DataFrame(columns=["Method", "Rating", "Comparison"])

# # Function to create a plot and return its base64 encoding
# def create_3d_plot():
#     fig = plt.figure(figsize=(6, 5))
#     ax = fig.add_subplot(111, projection='3d')

#     ax.plot(original_trajectory[:, 0], original_trajectory[:, 1], original_trajectory[:, 2], label="Original Trajectory", color='blue', lw=2)
#     ax.plot(method1_trajectory[:, 0], method1_trajectory[:, 1], method1_trajectory[:, 2], label="Method 1", color='green', lw=2)
#     ax.plot(method2_trajectory[:, 0], method2_trajectory[:, 1], method2_trajectory[:, 2], label="Method 2", color='red', lw=2)

#     ax.set_title("3D Trajectory Comparison")
#     ax.set_xlabel("X-axis")
#     ax.set_ylabel("Y-axis")
#     ax.set_zlabel("Z-axis")
#     ax.legend()

#     # Convert the plot to a PNG image and encode it to base64
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     image_base64 = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close(fig)
#     return image_base64

# @app.route('/')
# def index():
#     # Display the plot and the survey form
#     plot_image = create_3d_plot()
#     return render_template('index.html', plot_image=plot_image)

# @app.route('/submit', methods=['POST'])
# def submit():
#     method1_rating = request.form.get('method1_rating')
#     method2_rating = request.form.get('method2_rating')
#     comparison = request.form.get('comparison')

#     if method1_rating and method2_rating and comparison:
#         global response_data
#         response_data.loc[len(response_data)] = ["Method 1", method1_rating, ""]
#         response_data.loc[len(response_data)] = ["Method 2", method2_rating, ""]
#         response_data.loc[len(response_data)] = ["Comparison", "", comparison]
#         return redirect(url_for('thanks'))
#     else:
#         return "Please provide all ratings and comparison."

# @app.route('/thanks')
# def thanks():
#     return "Thank you for your submission!"

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for, session
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import io
import base64
import plotly.graph_objs as go
import plotly
import json
from scipy.interpolate import CubicSpline
import secrets
import sqlite3

# Function to initialize the database
def init_db():
    conn = sqlite3.connect('user_responses.db')
    c = conn.cursor()
    
    # Create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trajectory_index INTEGER,
            method1_rating INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

# Call this function once to create the table
init_db()

# Generate a secure random session key
session_key = secrets.token_hex(16)  # 32 characters in hexadecimal (128 bits)
def smoothned_data(val,num_points=80):
   t = np.arange(len(val))
   spline_val = CubicSpline(t, val)
   t_new = np.linspace(0, len(val) - 1, num=num_points)
   val_smooth = spline_val(t_new)
   return val_smooth
def get_json_data(trajectory_path):
    with open(trajectory_path, 'r') as file:
        data = json.load(file)
        return data

def get_trajectory(data):
        # Extract the 'trajectory' key and process it into a list of tuples
    trajectory = data.get('trajectory', [])
    processed_trajectory = [[point['x'], point['y'], point['z'], point['velocity']] for point in trajectory]
        
    return processed_trajectory
app = Flask(__name__)
app.secret_key = session_key # Required for session management

original_trajectory_path='/home/tashmoy/IISC/HIRO/gpt/Language-models-for-trajectory-formatting-master/new_traj/temp_traj.json'
modefied_trajectory_path="/home/tashmoy/IISC/HIRO/gpt/Language-models-for-trajectory-formatting-master/modefied_trajectory/shift_wo_obj"
original_data=get_json_data(original_trajectory_path)

method1_trajectories=np.array([get_trajectory(get_json_data(modefied_trajectory_path+f"/modefied_data_{i+1}.json")[0]) for i in range(5)])
# Sample 3D trajectory data (replace this with your dataset)
num_trajectories = 5  # Number of trajectories
original_trajectories = np.array(get_trajectory(original_data[0])) #[np.cumsum(np.random.randn(100, 3), axis=0) for _ in range(num_trajectories)]
# method1_trajectories = [traj + np.random.normal(0, 0.5, traj.shape) for traj in original_trajectories]
#vel = np.cos(np.linspace(0, 10,len(original_trajectories))) + np.random.normal(0, 0.1, len(original_trajectories))
# method2_trajectories = [traj + np.random.normal(0, 1.0, traj.shape) for traj in original_trajectories]

# Define some object points and their corresponding colors
object_points = np.array([[point['x'], point['y'], point['z']] for point in original_data[0]["objects"]])#np.array([[1, 2, 1], [3, 1, 0], [0, 0, 3], [2, 3, 2], [1, 1, 1]])
object_colors = ['red',
'green',
'blue',
'yellow',
'cyan',
'magenta',
'black',
'white',
'gray',
'orange',
'purple',
'brown',
'pink',
'violet',
'gold',
'beige']  # List of colors for each object point

inst=["Go higher",
"Go lower",
"Go to the left",
"Go to the right",
"Go faster when you are in the middle of the trajectory."]
hlp=["Shift the goal position left.",
    "Keep the start position same",
    "modify the points in the middle to ensure a gradual change in the trajectory preserving the shape of the trajectory"]
llm_used=["gpt-4o","llama-3.1","gemeni","calude","gpt-4o"]

# Initialize DataFrame to store responses
response_data = pd.DataFrame(columns=["Method", "Rating", "Comparison"])
# print(original_trajectories)
# Function to create a plot and return its base64 encoding
def create_3d_plot(index):
    # fig = plt.figure(figsize=(6, 5))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.plot(original_trajectories[index][:, 0], original_trajectories[index][:, 1], original_trajectories[index][:, 2], label="Original Trajectory", color='blue', lw=2)
    # ax.plot(method1_trajectories[index][:, 0], method1_trajectories[index][:, 1], method1_trajectories[index][:, 2], label="Method 1", color='green', lw=2)
    # # ax.plot(method2_trajectories[index][:, 0], method2_trajectories[index][:, 1], method2_trajectories[index][:, 2], label="Method 2", color='red', lw=2)

    # ax.set_title(f"3D Trajectory Comparison {index + 1}/{num_trajectories}")
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.set_zlabel("Z-axis")
    # ax.legend()

    # # Convert the plot to a PNG image and encode it to base64
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    # plt.close(fig)
    # return image_base64
    # Create traces for the original trajectory and method 1 trajectory
    # Create a color scale based on the z-values

    fig = go.Figure()
    colorscale = 'Viridis'  # Choose a color scale
  # Assign the vel values as colors
    #trace_original = 
    fig.add_trace(go.Scatter3d(
        x=smoothned_data(original_trajectories[:, 0]), 
        y=smoothned_data(original_trajectories[:, 1]), 
        z=smoothned_data(original_trajectories[:, 2]),
        
        mode='lines+markers',
        name='Original Trajectory',
        marker=dict(
        size=5,
        color=smoothned_data(original_trajectories[:, 3]),  # Assign colors based on z values
        colorscale='BrBG',  # Set the color scale
        colorbar=dict(title='Vel Value'),
        opacity=0.8),  # Color bar title
        # line=dict(color='blue', width=4)),
        # line=dict(color='blue', width=4)
    ))

    #trace_method1 = 
    fig.add_trace(go.Scatter3d(
        x=method1_trajectories[index][:, 0], 
        y=method1_trajectories[index][:, 1], 
        z=method1_trajectories[index][:, 2],
        mode='lines+markers',
        name='Method 1',
        marker=dict(
        size=5,
        color=method1_trajectories[index][:, 3],  # Assign colors based on z values
        colorscale='RdBu',  # Set the color scale
        colorbar=dict(title='Z Value'),
        opacity=0.8),  # Color bar title
        # line=dict(color='red', width=4)),
        # line=dict(color='green', width=4)
    ))
    # Add object points to the 3D plot with different colors
    #object_data = 
    fig.add_trace(go.Scatter3d(
        x=object_points[:, 0],
        y=object_points[:, 1],
        z=object_points[:, 2],
        mode='markers',
        text=['{}'.format(point["name"]) for point in original_data[0]["objects"]],
        textposition='top center',  # Position of the text relative to the markers
        hoverinfo='text',
        name='Object Points',
        marker=dict(size=8, color=object_colors, symbol='circle')
    ))

     #Add annotations for the object names
    annotations = []
    for point in original_data[0]["objects"]:
        annotations.append(dict(
            x=object_points[:, 0],
            y=object_points[:, 1],
            z=object_points[:, 2],
            text=point["name"],
            showarrow=False,
            font=dict(size=1, color='black'),
            xanchor='auto'
        ))

    # Define the layout of the plot
    layout = go.Layout(
        title=f"3D Trajectory Comparison {index + 1}/{num_trajectories}",
        scene=dict(
            # annotations=annotations,
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
            zaxis=dict(title='Z-axis')
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    # Create the figure
    #fig = go.Figure(data=[trace_original, trace_method1,object_data], layout=layout)
    fig.update_layout(layout)
    # Convert the figure to JSON to pass to the frontend
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

# Function to create a 2D plot for plotly
def create_2d_plot(index):
    fig = go.Figure()
    
    # Plotting the original trajectory
    fig.add_trace(go.Scatter(
        x=[i+1 for i in range(len(smoothned_data(original_trajectories[:, 3])))],
        y=smoothned_data(original_trajectories[:, 3]-1),
        mode='lines+markers',
        name='Original Trajectory',
        line=dict(color='blue'),
        marker=dict(size=4)
    ))
    
    # Plotting the method1 trajectory
    fig.add_trace(go.Scatter(
        x=[i+1 for i in range(len(method1_trajectories[index][:, 3]))],
        y=method1_trajectories[index][:, 3],
        mode='lines+markers',
        name='Method 1',
        line=dict(color='green'),
        marker=dict(size=4)
    ))
    
    layout = go.Layout(
        title=f"Velocity Comparison {index + 1}/{num_trajectories}",
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        showlegend=True
    )

    fig.update_layout(layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
@app.route('/', methods=['GET', 'POST'])
def index(error_message=None):
    session['current_index'] = 0  # Initialize the trajectory index
    plot_image = create_3d_plot(session['current_index'])
    return redirect(url_for('display_trajectory'))
    #return render_template('index.html', plot_image=plot_image, current_index=session['current_index'], num_trajectories=num_trajectories, error_message=error_message)

@app.route('/next', methods=['POST',"GET"])
def next_trajectory():
    current_index = session.get('current_index', 0)
    if current_index < num_trajectories - 1:
        current_index += 1
        session['current_index'] = current_index
    return redirect(url_for('display_trajectory'))

@app.route('/previous', methods=['POST'])
def previous_trajectory():
    current_index = session.get('current_index', 0)
    if current_index > 0:
        current_index -= 1
        session['current_index'] = current_index
    return redirect(url_for('display_trajectory'))

@app.route('/display')
def display_trajectory():
    current_index = session['current_index']
    # plot_image = create_3d_plot(current_index)
    # return render_template('index.html', plot_image=plot_image, current_index=current_index, num_trajectories=num_trajectories)
    instruction_text=inst[current_index]
    llm_names=llm_used[current_index]
    hlp_text=hlp
    plot_json = create_3d_plot(current_index)
    vel_json=create_2d_plot(current_index)
    return render_template('index.html', llm_names=llm_names,plot_json=plot_json, current_index=current_index, num_trajectories=num_trajectories,instruction_text=instruction_text,hlp_text=hlp_text,vel_json=vel_json)


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    current_index = session['current_index']
    method1_rating = request.form.get('method1_rating')
    # method2_rating = request.form.get('method2_rating')
    # comparison = request.form.get('comparison')

    if method1_rating :  #and method2_rating and comparison
        # Connect to the database
        conn = sqlite3.connect('user_responses.db')
        c = conn.cursor()

        # Insert the data into the responses table
        c.execute('''
            INSERT INTO responses (trajectory_index, method1_rating)
            VALUES (?, ?)
        ''', (current_index, method1_rating))

        # Commit and close the connection
        conn.commit()
        conn.close()
        global response_data
        response_data.loc[len(response_data)] = ["Method 1", method1_rating, ""]
        # response_data.loc[len(response_data)] = ["Method 2", method2_rating, ""]
        # response_data.loc[len(response_data)] = ["Comparison", "", comparison]
        if current_index < num_trajectories - 1:
            return redirect(url_for('next_trajectory'))  # Go to the next trajectory after submission
        else:
            return redirect(url_for('thanks'))  # Thank you page after the last trajectory
    else:
        # Render the index template with an error message
        error_message = "Please provide all ratings and comparison."
        instruction_text=inst[current_index]
        hlp_text=hlp
        llm_names=llm_used[current_index]
        vel_json=create_2d_plot(current_index)
        plot_json = create_3d_plot(current_index)
        return render_template('index.html', llm_names=llm_names, plot_json=plot_json, current_index=current_index, num_trajectories=num_trajectories,instruction_text=instruction_text,hlp_text=hlp_text,vel_json=vel_json,error_message=error_message)
        #return index(error_message)

@app.route('/thanks')
def thanks():
    return "Thank you for your submission!"

if __name__ == '__main__':
    app.run(debug=True)
