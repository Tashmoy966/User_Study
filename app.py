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


from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
from werkzeug.security import generate_password_hash, check_password_hash
# from flask_sqlalchemy import SQLAlchemy
import psycopg2
from psycopg2.extras import RealDictCursor
# import re
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
# import io
# import base64
import plotly.graph_objs as go
import plotly
import json
from scipy.interpolate import CubicSpline
import secrets
import sqlite3
# import random as rn
import os
# import csv
import datetime

# PostgreSQL connection configuration
DATABASE_URL = "postgresql://username:password@localhost/yourdatabase"
# Function to handle duplicate usernames
def get_unique_filename(base_name, extension, folder="output"):
    count = 1
    unique_name = f"{base_name}.{extension}"
    while os.path.exists(os.path.join(folder, unique_name)):
        unique_name = f"{base_name}_{count}.{extension}"
        count += 1
    return unique_name
def get_db_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Function to initialize the database
def init_db():
    conn = sqlite3.connect('user_responses.db')
    # conn = get_db_connection()
    c = conn.cursor()
    
    # # Drop the table if it already exists
    # c.execute('DROP TABLE IF EXISTS responses')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )''')
    # Create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            trajectory_index INTEGER,
            trajectory_quality_rating_without_feedback INTEGER,
            trajectory_quality_rating_with_feedback INTEGER,
            hlp_quality_rating INTEGER,
            llm_name TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id) 
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
    # trajectory = data.get('trajectory', [])
    processed_trajectory = data['trajectory']#[[point[0], point[1], point[2], point[3]] for point in trajectory]
        
    return np.array(processed_trajectory)
def save_data(username,response_data):
    folder = "output"
    os.makedirs(folder, exist_ok=True)
    # Prepare filenames
    csv_filename = f"{username}.csv"
    json_filename = f"{username}.json"
    # Save to CSV
    csv_path = os.path.join(folder, csv_filename)
    response_data.to_csv(csv_path, index=False)
    json_path = os.path.join(folder, json_filename)
    # with open(json_path, mode='w', encoding='utf-8') as json_file:
    #     json.dump(json_data, json_file, indent=4)
    response_data.to_json(json_path, orient="records", indent=4)
    message = f"Response saved successfully as {csv_filename} and {json_filename}!"
    flash(message,"info")
    # Clear the response_data after saving and uploading
    response_data.drop(response_data.index, inplace=True)  # Empty the DataFrame
    
app = Flask(__name__)
app.secret_key = session_key # Required for session management


original_trajectory_path='/home/tashmoy/iisc/HIRO/GPT/Manipulator-20241207T074856Z-001/traj_list/trajectory_1.json'
modefied_trajectory_path="/home/tashmoy/iisc/HIRO/GPT/Manipulator-20241207T074856Z-001/traj_list"
feedback_trajectory_path="/home/tashmoy/iisc/HIRO/GPT/traj_kuka/cart/dummy"
original_data=get_json_data(original_trajectory_path)
# print([get_trajectory(get_json_data(modefied_trajectory_path+f"/trajectory_{i+2}.json")) for i in range(5)])
modefied_trajectories=[get_trajectory(get_json_data(modefied_trajectory_path+f"/trajectory_{i+2}.json")) for i in range(5)]
feedback_trajectories=[get_trajectory(get_json_data(feedback_trajectory_path+f"/trajectory_{i+1}.json")) for i in range(5)]

# Sample 3D trajectory data (replace this with your dataset)
num_trajectories = 5  # Number of trajectories
original_trajectories = get_trajectory(original_data) #[np.cumsum(np.random.randn(100, 3), axis=0) for _ in range(num_trajectories)]
# method1_trajectories = [traj + np.random.normal(0, 0.5, traj.shape) for traj in original_trajectories]
#vel = np.cos(np.linspace(0, 10,len(original_trajectories))) + np.random.normal(0, 0.1, len(original_trajectories))
# method2_trajectories = [traj + np.random.normal(0, 1.0, traj.shape) for traj in original_trajectories]
# print(original_trajectories)
# Define some object points and their corresponding colors
object_points = np.array([[point['x'], point['y'], point['z']] for point in original_data["objects"]])#np.array([[1, 2, 1], [3, 1, 0], [0, 0, 3], [2, 3, 2], [1, 1, 1]])
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
["Go lower","Go lower towards left","Keep the start and goal same"],
"Go to the left",
"Go to the right",
"Go faster when you are in the middle of the trajectory."]
hlp=["Shift the goal position left.",
    "Keep the start position same",
    "modify the points in the middle to ensure a gradual change in the trajectory preserving the shape of the trajectory"]
llm_used=["gpt-4o","llama-3.1","gemeni","calude","gpt-4o"]
generated_code="""
    # Python program for
# Creation of Arrays
import numpy as np
 
# Creating a rank 1 Array
arr = np.array([1, 2, 3])
print("Array with Rank 1: \n",arr)
 
# Creating a rank 2 Array
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print("Array with Rank 2: \n", arr)
 
# Creating an array from tuple
arr = np.array((1, 3, 2))
print("\nArray created using "
      "passed tuple:\n", arr)
    """
generated_code1=[
        "\nbanana_position = detect_objects('banana')\nif banana_position is not None:\n    trajectory = get_trajectory()\n    modified_trajectory = []\n    min_distance = 0.2\n    for point in trajectory:\n        x, y, z, velocity = point\n        distance = ((x - banana_position[0]) ** 2 + (y - banana_position[1]\n            ) ** 2 + (z - banana_position[2]) ** 2) ** 0.5\n        if distance < min_distance:\n            direction_vector = [x - banana_position[0], y - banana_position\n                [1], z - banana_position[2]]\n            norm = (direction_vector[0] ** 2 + direction_vector[1] ** 2 + \n                direction_vector[2] ** 2) ** 0.5\n            direction_vector = [(component / norm) for component in\n                direction_vector]\n            x = banana_position[0] + direction_vector[0] * min_distance\n            y = banana_position[1] + direction_vector[1] * min_distance\n            z = banana_position[2] + direction_vector[2] * min_distance\n        modified_trajectory.append((x, y, z, velocity))\n    for i in range(1, len(modified_trajectory) - 1):\n        prev_point = modified_trajectory[i - 1]\n        next_point = modified_trajectory[i + 1]\n        current_point = modified_trajectory[i]\n        smoothed_x = (prev_point[0] + current_point[0] + next_point[0]) / 3\n        smoothed_y = (prev_point[1] + current_point[1] + next_point[1]) / 3\n        smoothed_z = (prev_point[2] + current_point[2] + next_point[2]) / 3\n        modified_trajectory[i\n            ] = smoothed_x, smoothed_y, smoothed_z, current_point[3]\nelse:\n    modified_trajectory = get_trajectory()\n",
        "banana_position = detect_objects('banana')\nif banana_position is not None:\n    trajectory = get_trajectory()\n    modified_trajectory = []\n    min_distance = 0.2\n    speed_reduction_factor = 0.8\n    for point in trajectory:\n        x, y, z, velocity = point\n        distance = ((x - banana_position[0]) ** 2 + (y - banana_position[1]\n            ) ** 2 + (z - banana_position[2]) ** 2) ** 0.5\n        if distance < min_distance:\n            direction_vector = [x - banana_position[0], y - banana_position\n                [1], z - banana_position[2]]\n            norm = (direction_vector[0] ** 2 + direction_vector[1] ** 2 + \n                direction_vector[2] ** 2) ** 0.5\n            direction_vector = [(component / norm) for component in\n                direction_vector]\n            x = banana_position[0] + direction_vector[0] * min_distance\n            y = banana_position[1] + direction_vector[1] * min_distance\n            z = banana_position[2] + direction_vector[2] * min_distance\n            velocity *= speed_reduction_factor\n        modified_trajectory.append((x, y, z, velocity))\n    for i in range(1, len(modified_trajectory) - 1):\n        prev_point = modified_trajectory[i - 1]\n        next_point = modified_trajectory[i + 1]\n        current_point = modified_trajectory[i]\n        smoothed_x = (prev_point[0] + current_point[0] + next_point[0]) / 3\n        smoothed_y = (prev_point[1] + current_point[1] + next_point[1]) / 3\n        smoothed_z = (prev_point[2] + current_point[2] + next_point[2]) / 3\n        modified_trajectory[i\n            ] = smoothed_x, smoothed_y, smoothed_z, current_point[3]\nelse:\n    modified_trajectory = get_trajectory()\n"
    ]
# Initialize DataFrame to store responses
response_data = pd.DataFrame(columns=["Trajectory_Index","Username","trajectory_quality_rating_without_feedback", "trajectory_quality_rating_with_feedback", "hlp_quality_rating","llm_name"])
print(len(response_data))
# print(original_trajectories)
# Function to create a plot and return its base64 encoding
def create_3d_plot(index,mod_trajectory):
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
    # print(original_trajectories[:, 0])

    st_gl_points=[original_trajectories[0],original_trajectories[-1],mod_trajectory[index][0],mod_trajectory[index][-1]]
    st_gl_text=["Original Start","Original End","Modefied Start","Modefied End"]
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
        size=3,
        color=smoothned_data(original_trajectories[:, 3]),  # Assign colors based on z values
        colorscale='BrBG',  # Set the color scale
        colorbar=dict(title='Vel Value'),
        opacity=0.8),  # Color bar title
        # line=dict(color='blue', width=4)),
        # line=dict(color='blue', width=4)
    ))

    #trace_method1 = 
    fig.add_trace(go.Scatter3d(
        x=mod_trajectory[index][:, 0], 
        y=mod_trajectory[index][:, 1], 
        z=mod_trajectory[index][:, 2],
        mode='lines+markers',
        name='Modefied Trajectory',
        marker=dict(
        size=3,
        color=np.random.uniform(0,40,len(mod_trajectory[index][:, 0])) if len(mod_trajectory[index][0])<4 else mod_trajectory[index][:, 3],  # Assign colors based on z values
        colorscale='RdBu',  # Set the color scale
        colorbar=dict(title='Z Value',x=0.9),
        opacity=0.8),  # Color bar title
        # line=dict(color='red', width=4)),
        # line=dict(color='green', width=4)
    ))
    # Add object points to the 3D plot with different colors
    object_names = [point["name"] for point in original_data["objects"]]
    for obj_data,name,color in zip(object_points,object_names,object_colors):
        fig.add_trace(go.Scatter3d(
            x=[obj_data[0]],
            y=[obj_data[1]],
            z=[obj_data[2]],
            mode='markers+text',
            text=name,
            textposition='top center',  # Position of the text relative to the markers
            # hoverinfo='text',
            name=name,
            marker=dict(size=8, color=color, symbol='circle')
        ))
    # Add start and goal marker
    for point,text,color,symbol in zip(st_gl_points,st_gl_text,["green","red","cyan","orange"],["circle","diamond","circle","diamond"]):
        fig.add_trace(go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode='markers+text',
            text=text,
            textposition='top center',  # Position of the text relative to the markers
            # hoverinfo='text',
            name=text,
            marker=dict(size=8, color=color, symbol=symbol)
        ))

     #Add annotations for the object names
    annotations = []
    for point in original_data["objects"]:
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
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(
        x=0.7,  # Horizontal position of the legend (0 to 1, left to right)
        y=0.5,  # Vertical position of the legend (0 to 1, bottom to top)
        bgcolor='rgba(255, 255, 255, 0.5)',  # Background color with some transparency
        bordercolor='black',
        borderwidth=2
    )
    )

    # Create the figure
    #fig = go.Figure(data=[trace_original, trace_method1,object_data], layout=layout)
    fig.update_layout(layout)
    # Convert the figure to JSON to pass to the frontend
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graph_json

# Function to create a 2D plot for plotly
def create_2d_plot(index,mod_trajectory):
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
        x=[i+1 for i in range(len(smoothned_data(mod_trajectory[index][:, 0])))],
        y=smoothned_data(np.random.uniform(0,40,len(mod_trajectory[index][:, 0]))) if len(mod_trajectory[index][0])<4 else smoothned_data(mod_trajectory[index][:, 3]),
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
def home():
    return redirect(url_for('login'))
# Route: Register
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = sqlite3.connect('user_responses.db')
        # conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists. Please try a different one.", "danger")
        # except psycopg2.Error as e:
        #         flash(f"Error: {e.pgerror}", 'danger')
        finally:
            conn.close()

    return render_template('register.html')
# Route: Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('user_responses.db')
        # conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[1], password):
            session['user_id'] = user[0]
            session['username'] = username
            # Add login timestamp to the data
            login_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            flash(f"Login successful! at {login_timestamp}", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password.", "danger")

    return render_template('login.html')
# Route: Dashboard
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash("Please log in to access the dashboard.", "warning")
        return redirect(url_for('login'))
    session['current_index'] = 0 
    # return redirect(url_for('display_trajectory'))
    return render_template('dashboard.html', username=session['username'])


@app.route('/index', methods=['GET', 'POST'])
def index(error_message=None):
    session['current_index'] = 0  # Initialize the trajectory index
    # plot_image = create_3d_plot(session['current_index'])
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
    plot_json = create_3d_plot(current_index,modefied_trajectories)
    vel_json=create_2d_plot(current_index,modefied_trajectories)
    feedback_vel_json=create_2d_plot(current_index,feedback_trajectories)
    feedback_plot_json=create_3d_plot(current_index,feedback_trajectories)
    return render_template('index.html', generated_code=generated_code1[0],llm_names=llm_names,plot_json=plot_json, current_index=current_index, num_trajectories=num_trajectories,instruction_text=instruction_text,hlp_text=hlp_text,vel_json=vel_json,feedback_plot_json=feedback_plot_json,feedback_vel_json=feedback_vel_json)


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    global username
    username = session['username'].replace(" ", "_").strip()
    current_index = session['current_index']
    trajectory_quality_rating_without_feedback = request.form.get('trajectory_quality_rating_without_feedback')
    trajectory_quality_rating_with_feedback = request.form.get('trajectory_quality_rating_with_feedback')
    hlp_quality_rating = request.form.get('hlp_quality_rating')
    llm_name=llm_used[current_index]
    # method2_rating = request.form.get('method2_rating')
    # comparison = request.form.get('comparison')

    if trajectory_quality_rating_without_feedback and trajectory_quality_rating_with_feedback and hlp_quality_rating and username:  #and method2_rating and comparison
        # Connect to the database
        conn = sqlite3.connect('user_responses.db')
        # conn = get_db_connection()
        c = conn.cursor()

        # Insert the data into the responses table
        c.execute('''
            INSERT INTO responses (trajectory_index, username, trajectory_quality_rating_without_feedback, trajectory_quality_rating_with_feedback, hlp_quality_rating, llm_name)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (current_index, username,trajectory_quality_rating_without_feedback, trajectory_quality_rating_with_feedback,hlp_quality_rating, llm_name))

        # Commit and close the connection
        conn.commit()
        conn.close()
        global response_data
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # response_data.loc[len(response_data)] = ["Username", username, "","",""]
        # response_data.loc[len(response_data)] = ["trajectory_quality_rating_without_feedback", trajectory_quality_rating_without_feedback, "","",""]
        # response_data.loc[len(response_data)] = ["trajectory_quality_rating_with_feedback", trajectory_quality_rating_with_feedback, "","",""]
        # response_data.loc[len(response_data)] = ["hlp_quality_rating",hlp_quality_rating ,"","",""]
        # response_data.loc[len(response_data)] = ["llm_name",llm_name ,"","",""]
        index_exists=response_data[response_data['Trajectory_Index'] == current_index]
        if not index_exists.empty:
        # If user exists, replace their ratings
            response_data.loc[response_data['Trajectory_Index'] == current_index, 'trajectory_quality_rating_without_feedback'] = trajectory_quality_rating_without_feedback
            response_data.loc[response_data['Trajectory_Index'] == current_index, 'trajectory_quality_rating_with_feedback'] = trajectory_quality_rating_with_feedback
            response_data.loc[response_data['Trajectory_Index'] == current_index, 'hlp_quality_rating'] = hlp_quality_rating
            response_data.loc[response_data['Trajectory_Index'] == current_index, 'llm_name'] = llm_name
        else:
            response_data.loc[len(response_data)] = [
            current_index, # Trajectory Index
            username,  # Username
            trajectory_quality_rating_without_feedback,  # Trajectory rating without feedback
            trajectory_quality_rating_with_feedback,  # Trajectory rating with feedback
            hlp_quality_rating,  # HLP quality rating
            llm_name  # LLM name
            ]
        # # Save to JSON
        # json_data = {
        #     "Trajectory_Index":current_index,
        #     "Username": username,
        #     "trajectory_quality_rating_without_feedback": trajectory_quality_rating_without_feedback,
        #     "trajectory_quality_rating_with_feedback": trajectory_quality_rating_with_feedback,
        #     "hlp_quality_rating": hlp_quality_rating,
        #     "llm_name": llm_name
        # }
        
        print(response_data)
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
        vel_json=create_2d_plot(current_index,modefied_trajectories)
        feedback_vel_json=create_2d_plot(current_index,feedback_trajectories)
        plot_json = create_3d_plot(current_index,modefied_trajectories)
        feedback_plot_json=create_3d_plot(current_index,feedback_trajectories)
        return render_template('index.html', generated_code=generated_code1[0],llm_names=llm_names, plot_json=plot_json, current_index=current_index, num_trajectories=num_trajectories,instruction_text=instruction_text,hlp_text=hlp_text,vel_json=vel_json,error_message=error_message,feedback_plot_json=feedback_plot_json,feedback_vel_json=feedback_vel_json)
        #return index(error_message)



@app.route('/thanks')
def thanks():
    flash("Thank you for your submission!", "info")
    save_data(username,response_data)
    return redirect(url_for('dashboard'))
# Route: Logout
@app.route('/logout')
def logout():
    session.clear()
    # Add login timestamp to the data
    logout_timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    flash(f"You have been logged out at {logout_timestamp}.", "info")
    return redirect(url_for('dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
