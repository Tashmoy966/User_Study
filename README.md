# User_Study
User Study over Trajectory Data

# Clone Repositorty 
git clone https://github.com/Tashmoy966/User_Study.git

# Move To the Cloned Directory
cd /Path_to/User_Study

# Install Dependencies
pip install -r requirements.txt

# Execution
python app.py

**1.** The terminal should display something similar:

![image](https://github.com/user-attachments/assets/152ae484-32a0-4ba8-9e11-76f6b8e68de5)

**2.** Open your desired browser and put the localhost IP mentioned something like **http://127.0.0.1:5000**

  A login page will appear

  ![image](https://github.com/user-attachments/assets/98d9e278-3d36-4195-bdbc-04330a46558e)

**3.** **Register** if u r a **new user** or **login** directly

**4.** Once logged in a **dashboard** appears **click submit response** to start the rating process

![image](https://github.com/user-attachments/assets/d50b6cbe-98cc-48c5-99c5-ac6d3802be16)

![image](https://github.com/user-attachments/assets/1dea3b2e-ba60-40a1-b6b4-077e14383a41)

![image](https://github.com/user-attachments/assets/8fa96d30-b209-4255-bf14-ade2d8882e12)

**The page information follows:**

**a.** **LLM** used for trajectory adaptation for the respective visualization.

**b.** **Original**, **zero-shot**, and **feedback-oriented trajectories** can be visualized, and switching between **zero-shot** and **feedback trajectories** can be done using the given **dropdown menu**.

**c.** **Velocity profile** for the trajectories.

**d.** Info regarding the **user**, corresponding **feedback instruction**, **generated high-level plan**, and **code**(**Click**) is displayed.

**e.** At last **three single-choice questions** are displayed select the option according to your preference collected from the above visualization.

**f.** Click **Submit Response** to move to the next trajectory. **Next** and **Previous** buttons are also provided for flexibility over navigation during the process.



Once every rating is provided plz **logout**


![image](https://github.com/user-attachments/assets/696ec2a3-a877-4037-915a-78cbe10e9bb9)


**Ratings** will be saved as a **.json and .csv** with **prefix** as your **username** in **output folder**.
