# Face Attendance System



This project is a Face Attendance System that uses the `face_recognition` library for facial recognition. Follow the instructions below to set up the project and get it running.



## Prerequisites



- Python 3.6 or higher

- Git

- PowerShell (for Windows users)



## Setup Instructions



### 1. Clone the Repository



First, clone the repository to your local machine:



```sh

git clone <repository_url>

cd Face Attendance System

```



### 2. Create a Virtual Environment



Create a virtual environment to manage dependencies:



```sh

python -m venv venv

```



### 3. Activate the Virtual Environment



Activate the virtual environment:



For Windows:



```sh

.\venv\Scripts\Activate.ps1

```



For macOS/Linux:



```sh

source venv/bin/activate

```



### 4. Upgrade pip



Ensure you have the latest version of `pip`:



```sh

python.exe -m pip install --upgrade pip

```



### 5. Install Dependencies



Install the required dependencies, including `face_recognition_models` and `face_recognition`:



```sh

# Install face_recognition_models from GitHub

pip install git+https://github.com/ageitgey/face_recognition_models



# Install face_recognition

pip install face_recognition



# Install other dependencies

pip install -r requirements.txt

```



### 6. Install dlib



Install `dlib`, which is required by `face_recognition`:



1. Download the appropriate wheel file for your Python version from [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/).

2. Install the downloaded wheel file. For example:



```sh

pip install path\to\downloaded\wheel_file.whl

```



Alternatively, you can try installing `dlib` directly:



```sh

pip install dlib

```



### 7. Run the Streamlit App



Run the Streamlit application:



```sh

streamlit run streamlit_app.py

```



## Troubleshooting



### Script Execution Policy (Windows)



If you encounter an error related to script execution policy, you may need to change the execution policy for PowerShell:



1. Open PowerShell as an administrator.

2. Run the following command:



```sh

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

```



### Missing or Incorrect Wheel File



If you get an error like `ERROR: dlib-19.24.1-cp311-cp311-win_amd64.whl is not a supported wheel on this platform`, ensure you have downloaded the correct wheel file for your Python version and system architecture.



## Contributing



If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.



## License



This project is licensed under the MIT License. See the LICENSE file for details.



