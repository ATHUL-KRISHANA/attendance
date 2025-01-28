import subprocess
from datetime import datetime
import os

# Get the current time
current_time = datetime.now()

# Path to the virtual environment's Python executable
python_path = "/home/user/Documents/Attendence/env/bin/python"

# Path to manage.py
manage_py_path = "/home/user/Documents/Attendence/project/manage.py"

try:
    # Start the Django development server
    print("Starting Django server...")
    server_command = f"{python_path} {manage_py_path} runserver"
    os.system(server_command)  # This will block and keep the server running

    # If the current time is before 10:00 AM, run mark_absent.py
    if current_time.hour < 10:
        print("Running mark_absent.py before 10:00 AM...")
        mark_absent_command = [python_path, "/home/user/Documents/Attendence/project/mark_absent.py"]
        result = subprocess.run(mark_absent_command, capture_output=True, text=True)

        # Print the output of mark_absent.py
        print("Output from mark_absent.py:")
        print(result.stdout)
        if result.stderr:
            print("Errors from mark_absent.py:")
            print(result.stderr)
    else:
        print("Current time is after 10:00 AM. Skipping mark_absent.py.")

except Exception as e:
    print(f"An error occurred: {e}")
