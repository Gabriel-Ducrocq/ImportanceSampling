import subprocess

N_scripts = 50

for i in range(N_scripts):
    subprocess.run(["python", "main.py", str(i)])