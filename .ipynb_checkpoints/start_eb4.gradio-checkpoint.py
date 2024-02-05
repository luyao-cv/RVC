import os
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)
print(script_path)
os.system("pwd")


try:
    import torch
except:
    os.system('sh prepare_env.sh')
    print("downloads env list")

try:
    os.system("python eb4_svc_new_v3.gradio.py")

except: 
    os.system("python eb4_svc_new_v3.gradio.py")



