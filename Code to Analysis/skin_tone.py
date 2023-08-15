import os
import subprocess

def create_folder(path):
    try:
        os.mkdir(path)
    except:
        pass

API_NAME = "facepp"

create_folder("result")
create_folder("result/"+API_NAME)
countries = os.listdir(API_NAME)
for country in countries:
    create_folder("result/"+API_NAME+"/"+country)
    img_types = os.listdir(API_NAME+"/"+country)
    for img_type in img_types:
        input_img_path = API_NAME+"/"+country+"/"+img_type+"/"
        out_res_path = "result/"+API_NAME+"/"+country+"/"+img_type+"/"
        create_folder(out_res_path)
        subprocess.run(['stone','-i',input_img_path,'-o',out_res_path], stdout=subprocess.PIPE)