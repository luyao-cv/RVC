
echo $PWD
export pwd_dir=$PWD
cd $pwd_dir

cd /opt/conda/envs/python35-paddle120-env/lib/python3.10/
rm -rf site-packages
wget https://bj.bcebos.com/v1/paddlenlp/models/community/luyao15/AISinger/site-packages.tar.gz
tar -xzvf site-packages.tar.gz -C ./
rm -rf site-packages.tar.gz

cd $pwd_dir
pip install -r env_re.txt

pip uninstall gradio -y
pip install gradio==3.50.1

# pip install https://studio-package.bj.bcebos.com/aistudio-0.1.0-py3-none-any.whl 

wget https://bj.bcebos.com/v1/paddlenlp/models/community/luyao15/RVC/assets.tar.gz
tar -xzvf assets.tar.gz -C ./
rm -rf assets.tar.gz
cp -r ljr.pth assets/weights/
cp -r zh.pth assets/weights/
cp -r mgr.pth assets/weights/
cp -r luyao.pth assets/weights/


