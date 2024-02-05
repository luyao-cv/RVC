
import logging
import os
import pathlib
import platform
if platform.system().lower() == 'windows':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
elif platform.system().lower() == 'linux':
    temp = pathlib.WindowsPath
    pathlib.WindowsPath = pathlib.PosixPath
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import torch
import multiprocessing



from configs.config import Config
import gradio as gr
import pathlib

import logging

import os
import sys
from dotenv import load_dotenv

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC

import fairseq
import warnings
import shutil


logging.getLogger("numba").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/uvr5_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "assets/weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)


config = Config()
vc = VC(config)


if config.dml == True:

    def forward_dml(ctx, x, scale):
        ctx.scale = scale
        res = x.clone().detach()
        return res

    fairseq.modules.grad_multiply.GradMultiply.forward = forward_dml
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(
            value in gpu_name.upper()
            for value in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
            ]
        ):
            # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])

from gradio.events import Dependency

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"
# import pdb
# pdb.set_trace()
# weight_root = os.getenv("weight_root")
# weight_uvr5_root = os.getenv("weight_uvr5_root")
# index_root = os.getenv("index_root")


config = Config()
vc = VC(config)


thread_count = multiprocessing.cpu_count()

print("Use",thread_count,"cpu cores for computing")

torch.set_num_threads(thread_count)
torch.set_num_interop_threads(thread_count)


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)

vc.get_vc('nifeng.pth',0.33,0.33)

def main():
    app = gr.Blocks(css="footer {visibility: hidden}",title="飞桨Feng Brother")
    with app:
        gr.HTML("<center>"
                "<h1>🌊💕🎶 声音克隆之实时体验 X 训练&推理 </h1>"
                "</center>")
        gr.Markdown("## <center>⚡ 只需3分钟训练，快速复刻您喜欢的声音；在线体验克隆飞桨帅气RD之Feng Brother的声音</center>")
        gr.Markdown("### <center>更多精彩音频应用，正在持续更新～联系作者：luyao15@baidu.com💕</center>")

 
        with gr.Tab("🎶 - 声音克隆-推理体验"):
            gr.Markdown("请录制或上传一段“干净”声音的音频，并点击”声音复刻“，优先处理麦克风声音")
            with gr.Row():
                with gr.Column():
                    record_audio_prompt = gr.Audio(label='请在此用麦克风录制您喜欢的声音', source='microphone', interactive=True, type="filepath")
                    upload_audio_prompt = gr.Audio(label='或者上传您的语音文件', source='upload', interactive=True, type="filepath")
                    
                with gr.Column():
                    audio_output = gr.Audio(label="为您合成的专属语音", elem_id="tts-audio", type="filepath", interactive=False)
                    vc_transform0 = gr.Number(
                                label="变调(整数, 半音数量, 升八度12降八度-12)", value=0
                            )
                    text_output_1 = gr.Textbox(label="声音克隆进度")
                    btn_1 = gr.Button("声音复刻", variant="primary")
                    btn_1.click(vc.vc_nifeng,
                              inputs=[record_audio_prompt, upload_audio_prompt, vc_transform0],
                              outputs=[text_output_1, audio_output],api_name="infer_convert")
                    
                    
        with gr.Tab("💕 - 声音克隆-训练体验"):
            gr.Markdown("现在开始奇妙的声音克隆训练体验之旅吧！（敬请期待......）")
            

        gr.Markdown("### <center>注意❗：请不要生成会对个人以及组织造成侵害的内容，此程序仅供科研、学习及个人娱乐使用，作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责。</center>")

    app.launch(share=True, server_port=8918, server_name="0.0.0.0")

if __name__ == "__main__":
    # formatter = (
    #     "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    # )
    # logging.basicConfig(format=formatter, level=logging.INFO)
    main()