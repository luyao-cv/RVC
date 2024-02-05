import requests
import json
import os

from pathlib import Path
script_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_path)
print(script_path)
from langchain.chat_models import ErnieBotChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from gradio.inputs import Textbox

from infer_file import Infer_Fun
from utils.hparams import set_hparams
from utils.hparams import hparams as hp
import importlib
from sklearn.cluster import MiniBatchKMeans
import faiss
import jieba
import time
import re
import shlex
import locale
import random
import aistudio
import yaml
import numpy as np
import soundfile as sf
import glob
import subprocess
import sys
import time
import traceback

from time import sleep
import librosa
import soundfile
import torch
import wave
import ast
from pydub import AudioSegment
from subprocess import Popen
from random import shuffle
import warnings
import traceback
import threading
from pypinyin import pinyin, lazy_pinyin
from concurrent.futures import ProcessPoolExecutor

from song_templates import lyrics_dict, notes_duration_dict, notes_dict, lyrics_slot_dict, Songs_CN_dict, \
    ONOMATOPOEIC_WORDS, Songs_ST_dict, Songs_V_dict, Singer_F_dict
from song_prompt import GEN_LYRICS_MAIN, WORD_REPLACE, THEME_EXTEND, SUMMARY

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
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["weight_root"] = "assets/weights"
os.environ["weight_uvr5_root"] = "assets/uvr5_weights"
os.environ["index_root"] = "logs"
os.environ["rmvpe_root"] = "assets/rmvpe"

import torch
import multiprocessing


from configs.config import Config
import gradio as gr
import pathlib

import logging

import os
import sys
from dotenv import load_dotenv
from lyrics_to_melody import lyrics_to_melody, SongsType

now_dir = os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC

import fairseq
import warnings
import shutil
from subprocess import Popen

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


training_threadpool = ProcessPoolExecutor(max_workers=1)


TRAIN_DATA = """
当我第一次见到那只小狗时，它正蜷缩在街角的一个破纸箱里，小小的身体颤抖着，湿漉漉的眼睛里充满了恐惧和期待。它的毛发蓬松，但已经被污垢染得斑驳。我伸出手，它畏缩了一下，但很快便在我的手掌下轻轻摩擦，发出细微的呜咽声。

那是一个寒冷的冬日，风刺骨般地吹着。我蹲下身，小心翼翼地把它抱起来，它的体重轻得令人吃惊。我把它带回了家，用温暖的水慢慢清洗掉它身上的污垢，小狗在水中颤抖，但它的眼睛里闪烁着对生命新希望的光芒。

我给它取名叫“奇奇”。奇奇很快就适应了新家，它活泼可爱，总是围着我转，仿佛它能理解我所有的情绪。每当我下班回家，它总是第一个迎上来，摇着尾巴，用它那湿漉漉的鼻子蹭我的手。在它的陪伴下，我再也不觉得孤独和寂寞。

奇奇对世界充满了好奇，每当我们一起散步时，它总是兴奋地跑来跑去，对每一株草、每一个角落都表现出无比的兴趣。它的眼睛里总是充满了热情和好奇，好像在说：“这个世界是如此美妙。”

有一次，我带它去公园玩耍。奇奇欢快地奔跑在草地上，突然，它停下来，对着一只蝴蝶摇尾巴。那一刻，我被它纯真的快乐深深感染，忘记了生活的压力和烦恼，只想和它一起在这片绿地上自由地奔跑。

奇奇不仅给我的生活带来了欢乐，它还教会了我爱和责任。有时，当我看着它熟睡的样子，我会想，它是如此依赖我，我的一个小动作，一个决定，都会影响到它的生活。我学会了更加关心和体贴，不再只是为自己而活。

随着时间的流逝，奇奇从一只脆弱的小狗长成了一只健康、快乐的狗。每当我感到疲倦或沮丧时，只需要看看奇奇那充满活力和乐观的眼神，我的心情就会大为好转。它教会了我，无论生活多么艰难，总有希望和快乐在等着我们。

现在，我和奇奇是不可分割的朋友。它不仅是我的宠物，更是我的家人，我的伙伴。我感激命运让我遇到了奇奇，它用它的方式改变了我的世界，给我的生活增添了无尽的色彩。

我常常想，如果不是那天我在街角遇见了奇奇，我的生活会是怎样的一番景象。也许，更加平淡无奇，缺少了这份特别的情感和陪伴。

奇奇的到来，让我的生活变得丰富多彩，充满了爱和欢笑。它教会了我许多宝贵的生活哲理，让我成为了一个更好的人。我希望能和奇奇一起度过更多美好的岁月，共同书写我们的故事。

"""



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
    gpu_info = "很遗憾您这没有能用的显卡来支持您训练"
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool", **kwargs)

    def get_block_name(self):
        return "button"


config = Config()
vc = VC(config)


thread_count = multiprocessing.cpu_count()

print("Use",thread_count,"cpu cores for computing")

torch.set_num_threads(thread_count)
torch.set_num_interop_threads(thread_count)


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)



os.environ["AISTUDIO_LOG"] = "debug"

llm = ErnieBotChat(ernie_client_id='96BMhQM5simx6R97yDl483Zm',
                   ernie_client_secret='9e05mDOjHoyXD7Sb9GA1l420uaZ6vGMo',
                   model_name='ERNIE-Bot-4',
                   top_p=0.4)

# 提取json格式的文本
def extract_json(text):
    try:
        pattern = r'```json(.*?)```'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            summary = match.group(1).strip().replace('\n', '')
            return ast.literal_eval(summary)
        return None
    except Exception as e:
        print(e)

# 扩展主题关键词
def extend_theme(theme):
    prompt = PromptTemplate(input_variables=["theme"], template=THEME_EXTEND)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    results = llm_chain.run(**{"theme": theme})
    results = extract_json(results)
    print("原始关键词：", results)
    return results


def get_summary(theme):
    prompt = PromptTemplate(input_variables=["theme"], template=SUMMARY)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    results = llm_chain.run(**{"theme": theme})
    results = extract_json(results)
    results = results["summary"]
    print("原始摘要：", results)
    return results



def gen_lyrics(theme, theme_summary, source, source_slot):
    # first trail of lyrics generation
    prompt = PromptTemplate(input_variables=["theme", "theme_summary", "source"], template=GEN_LYRICS_MAIN)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    results = llm_chain.run(**{"theme": theme, "theme_summary": theme_summary, "source": source})
    results = extract_json(results)
    results = results["lyrics"]
    # print(results)
    # filter results
    ori_list = source.split('\n')
    ori_line_count = len(ori_list) - 2
    punc = '[’!"#$%&\'()*+,-./:;<=>?？@[\\]^_`{|}~。！，]+'
    result_pure = re.sub(punc, "", results).replace(" ", "")
    result_pure = result_pure.split('\n')
    if len(result_pure) < ori_line_count:
        resdi = ori_line_count - len(result_pure)
        for i in range(resdi):
            result_pure.append(random.choice(ONOMATOPOEIC_WORDS))
    target_list = result_pure[0:ori_line_count]
    match_list = []
    correct_list = []

    # check words count, adjust words
    for i in range(ori_line_count):
        if len(target_list[i]) != len(ori_list[i + 1]):
            ori_slot_list = source_slot.split('\n')
            tmp_prompt = PromptTemplate(input_variables=["theme", "ori_sentence"], template=WORD_REPLACE)
            tmp_llm_chain = LLMChain(llm=llm, prompt=tmp_prompt)
            tmp_result = tmp_llm_chain.run(**{"theme": theme, "ori_sentence": ori_slot_list[i + 1]})
            tmp_result = extract_json(tmp_result)
            tmp_result = tmp_result["line"]
            tmp_result = tmp_result.replace("\n", "")
            tmp_result_pure = re.sub(punc, "", tmp_result.split('\n')[0])
            if "：" in tmp_result_pure:
                tmp_result_pure = tmp_result_pure.split("：")[1]
            target_list[i] = tmp_result_pure.replace(" ", "")
            correct_list.append(i + 1)
        pattern = re.compile(r'[^\u4e00-\u9fa5]')
        chinese = re.sub(pattern, "", target_list[i])
        target_list[i] = chinese
        match_list.append(len(ori_list[i + 1]) - len(target_list[i]))

    # combine with SP
    final_result = 'SP'.join(target_list)
    return target_list, match_list, correct_list, final_result



def retry_gen_lyrics(theme, theme_summary, source, source_slot, max_retries=3, delay_seconds=1):
    retries = 0
    while retries < max_retries:
        try:
            target_list, match_list, correct_list, final_result = gen_lyrics(theme, theme_summary, source, source_slot)
            return target_list, match_list, correct_list, final_result
        except Exception as e:
            print(f"歌词生成失败：{e}")
            retries += 1
            if retries < max_retries:
                print(f"进行第 {retries} 次重试...")
                time.sleep(delay_seconds)  # 可选：延迟一定时间后再重试
            else:
                print("重试次数达到上限，放弃重试。")
                break

 
def cut_duration(lyrics, durations):
    durations_lst = []
    idx = 0
    rest = []
    for i in range(len(lyrics)):
        len_lyric = len(lyrics[i])
        if i == 0:
            durations_lst.append(durations[idx:idx + len_lyric])
            rest.append(durations[idx + len_lyric])
        else:
            if i == len(lyrics) - 1:
                durations_lst.append(durations[idx:idx + len_lyric])
            else:
                durations_lst.append(durations[idx:idx + len_lyric])
                rest.append(durations[idx + len_lyric])
        idx += len_lyric + 1
    return durations_lst, rest

def judge_length(word_lst, need_len):
    res = ""
    for word in word_lst:

        tmp_len = len(res) + len(word)
        if tmp_len == need_len:
            res += word
            print("修正后长度：{}, 歌曲模板长度：{}, 修正后歌词：{}".format(len(res), need_len, res))
            return res
        elif tmp_len > need_len:
            resdi = tmp_len - need_len
            if resdi >= 2:
                if need_len - len(res) == 1:
                    res += random.choice(ONOMATOPOEIC_WORDS)
                else:
                    res += word[:-resdi]
            else:
                res = res + word[:-2] + random.choice(ONOMATOPOEIC_WORDS)
            print("修正后长度：{}, 歌曲模板长度：{}, 修正后歌词：{}".format(len(res), need_len, res))
            return res
        else:
            res += word


def gen_notes_and_durations(notes, durations, lyrics, new_lyrics):
    lyrics = lyrics.split("\n")[1:-1]
    new_lyrics = new_lyrics.split("SP")
    notes_o = notes.split(" | rest | ")
    notes = []
    for n in notes_o:
        n = n.split(" | ")
        notes.append(n)
    durations = durations.split(" | ")
    durations, rest = cut_duration(lyrics, durations)
    new_lyrics = new_lyrics[:len(lyrics)]
    durations_res = []
    notes_res = []
    new_lyrics_res = []
    for i in range(len(lyrics)):
        old_len = len(lyrics[i])
        new_len = len(new_lyrics[i])
        if old_len == new_len:
            notes_res.extend(notes[i])
            durations_res.extend(durations[i])
            new_lyrics_res.append(new_lyrics[i])
        elif new_len > old_len:
            residual = new_len - old_len
            if residual >= 2:
                word_lst = jieba.lcut(new_lyrics[i], cut_all=False)
                lyric_i = judge_length(word_lst, old_len)
                new_lyrics_res.append(lyric_i)
                notes_res.extend(notes[i])
                durations_res.extend(durations[i])
            else:
                notes_res.extend(notes[i][:-residual])
                durations_res.extend(durations[i][:-residual])
 
                while residual > 0:
                    if " " in durations[i][-residual] or " " in notes[i][-residual]:
                        print(durations[i][-residual].split(" ")[0], durations[i][-residual].split(" ")[1])
                        notes_res.append(notes[i][-residual].split(" ")[0])
                        notes_res.append(notes[i][-residual].split(" ")[1])
                        durations_res.append(durations[i][-residual].split(" ")[0])
                        durations_res.append(durations[i][-residual].split(" ")[1])
                    else:
                        notes_res.append(notes[i][-residual])
                        notes_res.append(notes[i][-residual])
                        durations_res.append(str(float(durations[i][-residual]) / 2))
                        durations_res.append(str(float(durations[i][-residual]) / 2))
                    residual -= 1
                new_lyrics_res.append(new_lyrics[i])
        else:
            residual = new_len - old_len
            notes_res.extend(notes[i])
            durations_res.extend(durations[i])
            new_lyrics_res.append(new_lyrics[i] + random.choice(ONOMATOPOEIC_WORDS) * abs(residual))
        #print(len(notes_res), len(durations_res), len("".join(new_lyrics_res).replace("SP", "|")))
        #print(new_lyrics_res)
        #print(" | ".join(notes_res))
        #print(" | ".join(durations_res))
        #print("".join(new_lyrics_res))
        if i != len(lyrics) - 1:
            notes_res.append("rest")
            durations_res.append(rest[i])
            new_lyrics_res.append("SP")

    new_lyrics_res = "".join(new_lyrics_res)
    print(len(notes_res), len(durations_res), len(new_lyrics_res.replace("SP", "|")))
    notes_res = " | ".join(notes_res)
    durations_res = " | ".join(durations_res)
    return notes_res, durations_res, new_lyrics_res
def gen_songs_needs(theme, song):
    theme_summary = get_summary(theme)
    target_list, match_list, correct_list, new_lyrics_tmp = retry_gen_lyrics(theme=theme,
                                                                             theme_summary=theme_summary,
                                                                             source=lyrics_dict.get(song),
                                                                             source_slot=lyrics_slot_dict.get(song))
    notes = notes_dict.get(song)
    duration = notes_duration_dict.get(song)
    lyrics = lyrics_dict.get(song)
    if not correct_list:
        notes_res, durations_res, new_lyrics_res = notes, duration, new_lyrics_tmp
    else:
        notes_res, durations_res, new_lyrics_res = gen_notes_and_durations(notes, duration, lyrics, new_lyrics_tmp)
    print("新歌词：{}\n曲谱：{}\n时长：{}".format(new_lyrics_res, notes_res, durations_res))
    return notes_res, durations_res, new_lyrics_res


class Info:
    def __init__(self) -> None:
        pass

LANGUAGE_LIST = ['zh_CN', 'en_US']
LANGUAGE_ALL = {
    'zh_CN': {
        'SUPER': 'END',
        'LANGUAGE': 'zh_CN',
        '初始化成功': '初始化成功',
        '就绪': '就绪',
        '预处理-训练': '预处理-训练',
        '训练说明': '训练说明',
        '### 预处理参数设置': '### 预处理参数设置',
        '模型名称': '模型名称',
        'f0提取器': 'f0提取器',
        '预处理线程数': '预处理线程数',
        '### 训练参数设置': '### 训练参数设置',
        '学习率': '学习率',
        '批大小': '批大小',
        '训练日志记录间隔（step）': '训练日志记录间隔（step）',
        '验证集验证间隔（epoch）': '验证集验证间隔（epoch）',
        '检查点保存间隔（epoch）': '检查点保存间隔（epoch）',
        '保留最新的检查点文件(0保存全部)': '保留最新的检查点文件(0保存全部)',
        '是否添加底模': '是否添加底模',
        '### 开始训练': '### 开始训练',
        '打开数据集文件夹': '打开数据集文件夹',
        '一键训练': '一键训练',
        '启动Tensorboard': '启动Tensorboard',
        '### 恢复训练': '### 恢复训练',
        '从检查点恢复训练进度': '从检查点恢复训练进度',
        '刷新': '刷新',
        '恢复训练': '恢复训练',
        '推理': '推理',
        '推理说明': '推理说明',
        '### 推理参数设置': '### 推理参数设置',
        '变调': '变调',
        '文件列表': '文件列表',
        '选择要导出的模型': '选择要导出的模型',
        '刷新模型和音色': '刷新模型和音色',
        '导出模型': '导出模型',
        '选择音色文件': '选择音色文件',
        '选择待转换音频': '选择待转换音频',
        '开始转换': '开始转换',
        '输出音频': '输出音频',
        '打开文件夹失败！': '打开文件夹失败！',
        '开始预处理': '开始预处理',
        '开始训练': '开始训练',
        '开始导出模型': '开始导出模型',
        '导出模型成功': '导出模型成功',
        '出现错误：': '出现错误：',
        '缺少模型文件': '缺少模型文件',
        '缺少文件': '缺少文件',
        '已清理残留文件': '已清理残留文件',
        '无需清理残留文件': '无需清理残留文件',
        '开始推理': '开始推理',
        '推理成功': '推理成功',
        '### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)第一次编写|[@thestmitsuk](https://github.com/thestmitsuki)二次补完': '### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)第一次编写|[@thestmitsuk](https://github.com/thestmitsuki)二次补完',
        ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音": ">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音",
        "A模型权重": "A模型权重",
        "A模型路径": "A模型路径",
        "B模型路径": "B模型路径",
        "F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调": "F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调",
        "Index Rate": "Index Rate",
        "Onnx导出": "Onnx导出",
        "Onnx输出路径": "Onnx输出路径",
        "RVC模型路径": "RVC模型路径",
        "ckpt处理": "ckpt处理",
        "harvest进程数": "harvest进程数",
        "index文件路径不可包含中文": "index文件路径不可包含中文",
        "pth文件路径不可包含中文": "pth文件路径不可包含中文",
        "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程": "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程",
        "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. ": "step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. ",
        "step1:正在处理数据": "step1:正在处理数据",
        "step2:正在提取音高&正在提取特征": "step2:正在提取音高&正在提取特征",
        "step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. ": "step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. ",
        "step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)": "step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)",
        "step3: 填写训练设置, 开始训练模型和索引": "step3: 填写训练设置, 开始训练模型和索引",
        "step3a:正在训练模型": "step3a:正在训练模型",
        "一键训练": "一键训练",
        "也可批量输入音频文件, 二选一, 优先读文件夹": "也可批量输入音频文件, 二选一, 优先读文件夹",
        "人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。": "人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>  (1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。",
        "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2": "以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2",
        "伴奏人声分离&去混响&去回声": "伴奏人声分离&去混响&去回声",
        "保存名": "保存名",
        "保存的文件名, 默认空为和源文件同名": "保存的文件名, 默认空为和源文件同名",
        "保存的模型名不带后缀": "保存的模型名不带后缀",
        "保存频率save_every_epoch": "保存频率save_every_epoch",
        "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果": "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果",
        "修改": "修改",
        "修改模型信息(仅支持weights文件夹下提取的小模型文件)": "修改模型信息(仅支持weights文件夹下提取的小模型文件)",
        "停止音频转换": "停止音频转换",
        "全流程结束！": "全流程结束！",
        "刷新音色列表和索引路径": "刷新音色列表和索引路径",
        "加载模型": "加载模型",
        "加载预训练底模D路径": "加载预训练底模D路径",
        "加载预训练底模G路径": "加载预训练底模G路径",
        "单次推理": "单次推理",
        "卸载音色省显存": "卸载音色省显存",
        "变调(整数, 半音数量, 升八度12降八度-12)": "变调(整数, 半音数量, 升八度12降八度-12)",
        "后处理重采样至最终采样率，0为不进行重采样": "后处理重采样至最终采样率，0为不进行重采样",
        "否": "否",
        "响应阈值": "响应阈值",
        "响度因子": "响度因子",
        "处理数据": "处理数据",
        "导出Onnx模型": "导出Onnx模型",
        "导出文件格式": "导出文件格式",
        "常见问题解答": "常见问题解答",
        "常规设置": "常规设置",
        "开始音频转换": "开始音频转换",
        "很遗憾您这没有能用的显卡来支持您训练": "很遗憾您这没有能用的显卡来支持您训练",
        "性能设置": "性能设置",
        "总训练轮数total_epoch": "总训练轮数total_epoch",
        "批量推理": "批量推理",
        "批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. ": "批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. ",
        "指定输出主人声文件夹": "指定输出主人声文件夹",
        "指定输出文件夹": "指定输出文件夹",
        "指定输出非主人声文件夹": "指定输出非主人声文件夹",
        "推理时间(ms):": "推理时间(ms):",
        "推理音色": "推理音色",
        "提取": "提取",
        "提取音高和处理数据使用的CPU进程数": "提取音高和处理数据使用的CPU进程数",
        "是": "是",
        "是否仅保存最新的ckpt文件以节省硬盘空间": "是否仅保存最新的ckpt文件以节省硬盘空间",
        "是否在每次保存时间点将最终小模型保存至weights文件夹": "是否在每次保存时间点将最终小模型保存至weights文件夹",
        "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速": "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速",
        "显卡信息": "显卡信息",
        "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.": "本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.",
        "查看": "查看",
        "查看模型信息(仅支持weights文件夹下提取的小模型文件)": "查看模型信息(仅支持weights文件夹下提取的小模型文件)",
        "检索特征占比": "检索特征占比",
        "模型": "模型",
        "模型推理": "模型推理",
        "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况": "模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况",
        "模型是否带音高指导": "模型是否带音高指导",
        "模型是否带音高指导(唱歌一定要, 语音可以不要)": "模型是否带音高指导(唱歌一定要, 语音可以不要)",
        "模型是否带音高指导,1是0否": "模型是否带音高指导,1是0否",
        "模型版本型号": "模型版本型号",
        "模型融合, 可用于测试音色融合": "模型融合, 可用于测试音色融合",
        "模型路径": "模型路径",
        "每张显卡的batch_size": "每张显卡的batch_size",
        "淡入淡出长度": "淡入淡出长度",
        "版本": "版本",
        "特征提取": "特征提取",
        "特征检索库文件路径,为空则使用下拉的选择结果": "特征检索库文件路径,为空则使用下拉的选择结果",
        "男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. ": "男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. ",
        "目标采样率": "目标采样率",
        "算法延迟(ms):": "算法延迟(ms):",
        "自动检测index路径,下拉式选择(dropdown)": "自动检测index路径,下拉式选择(dropdown)",
        "融合": "融合",
        "要改的模型信息": "要改的模型信息",
        "要置入的模型信息": "要置入的模型信息",
        "训练": "训练",
        "训练模型": "训练模型",
        "训练特征索引": "训练特征索引",
        "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log": "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log",
        "请指定说话人id": "请指定说话人id",
        "请选择index文件": "请选择index文件",
        "请选择pth文件": "请选择pth文件",
        "请选择说话人id": "请选择说话人id",
        "转换": "转换",
        "输入实验名": "输入实验名",
        "输入待处理音频文件夹路径": "输入待处理音频文件夹路径",
        "输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)": "输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)",
        "输入待处理音频文件路径(默认是正确格式示例)": "输入待处理音频文件路径(默认是正确格式示例)",
        "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络": "输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络",
        "输入监听": "输入监听",
        "输入训练文件夹路径": "输入训练文件夹路径",
        "输入设备": "输入设备",
        "输入降噪": "输入降噪",
        "输出信息": "输出信息",
        "输出变声": "输出变声",
        "输出设备": "输出设备",
        "输出降噪": "输出降噪",
        "输出音频(右下角三个点,点了可以下载)": "输出音频(右下角三个点,点了可以下载)",
        "选择.index文件": "选择.index文件",
        "选择.pth文件": "选择.pth文件",
        "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU": "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU",
        "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU": "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU",
        "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU": "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU",
        "采样长度": "采样长度",
        "重载设备列表": "重载设备列表",
        "音调设置": "音调设置",
        "音频设备(请使用同种类驱动)": "音频设备(请使用同种类驱动)",
        "音高算法": "音高算法",
        "额外推理时长": "额外推理时长"
        },
    'en_US': {
        'SUPER': 'zh_CN',
        'LANGUAGE': 'en_US',
        '初始化成功': 'Initialization successful',
        '就绪': 'Ready',
        '预处理-训练': 'Preprocessing-Training',
        '训练说明': 'Training instructions',
        '### 预处理参数设置': '### Preprocessing parameter settings',
        '模型名称': 'Model name',
        'f0提取器': 'f0 extractor',
        '预处理线程数': 'Preprocessing thread number',
        '### 训练参数设置': '### Training parameter settings',
        '学习率': 'Learning rate',
        '批大小': 'Batch size',
        '训练日志记录间隔（step）': 'Training log recording interval (step)',
        '验证集验证间隔（epoch）': 'Validation set validation interval (epoch)',
        '检查点保存间隔（epoch）': 'Checkpoint save interval (epoch)',
        '保留最新的检查点文件(0保存全部)': 'Keep the latest checkpoint file (0 save all)',
        '是否添加底模': 'Whether to add the base model',
        '### 开始训练': '### Start training',
        '打开数据集文件夹': 'Open the dataset folder',
        '启动Tensorboard': 'Start Tensorboard',
        '### 恢复训练': '### Resume training',
        '从检查点恢复训练进度': 'Restore training progress from checkpoint',
        '刷新': 'Refresh',
        '恢复训练': 'Resume training',
        "推理": "Inference",
        "推理说明": "Inference instructions",
        "### 推理参数设置": "### Inference parameter settings",
        "变调": "Pitch shift",
        "文件列表": "File list",
        "选择要导出的模型": "Select the model to export",
        "刷新模型和音色": "Refresh model and timbre",
        "导出模型": "Export model",
        "选择音色文件": "Select timbre file",
        "选择待转换音频": "Select audio to be converted",
        "开始转换": "Start conversion",
        "输出音频": "Output audio",
        "打开文件夹失败！": "Failed to open folder!",
        "开始预处理": "Start preprocessing",
        "开始训练": "Start training",
        "开始导出模型": "Start exporting model",
        "导出模型成功": "Model exported successfully",
        "出现错误：": "An error occurred:",
        "缺少模型文件": "Missing model file",
        '缺少文件': 'Missing file',
        "已清理残留文件": "Residual files cleaned up",
        "无需清理残留文件": "No need to clean up residual files",
        "开始推理": "Start inference",
        '### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)第一次编写|[@thestmitsuk](https://github.com/thestmitsuki)二次补完': '### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)first writing|[@thestmitsuk](https://github.com/thestmitsuki)second completion'
    }
}

class I18nAuto:
    def __init__(self, language=None):
        self.language_list = LANGUAGE_LIST
        self.language_all = LANGUAGE_ALL
        self.language_map = {}
        self.language = language or locale.getdefaultlocale()[0]
        if self.language not in self.language_list:
            self.language = 'zh_CN'
        self.read_language(self.language_all['zh_CN'])
        while self.language_all[self.language]['SUPER'] != 'END':
            self.read_language(self.language_all[self.language])
            self.language = self.language_all[self.language]['SUPER']

    def read_language(self, lang_dict: dict):
        self.language_map.update(lang_dict)

    def __call__(self, key):
        return self.language_map[key]



 

def get_file_options(directory, extension):
    return [file for file in os.listdir(directory) if file.endswith(extension)]


def split_wav(input_file, output_dir, duration=8):
    # 打开WAV文件
    with wave.open(input_file, 'rb') as wav_file:
        # 获取音频参数
        params = wav_file.getparams()
        num_channels = params.nchannels
        sample_width = params.sampwidth
        frame_rate = params.framerate
        num_frames = params.nframes

        # 计算每段音频的帧数
        segment_frames = int(duration * frame_rate)

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 读取音频数据
        frames = wav_file.readframes(num_frames)
        audio_data = np.frombuffer(frames, dtype=np.int16)

        # 切分音频数据
        num_segments = len(audio_data) // segment_frames + int(len(audio_data) % segment_frames > 0)
        for i in range(num_segments):
            segment_start = i * segment_frames
            segment_end = (i + 1) * segment_frames
            segment_data = audio_data[segment_start:segment_end]

            # 创建输出文件名
            basename_path = os.path.basename(input_file).rsplit('.', maxsplit=1)[0]
            segment_filename = basename_path + f"_{i+1}.wav"
            segment_filepath = os.path.join(output_dir, segment_filename)

            # 写入切分后的音频数据到输出文件
            with wave.open(segment_filepath, 'wb') as segment_wav:
                segment_wav.setparams(params)
                segment_wav.writeframes(segment_data.tobytes())

    print("音频切分完成！")


def concatenate_wav_files(input_files, output_file):
    merged_audio = None

    for file in input_files:
        audio = AudioSegment.from_wav(file)

        if merged_audio is None:
            merged_audio = audio
        else:
            # 将音频文件进行拼接
            merged_audio += audio

    # 导出合并后的音频文件
    merged_audio.export(output_file, format='wav')


# def merge_audio_files(file1, file2, output_file, decibel_reduction=5):
#     audio1 = AudioSegment.from_file(file1)
#     audio2 = AudioSegment.from_file(file2)

#     # 计算audio2降低指定分贝后的新音量
#     new_volume = audio2.dBFS - decibel_reduction
#     audio2 = audio2.apply_gain(new_volume - audio2.dBFS)

#     # 将两个音频文件进行叠加
#     merged_audio = audio1.overlay(audio2)

#     # 导出合并后的音频文件
#     merged_audio.export(output_file, format='wav')

from pydub import AudioSegment

def merge_audio_files(start_position_ms, file1, file2, output_file, decibel_reduction):
    
    # 从文件加载音频
    audio1 = AudioSegment.from_file(file1)
    audio2 = AudioSegment.from_file(file2)

    # 计算audio1增加指定分贝后的新音量
    new_volume = audio1.dBFS + decibel_reduction
    audio1 = audio1.apply_gain(new_volume - audio1.dBFS)

    # 将audio1插入到audio2的指定位置
    part_before = audio2[:start_position_ms]
    part_after = audio2[start_position_ms + len(audio1):]
    merged_audio = part_before + audio1.overlay(audio2[start_position_ms : start_position_ms + len(audio1)]) + part_after

    # 导出合并后的音频文件
    merged_audio.export(output_file, format='wav')

# 使用例子
# merge_audio_files("path_to_file1.wav", "path_to_file2.wav", "output.wav", start_position_ms=5000)



debug = True
local_model_root = 'weights'
# Author_ZH = {"专业女歌手":"专业女歌手","yuhui":"RD瑜晖","nifeng":"Paddle小倪", "Jay_Chou": "周杰伦", "ljr": "梁静茹", "dlj":"邓丽君", "luyao":"Paddle小鹿","yujun":"Paddle小军"}
# Author_EN = {"专业女歌手":"专业女歌手","RD瑜晖":"yuhui", "Paddle小倪":"nifeng", "Paddle小军": "yujun", "周杰伦" : "Jay_Chou" ,"梁静茹": "ljr", "邓丽君":"dlj", "Paddle小鹿":"luyao"}
# Author_Sing = {"专业女歌手":"宁夏-改歌词","RD瑜晖":"yuhui", "Paddle小倪":"nifeng", "Paddle小军":"翻唱-白月光与朱砂痣", "Paddle小鹿":"翻唱周杰伦-发如雪", "周杰伦" : "翻唱女生版-粉色海洋" ,"梁静茹": "翻唱张韶涵-亲爱的那不是爱情", "邓丽君":"翻唱日语版-我只在乎你"}
    # "🔥星辉女音-女歌手👩‍🎤": 0,
    # "🔥摇滚先锋-男歌手👨‍🎤": -10,
    # '🔥流行音王-男歌手👨‍🎤': -2, 
    # "🔥温柔情歌-女歌手👩‍🎤":  0,
    # "🔥经典天籁-女歌手👩‍🎤": -3,
Author_ZH = {"🔥星辉女音-女歌手👩‍🎤":"🔥星辉女音-女歌手👩‍🎤","yuhui":"🔥摇滚先锋-男歌手👨‍🎤",'zh':"🔥魅声少年-男歌手👨‍🎤", "ljr": "🔥温柔情歌-女歌手👩‍🎤", "luyao":"🔥宇宙少女-女歌手👩‍🎤", "mgr":"🔥银幕音魂-男歌手👨‍🎤"}
Author_EN = {"🔥星辉女音-女歌手👩‍🎤":"🔥星辉女音-女歌手👩‍🎤","🔥摇滚先锋-男歌手👨‍🎤":"yuhui", "🔥魅声少年-男歌手👨‍🎤":"zh", "🔥温柔情歌-女歌手👩‍🎤": "ljr", "🔥宇宙少女-女歌手👩‍🎤":"luyao", "🔥银幕音魂-男歌手👨‍🎤": "mgr"}
Author_Sing = {"🔥星辉女音-女歌手👩‍🎤":"星辉女音","🔥摇滚先锋-男歌手👨‍🎤":"摇滚先锋","🔥魅声少年-男歌手👨‍🎤":"魅声少年", "🔥温柔情歌-女歌手👩‍🎤": "温柔情歌", "🔥宇宙少女-女歌手👩‍🎤":"宇宙少女", "🔥银幕音魂-男歌手👨‍🎤":"银幕音魂"}

output_format = "wav"
# vc_transform = 0
cluster_ratio = 0.3
slice_db = -40
noise_scale = 0.4
pad_seconds = 0.5
cl_num = 0
lg_num = 0
lgr_num = 0.75
f0_predictor = "pm"
enhancer_adaptive_key = 0
cr_threshold = 0.05
k_step = 100
use_spk_mix = False
second_encoding = False
loudness_envelope_adjustment = 0

cuda = {}

spks = list([_name for _name in Author_ZH.values()])

ckpt_list = []
create_list = ["模板作曲", "自动作曲"]
songs_list = list([_name for _name in Songs_CN_dict.keys()])
style_list = SongsType.keys()
model = None
SINGERS = list([_name for _name in Singer_F_dict.keys()])

i18n = I18nAuto()


if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"
        

def modelAnalysis(device, sid, msg, choice_ckpt):
    global model, spks, Author_ZH, Author_EN

    extra_author = glob.glob("assets/weights/*.pth")
    for author in extra_author:
        filename = os.path.basename(author) 
        name, _ = os.path.splitext(filename)
        if name not in Author_ZH.keys():
            Author_ZH.update({name:name})
            Author_EN.update({name:name})
            Author_Sing.update({name:""})

    device = cuda[device] if "CUDA" in device else device

    device=device if device != "Auto" else None

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    print(device)
    if sid != "🔥星辉女音-女歌手👩‍🎤":
        vc.get_vc(choice_ckpt,0.33,0.33)

    model_path = os.path.join(local_model_root, Author_EN[sid], "logs", choice_ckpt)
    
    device_name = torch.cuda.get_device_properties(dev).name if "cuda" in str(dev) else str(dev)
    _name = "weights/"
    msg = f"成功加载模型到设备{device_name}上\n"
    msg += f"模型{model_path.split(_name)[1]}加载成功\n"
    msg += "当前模型的可用音色：\n"
    for i in SINGERS:
        msg += i + " "

    model = model_path

    return sid, msg



def vc_batch_fn(sid, input_audio, auto_f0, vc_transform, choice_ckpt):

    global model
        
    print(i18n('开始推理'))
    print(f"Start processing: {input_audio}")
    # import pdb
    # pdb.set_trace()
    index_file = glob.glob(f"logs/{Author_EN[sid]}/added*.index")[0]
    spk_item=0
    input_audio0=input_audio
    vc_transform0=vc_transform
    f0_file=None
    f0method0="rmvpe"
    file_index1=index_file
    file_index2=index_file
    index_rate1=0.75
    filter_radius0=3
    resample_sr0=0
    rms_mix_rate0=0.25
    protect0=0.33

    vc_output1, vc_output2 = vc.vc_single(spk_item,input_audio0, vc_transform0, f0_file, f0method0, file_index1, file_index2,index_rate1, filter_radius0,resample_sr0, rms_mix_rate0, protect0) 
    
    print(i18n("推理成功"))
    return "音乐合成成功啦，快来听听吧!", vc_output2

def refresh_options(sid, vc_transform, exp_sex):
    # import pdb
    # pdb.set_trace()
    global ckpt_list, Author_Sing, spks, Author_ZH

    if Author_Sing[sid]:
        audio_wav_file = Author_Sing[sid] + '.wav'
    else:
        audio_wav_file = None

    if sid not in Singer_F_dict.keys():
        if exp_sex=="男声":
            vc_transform = -10
        else:
            vc_transform = 0
    else:
        vc_transform = Singer_F_dict[sid]
        
    if sid == "🔥星辉女音-女歌手👩‍🎤":
        return gr.Dropdown.update(choices=spks,value=sid), gr.Dropdown.update(choices=[], value="默认"), gr.Audio.update(label=Author_Sing[sid], value=audio_wav_file, interactive=False), vc_transform, ""
    
    ckpt_list = [file for file in get_file_options(os.path.join(now_dir, "assets/weights"), f"{Author_EN[sid]}.pth")]
    print("ckpt_list: ",ckpt_list, sid, f"{ Author_EN[sid]}.pth")
    spks = list([_name for _name in Author_Sing.keys()])
    
    return gr.Dropdown(label="音色（歌手）", choices=spks,value=str(sid)), gr.Dropdown.update(choices=ckpt_list,value=ckpt_list[0]), gr.Audio.update(label=Author_Sing[sid], value=audio_wav_file, interactive=False), vc_transform, ""

sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    logger.debug("Write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    logger.info("Use gpus: %s", str(gpus16))
    if pretrained_G14 == "":
        logger.info("No pretrained Generator")
    if pretrained_D15 == "":
        logger.info("No pretrained Discriminator")
    if version19 == "v1" or sr2 == "40k":
        config_path = "v1/%s.json" % sr2
    else:
        config_path = "v2/%s.json" % sr2
    config_save_path = os.path.join(exp_dir, "config.json")
    if not pathlib.Path(config_save_path).exists():
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(
                config.json_config[config_path],
                f,
                ensure_ascii=False,
                indent=4,
                sort_keys=True,
            )
            f.write("\n")
    if gpus16:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    else:
        cmd = (
            '"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
            % (
                config.python_cmd,
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                "-pg %s" % pretrained_G14 if pretrained_G14 != "" else "",
                "-pd %s" % pretrained_D15 if pretrained_D15 != "" else "",
                1 if if_save_latest13 == i18n("是") else 0,
                1 if if_cache_gpu17 == i18n("是") else 0,
                1 if if_save_every_weights18 == i18n("是") else 0,
                version19,
            )
        )
    # logger.info(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    return "训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    # exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    exp_dir = "logs/%s" % (exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if not os.path.exists(feature_dir):
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    infos = []
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * config.n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except:
            info = traceback.format_exc()
            # logger.info(info)
            infos.append(info)
            yield "\n".join(infos)

    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )

    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key( 
    sid,
    exp_sex,
    exp_dir1,
    sr2,
    if_f0_3,
    record_audio_prompt,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    gpus_rmvpe,
):

    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)


    global Author_ZH, Author_EN, Author_Sing
    os.system(f"cp -r assets/weights/yuhui.pth {exp_dir1}.pth")
    os.system(f"cp -r logs/yuhui logs/{exp_dir1}")    
    yield get_info_str("快去点击在线体验试试你的声音吧！"), exp_dir1



#     if not exp_dir1:
#         yield get_info_str("请给声音模型取个名字吧～"), sid
#         return 
#     if not exp_sex:
#         yield get_info_str("请选择男声or女声～"), sid
#         return

#     # step1:处理数据
#     yield get_info_str(i18n("step1:正在处理数据")), exp_dir1
#     if record_audio_prompt:
#         trainset_dir4 = [record_audio_prompt]
#     [get_info_str(_) for _ in preprocess_dataset(trainset_dir4, exp_dir1, sr2, np7)]

#     # step2a:提取音高
#     yield get_info_str(i18n("step2:正在提取音高&正在提取特征")), exp_dir1
#     [
#         get_info_str(_)
#         for _ in extract_f0_feature(
#             gpus16, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe
#         )
#     ]

#     # step3a:训练模型
#     yield get_info_str(i18n("step3a:正在训练模型")), exp_dir1


#     future = training_threadpool.submit(click_train(
#         exp_dir1,
#         sr2,
#         if_f0_3,
#         spk_id5,
#         save_epoch10,
#         total_epoch11,
#         batch_size12,
#         if_save_latest13,
#         pretrained_G14,
#         pretrained_D15,
#         gpus16,
#         if_cache_gpu17,
#         if_save_every_weights18,
#         version19,
#     ) )

#     start_time = time.time()

#     while not future.done():
#         is_processing = future.running()
#         if  is_processing:
#             passed_time = int(time.time() - start_time)
#             # yield get_info_str("训练中, 预计需要180秒, 请耐心等待, 当前已等待{}秒...".format(passed_time)), exp_dir1
#         time.sleep(1)

#     # step3b:训练索引
#     [get_info_str(_) for _ in train_index(exp_dir1, version19)]
#     yield get_info_str(i18n("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log")), exp_dir1

#     os.system("mv logs/{}/add* ../".format(exp_dir1))
#     os.system("rm -rf logs/{}/* ".format(exp_dir1))
    # os.system("mv ../add* logs/{}/".format(exp_dir1))
    # os.system(f"ls logs/{exp_dir1}")
    

#     _Author_ZH = {exp_dir1 : exp_dir1}
#     _Author_EN = {exp_dir1 : exp_dir1}
#     _Author_Sing = {exp_dir1 : ""}

#     _Author_ZH.update(Author_ZH)
#     Author_ZH = _Author_ZH
#     _Author_EN.update(Author_EN)
#     Author_EN = _Author_EN
#     _Author_Sing.update(Author_Sing)
#     Author_Sing = _Author_Sing
#     print(Author_ZH, Author_EN, Author_Sing)
    

#     yield get_info_str("快去点击在线体验试试你的声音吧！"), exp_dir1


#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log")):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


F0GPUVisible = config.dml == False


def if_done(done, p):
    while 1:
        if p.poll() is None:
            sleep(0.5)
        else:
            break
    done[0] = True

def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() is None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    # import pdb
    # pdb.set_trace()
    if exp_dir:
        os.system("rm -rf %s/datasets/%s" % (now_dir, exp_dir))
        os.system("rm -rf %s/logs/%s" % (now_dir, exp_dir))
    os.makedirs("%s/datasets/%s" % (now_dir, exp_dir), exist_ok=True)

    paths = trainset_dir
    if not isinstance(trainset_dir[0], str):
        paths = [path.name for path in paths]
    cnt = 0
    for path in paths:
        shutil.move(path, "%s/datasets/%s" % (now_dir, exp_dir) )

    trainset_dir = "%s/datasets/%s" % (now_dir, exp_dir)

    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)

    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    per = 3.0 if config.is_half else 3.7
    cmd = '"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f' % (
        config.python_cmd,
        trainset_dir,
        sr,
        n_p,
        now_dir,
        exp_dir,
        config.noparallel,
        per,
    )
    # logger.info(cmd)
    # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    p = Popen(cmd, shell=True)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    # logger.info(log)
    yield log


def get_pretrained_models(path_str, f0_str, sr2):
    if_pretrained_generator_exist = os.access(
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if_pretrained_discriminator_exist = os.access(
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK
    )
    if not if_pretrained_generator_exist:
        logger.warning(
            "assets/pretrained%s/%sG%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    if not if_pretrained_discriminator_exist:
        logger.warning(
            "assets/pretrained%s/%sD%s.pth not exist, will not use pretrained model",
            path_str,
            f0_str,
            sr2,
        )
    return (
        "assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_generator_exist
        else "",
        "assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)
        if if_pretrained_discriminator_exist
        else "",
    )

def change_f0_method(f0method8):
    # if f0method8 == "rmvpe_gpu":
    #     visible = F0GPUVisible
    # else:
    visible = False
    return {"visible": visible, "__type__": "update"}

def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    return get_pretrained_models(path_str, f0_str, sr2)

def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    if sr2 == "32k" and version19 == "v1":
        sr2 = "40k"
    to_return_sr2 = (
        {"choices": ["40k", "48k"], "__type__": "update", "value": sr2}
        if version19 == "v1"
        else {"choices": ["40k", "48k", "32k"], "__type__": "update", "value": sr2}
    )
    f0_str = "f0" if if_f0_3 else ""
    return (
        *get_pretrained_models(path_str, f0_str, sr2),
        to_return_sr2,
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    return (
        {"visible": if_f0_3, "__type__": "update"},
        {"visible": if_f0_3, "__type__": "update"},
        *get_pretrained_models(path_str, "f0" if if_f0_3 == True else "", sr2),
    )



def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, gpus_rmvpe):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        if f0method != "rmvpe_gpu":
            cmd = (
                '"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'
                % (
                    config.python_cmd,
                    now_dir,
                    exp_dir,
                    n_p,
                    f0method,
                )
            )
            # logger.info(cmd)
            p = Popen(
                cmd, shell=True, cwd=now_dir
            )  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
            # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
            done = [False]
            threading.Thread(
                target=if_done,
                args=(
                    done,
                    p,
                ),
            ).start()
        else:
            if gpus_rmvpe != "-":
                gpus_rmvpe = gpus_rmvpe.split("-")
                leng = len(gpus_rmvpe)
                ps = []
                for idx, n_g in enumerate(gpus_rmvpe):
                    cmd = (
                        '"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '
                        % (
                            config.python_cmd,
                            leng,
                            idx,
                            n_g,
                            now_dir,
                            exp_dir,
                            config.is_half,
                        )
                    )
                    # logger.info(cmd)
                    p = Popen(
                        cmd, shell=True, cwd=now_dir
                    )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                    ps.append(p)
                # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
                done = [False]
                threading.Thread(
                    target=if_done_multi,  #
                    args=(
                        done,
                        ps,
                    ),
                ).start()
            else:
                cmd = (
                    config.python_cmd
                    + ' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '
                    % (
                        now_dir,
                        exp_dir,
                    )
                )
                # logger.info(cmd)
                p = Popen(
                    cmd, shell=True, cwd=now_dir
                )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
                p.wait()
                done = [True]
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0]:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        # logger.info(log)
        yield log
    # 对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            '"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'
            % (
                config.python_cmd,
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        # logger.info(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    # 煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0]:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    # logger.info(log)
    yield log

def cpop_pinyin2ph_func():
    # In the README file of opencpop dataset, they defined a "pinyin to phoneme mapping table"
    pinyin2phs = {'AP': 'AP', 'SP': 'SP'}
    with open('cpop_pinyin2ph.txt') as rf:
        for line in rf.readlines():
            elements = [x.strip() for x in line.split('|') if x.strip() != '']
            pinyin2phs[elements[0]] = elements[1]
    return pinyin2phs

pinyin2phs = cpop_pinyin2ph_func()

def convert_note_midi_dur(text, old_note, old_midi_dur):
    pinyins = lazy_pinyin(text, strict=False)
    ph_per_word_lst = [pinyin2phs[pinyin.strip()]
                        for pinyin in pinyins if pinyin.strip() in pinyin2phs]

    # Note
    note_per_word_lst = [x.strip()
                            for x in old_note.split('|') if x.strip() != '']
    mididur_per_word_lst = [
        x.strip() for x in old_midi_dur.split('|') if x.strip() != '']

    note_lst = []
    ph_lst = []
    midi_dur_lst = []
    is_slur = []
    for idx, ph_per_word in enumerate(ph_per_word_lst):
        # for phs in one word:
        # single ph like ['ai']  or multiple phs like ['n', 'i']
        ph_in_this_word = ph_per_word.split()

        # for notes in one word:
        # single note like ['D4'] or multiple notes like ['D4', 'E4'] which means a 'slur' here.
        note_in_this_word = note_per_word_lst[idx].split()
        midi_dur_in_this_word = mididur_per_word_lst[idx].split()
        # process for the model input
        # Step 1.
        #  Deal with note of 'not slur' case or the first note of 'slur' case
        #  j        ie
        #  F#4/Gb4  F#4/Gb4
        #  0        0
        # if ph_in_this_word[0] == "SP":
        #     print(midi_dur_in_this_word[0])
        
        for ph in ph_in_this_word:
            # if ph == 'SP' and float(midi_dur_in_this_word[0]) < 0.2:
            #     ph = "AP"
            ph_lst.append(ph)
            note_lst.append(note_in_this_word[0])
            midi_dur_lst.append(midi_dur_in_this_word[0])
            is_slur.append('0')
        # step 2.
        #  Deal with the 2nd, 3rd... notes of 'slur' case
        #  j        ie         ie
        #  F#4/Gb4  F#4/Gb4    C#4/Db4
        #  0        0          1
        # is_slur = True, we should repeat the YUNMU to match the 2nd, 3rd... notes.
        if len(note_in_this_word) > 1:
            for idx in range(1, len(note_in_this_word)):
                # if ph_in_this_word == ['n', 'i']:
                #     import pdb
                #     pdb.set_trace()
                # print(ph_in_this_word, note_in_this_word)
                if len(ph_in_this_word)>idx and len(midi_dur_in_this_word)>idx and 'i' in ph_in_this_word[idx] and float(midi_dur_in_this_word[idx]) - float(midi_dur_in_this_word[idx-1]) > 0.2:
                    note_lst[-1]= note_in_this_word[idx]
                

                note_lst.append(note_in_this_word[idx])
                ph_lst.append(ph_in_this_word[-1])   
                midi_dur_lst.append(midi_dur_in_this_word[idx])
                is_slur.append('1')
            
    # print(ph_lst)
    # print(note_lst)
    # print(midi_dur_lst)
    # print(is_slur)

    if len(ph_lst) == len(note_lst) == len(midi_dur_lst):
        print("sucess convert note and midi_dur")
    else:
        print("输入有误，分割后的音符长度：{}, 音素长度: {}, 时长长度: {}".format(len(note_lst), len(ph_lst), len(midi_dur_lst)))
        return None

    return ' '.join(ph_lst), ' '.join(note_lst), ' '.join(midi_dur_lst), ' '.join(is_slur)
        

class GradioInfer:
    def __init__(self, exp_name, inference_cls, title, description, article, example_inputs):
        self.exp_name = exp_name
        self.title = title
        self.description = description
        self.article = article
        self.example_inputs = example_inputs
        inference_cls="infer_file.Infer_Fun"
        pkg = ".".join(inference_cls.split(".")[:-1])
        cls_name = inference_cls.split(".")[-1]
        # import pdb
        # pdb.set_trace()
        self.inference_cls = getattr(importlib.import_module(pkg), cls_name)
        
    def clear(self, name, text, notes, notes_duration):
        return '','','','',''

    def select_sid(self, name):
        text = lyrics_dict[Songs_CN_dict[name]]
        return name, text, gr.Dropdown.update(choices=songs_list,value=name)

    def select_style(self, choice_style):
        return gr.Dropdown.update(choices=style_list,value=choice_style)
    
    def select_format(self, create_songs):
        if create_songs==create_list[0]:
            return gr.Dropdown.update(choices=create_list,value=create_list[0]), gr.Dropdown.update(choices=songs_list,value="", visible=True), gr.Dropdown.update(choices=style_list,value="", visible=False), ''
        else:
            return gr.Dropdown.update(choices=create_list,value=create_list[1]), gr.Dropdown.update(choices=songs_list,value="", visible=False), gr.Dropdown.update(choices=style_list, value="", visible=True), ''
    
    def generate_lyrics(self, create_songs, choice_style, name, sid_name, text):
        try:
            if not name:
                return None, None, text, "请输入主题！"

            if create_songs==create_list[1]:
                if not choice_style:
                    return None, None, text, "请进行风格选择"
                text, notes, notes_duration = lyrics_to_melody(name, choice_style)
                
            else:
                notes, notes_duration, text   = gen_songs_needs(name, Songs_CN_dict[sid_name])
            
                len_notes = len(notes.split('|'))
                len_notes_duration = len(notes_duration.split('|'))
                len_text = len(text) - text.count("SP")
                
                print(len_notes, len_notes_duration, len_text)
                
                if len_text!=len_notes_duration:
                    return notes, notes_duration, text, "歌词长度为{}个单词，需要保持{}个单词，请点击【使用主题】重新生成!".format(len_text,len_notes_duration)
                
            text = text.replace("SP", "\n")
            
            return notes, notes_duration, text, "创作成功，快点击歌声合成吧～"
        except:
            return '', '', text, "不好意思，我尝试失败了，请点击【使用主题】重新然我重新试试～"
    
    def greet(self, create_songs, name, text, notes, notes_duration, sid, auto_f0, vc_transform, choice_ckpt, sid_name, Ins):
        # import pdb
        # pdb.set_trace()

        if not text:
            return None, "请输入或创作歌词"

        if not notes:
            notes = notes_dict[Songs_CN_dict[sid_name]]
            notes_duration = notes_duration_dict[Songs_CN_dict[sid_name]]
                
        len_notes = len(notes.split('|'))
        len_notes_duration = len(notes_duration.split('|'))
        if text[0] == "\n":
            text = text[1:]
        if text[-1]=="\n" and create_songs==create_list[0]:
            text = text[:-1]
        text = text.replace("\n", "SP").replace(" ", "")
        len_text = len(text) - text.count("SP")
        print(len_notes, len_notes_duration, len_text)

        if len_text!=len_notes_duration:
            return None, "歌词长度为{}个单词，请保持{}个单词!".format(len_text,len_notes_duration)

        ph_seq, notes, notes_duration, notes_slur = convert_note_midi_dur(text, notes, notes_duration)

        PUNCS = '。？；：'
        sents = re.split(rf'([{PUNCS}])', text.replace('\n', ','))
        sents_notes = re.split(rf'([{PUNCS}])', notes.replace('\n', ','))
        sents_notes_dur = re.split(rf'([{PUNCS}])', notes_duration.replace('\n', ','))
        sents_ph_seq = re.split(rf'([{PUNCS}])', ph_seq.replace('\n', ','))
        sents_notes_slur = re.split(rf'([{PUNCS}])', notes_slur.replace('\n', ','))
        
        if sents[-1] not in list(PUNCS):
            sents = sents + ['']
            sents_notes = sents_notes + ['']
            sents_notes_dur = sents_notes_dur + ['']
            sents_ph_seq = sents_ph_seq + ['']
            sents_notes_slur = sents_notes_slur + ['']

        audio_outs = []
        s, n, n_dur, ph, n_slur = "", "", "", "",  ""
        for i in range(0, len(sents), 2):
            if len(sents[i]) > 0:
                s += sents[i] + sents[i + 1]
                n += sents_notes[i] + sents_notes[i+1]
                n_dur += sents_notes_dur[i] + sents_notes_dur[i+1]
                ph += sents_ph_seq[i] + sents_ph_seq[i+1]
                n_slur += sents_notes_slur[i] + sents_notes_slur[i+1]
            if len(s) >= 400 or (i >= len(sents) - 2 and len(s) > 0):
                audio_out = self.infer_ins.infer_once({
                    # 'text': s,
                    # 'notes': n,
                    # 'notes_duration': n_dur,
                    'text': s,
                    'ph_seq': ph,
                    'note_seq': n,
                    'note_dur_seq': n_dur,
                    'is_slur_seq': n_slur,
                    'input_type' : 'phoneme'
                })
                audio_out = audio_out * 32767
                audio_out = audio_out.astype(np.int16)
                audio_outs.append(audio_out)
                audio_outs.append(np.zeros(int(hp['audio_sample_rate'] * 0.3)).astype(np.int16))
                s = ""
                n = ""
        audio_outs = np.concatenate(audio_outs)
        sf.write('temp.wav', audio_outs, hp['audio_sample_rate'])
        output_file_path = "temp.wav"

        if sid != "🔥星辉女音-女歌手👩‍🎤":
            with torch.no_grad():
                _,  output_file_path = vc_batch_fn(sid, 'temp.wav', auto_f0, vc_transform, choice_ckpt)
        
        if Ins == "是":
            if isinstance(output_file_path, tuple):
                sf.write('temp.wav', output_file_path[1], output_file_path[0])
                output_file_path = "temp.wav"

            merge_audio_files(Songs_ST_dict[sid_name], output_file_path, os.path.join("instrument",f"{Songs_CN_dict[sid_name]}"+".wav"), output_file_path, Songs_V_dict[sid_name])

        return output_file_path, "合成成功，快去听听吧～"
    
    def run(self):

        set_hparams(os.path.join(script_path, "model/config.yaml"))
        infer_cls = self.inference_cls
        self.infer_ins: Infer_Fun = infer_cls(hp)
        example_inputs = self.example_inputs
        for i in range(len(example_inputs)):
            text, notes, notes_dur = example_inputs[i].split('<sep>')
            example_inputs[i] = ['', text, notes, notes_dur]
        iface = gr.Blocks()

        with iface:
            
            gr.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>💕🎤 {self.title}</h1>")
            gr.Markdown("## <center>⚡ 只需3分钟训练，快速复刻您喜欢的声音；在线体验，让你沉浸其中。</center>")
            
            gr.Markdown(value="""
                ### 注意‼️：本应用仅供个人娱乐和非商业用途，禁止用于血腥、暴力、性相关、政治相关内容

                联系制作者：【飞桨&百度研究院大数据实验室】（陆瑶、王硕、边江）
                """)
            gr.Markdown(value=self.description)
            with gr.Tab("🎶 - 在线体验"):
                with gr.Group(): 
                    gr.Markdown(value="""
                        <font size=3> 🎙️Step 1: 音色模型选择</font>
                        """)
                    with gr.Column():

                        with gr.Row():
                            sid = gr.Dropdown(label="音色（歌手）",choices = spks,value=spks[0])
                            exp_sex = gr.Radio(label="选择男声🚹/女声🚺", choices=["男声", "女声"], value="",visible=False)
                            choice_ckpt = gr.Dropdown(label="模型选择", choices=ckpt_list, value="默认")
                            model_load_button = gr.Button(value="加载模型(⚠️点我点我)", variant="primary")
                        with gr.Row():
                            example_output = gr.Audio(label="音色预览", interactive=False, value="星辉女音.wav")
                            sid_output = gr.Textbox(label="Output Message")
                        auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）", value=False, visible=False)
                        
                with gr.Group(): 
                    gr.Markdown(value="""
                        <font size=3> 🎼Step 2: 歌曲创作</font>
                        """) 
                    with gr.Column():
                        
                        with gr.Row():
                            # with gr.Row():                        
                            create_songs = gr.Dropdown(label="创作形式", choices=create_list, value=create_list[0])
                            name = gr.Textbox(lines=2, placeholder=None, label="输入你想创作的主题，例如：回家，我们会进行AI创作歌词哦～")
                            
                            choice_songs = gr.Dropdown(label="模板选择", choices=songs_list, visible=True)
                            choice_style = gr.Dropdown(label="风格选择", choices=style_list, visible=False)
                            submit1_button = gr.Button(value="歌曲创作(⚠️点我点我)", variant="primary")    
                            
                        with gr.Row():
                            text = gr.Textbox(lines=2, placeholder=None, value="", label="创作歌曲预览")

                        sid_name = gr.Dropdown(label="歌曲",choices = list(Songs_CN_dict.keys()),value="", visible=False)
                        notes = gr.Textbox(lines=2, placeholder=None, value="", label="input note", visible=False)
                        notes_duration = gr.Textbox(lines=2, placeholder=None, value="", label="input duration", visible=False)
                        clear_button = gr.Button(value="清除信息", visible=False)
                with gr.Group(): 
                    gr.Markdown(value="""
                        <font size=3> 📓Step 3: 其他设置选项</font>
                        """) 
                    with gr.Column(): 

                        with gr.Row():
                            Ins = gr.Radio(label="是否加入伴奏", choices=["是", "否"], value="否", interactive=True)
                            vc_transform = gr.Number(label="音调调整", value=0)

                with gr.Group(): 
                    gr.Markdown(value="""
                        <font size=3> 🧑‍🎤Step 4: 歌声合成</font>
                        """) 
                    with gr.Column(): 

                        with gr.Row():
                            submit2_button = gr.Button(value="歌声合成(⚠️点我点我)", variant="primary")
                            output_audio = gr.Audio(label="Output Audio", interactive=False)

                choice_songs.select(self.select_sid, [choice_songs], [sid_name, text, choice_songs])
                create_songs.select(self.select_format, [create_songs], [create_songs, choice_songs, choice_style, text]).then(self.clear, [name, text, notes, notes_duration], [name, text, notes, notes_duration])
                choice_style.select(self.select_style, [choice_style])

                device = gr.Dropdown(label="推理设备, 默认为自动选择CPU和GPU", choices=["Auto",*cuda.keys(),"cpu"], value="Auto", visible=False)
                
                sid.select(refresh_options,[sid, vc_transform, exp_sex],[sid, choice_ckpt, example_output, vc_transform, sid_output])
                
                model_load_button.click(modelAnalysis,[device, sid, sid_output, choice_ckpt],[sid,sid_output])

                submit1_button.click(self.generate_lyrics, [create_songs, choice_style, name, sid_name, text], [notes, notes_duration, text, sid_output])
                
                submit2_button.click(self.greet, [create_songs, name, text, notes, notes_duration,sid, auto_f0, vc_transform, choice_ckpt, sid_name, Ins], [output_audio, sid_output])
                
                clear_button.click(self.clear, [name, text, notes, notes_duration], [name, text, notes, notes_duration])
        

            with gr.Tab("🔊 - 定制声音"):     
                gr.Markdown(
                    value="step1: 填写实验配置, 需手工输入模型名字, 例如(xiaoming). "
                )
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            exp_dir1 = gr.Textbox(label="给声音模型取个名字吧", value="")
                            exp_sex = gr.Radio(label="选择男声🚹/女声🚺", choices=["男声", "女声"], value="")
                            sr2 = gr.Radio(
                                label=i18n("目标采样率"),
                                choices=["40k", "48k"],
                                value="40k",
                                interactive=True,
                                visible=False,
                            )
                            if_f0_3 = gr.Radio(
                                label=i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                                choices=[True, False],
                                value=True,
                                interactive=True,
                                visible=False,
                            )
                            version19 = gr.Radio(
                                label=i18n("版本"),
                                choices=["v1", "v2"],
                                value="v2",
                                interactive=True,
                                visible=False,
                            )
                            np7 = gr.Slider(
                                minimum=0,
                                maximum=config.n_cpu,
                                step=1,
                                label=i18n("提取音高和处理数据使用的CPU进程数"),
                                value=int(np.ceil(config.n_cpu / 1.5)),
                                interactive=True,
                                visible=False,
                            )

                        gr.Markdown(
                            value="step2: ⏺️请录制或上传干声数据，支持批量音频文件的上传，优先处理麦克风声音. "
                        )
                        with gr.Row():
                            # with gr.Row():
                                
                            #     trainset_dir4 = gr.File(
                            #         label="上传待训练声音文件(请上传干声), 可批量输入音频文件", file_count="multiple"
                            #     )
                            with gr.Row():
                                with gr.Column():
                                    record_audio_prompt = gr.Audio(label='请在此用麦克风录制您喜欢的声音', source='microphone', interactive=True, type="filepath", autoplay="True")
                                    trainset_dir4 = gr.File(
                                    label="上传待训练声音文件(请上传干声), 可批量输入音频文件", file_count="multiple"
                                )  

                                spk_id5 = gr.Slider(
                                    minimum=0,
                                    maximum=4,
                                    step=1,
                                    label=i18n("请指定说话人id"),
                                    value=0,
                                    interactive=True,
                                    visible=False,
                                )
                                but1 = gr.Button(i18n("处理数据"), variant="primary", visible=False,)
                                info1 = gr.Textbox(label=i18n("输出信息"), value="", visible=False,)

                                
                        with gr.Row():
                            with gr.Column():
                                gpus6 = gr.Textbox(
                                    label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                                    value=gpus,
                                    interactive=True,
                                    visible=False,
                                )
                                gpu_info9 = gr.Textbox(
                                    label=i18n("显卡信息"), value=gpu_info, visible=False,
                                )
                            with gr.Column():
                                f0method8 = gr.Radio(
                                    label=i18n(
                                        "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU"
                                    ),
                                    choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                                    value="rmvpe_gpu",
                                    interactive=True,
                                    visible=False,
                                )
                                gpus_rmvpe = gr.Textbox(
                                    label=i18n(
                                        "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                                    ),
                                    value="%s-%s" % (gpus, gpus),
                                    interactive=True,
                                    visible=False,
                                )
                            but2 = gr.Button(i18n("特征提取"), variant="primary",visible=False,)
                            info2 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8, visible=False,)



                        with gr.Row():
                            save_epoch10 = gr.Slider(
                                minimum=1,
                                maximum=50,
                                step=1,
                                label=i18n("保存频率save_every_epoch"),
                                value=5,
                                interactive=True,
                                visible=False,
                            )
                            total_epoch11 = gr.Slider(
                                minimum=2,
                                maximum=1000,
                                step=1,
                                label=i18n("总训练轮数total_epoch"),
                                value=20,
                                interactive=True,
                                visible=False,
                            )
                            batch_size12 = gr.Slider(
                                minimum=1,
                                maximum=40,
                                step=1,
                                label=i18n("每张显卡的batch_size"),
                                value=default_batch_size,
                                interactive=True,
                                visible=False,
                            )
                            if_save_latest13 = gr.Radio(
                                label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                                choices=[i18n("是"), i18n("否")],
                                value=i18n("否"),
                                interactive=True,
                                visible=False,
                            )
                            if_cache_gpu17 = gr.Radio(
                                label=i18n(
                                    "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                                ),
                                choices=[i18n("是"), i18n("否")],
                                value=i18n("否"),
                                interactive=True,
                                visible=False,
                            )
                            if_save_every_weights18 = gr.Radio(
                                label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                                choices=[i18n("是"), i18n("否")],
                                value=i18n("否"),
                                interactive=True,
                                visible=False,
                            )
                        with gr.Row():
                            pretrained_G14 = gr.Textbox(
                                label=i18n("加载预训练底模G路径"),
                                value="assets/pretrained_v2/f0G40k.pth",
                                interactive=True,
                                visible=False,
                            )
                            pretrained_D15 = gr.Textbox(
                                label=i18n("加载预训练底模D路径"),
                                value="assets/pretrained_v2/f0D40k.pth",
                                interactive=True,
                                visible=False,
                                
                            )

                            gpus16 = gr.Textbox(
                                label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                                value=gpus.split("-")[0],
                                interactive=True,
                                visible=False,
                            )
                            but3 = gr.Button(i18n("训练模型"), variant="primary", visible=False,)
                            but4 = gr.Button(i18n("训练特征索引"), variant="primary", visible=False,)
                            but5 = gr.Button("一键训练(⚠️点我点我)", variant="primary")

                    with gr.Column():
                        info3 = gr.Textbox(label="‼️ 输出信息，请等待日志出现“全流程结束！”", value="", max_lines=10)
                        train_text = gr.Textbox(lines=22, placeholder=True, value=TRAIN_DATA, label="请阅读并录音下面这段话", interactive=False)
                    


                f0method8.change(
                    fn=change_f0_method,
                    inputs=[f0method8],
                    outputs=[gpus_rmvpe],
                )

                sr2.change(
                    change_sr2,
                    [sr2, if_f0_3, version19],
                    [pretrained_G14, pretrained_D15],
                )
                version19.change(
                    change_version19,
                    [sr2, if_f0_3, version19],
                    [pretrained_G14, pretrained_D15, sr2],
                )
                if_f0_3.change(
                    change_f0,
                    [if_f0_3, sr2, version19],
                    [f0method8, gpus_rmvpe, pretrained_G14, pretrained_D15],
                )

                but5.click(
                    train1key,
                    [   sid,
                        exp_sex,
                        exp_dir1,
                        sr2,
                        if_f0_3,
                        record_audio_prompt,
                        trainset_dir4,
                        spk_id5,
                        np7,
                        f0method8,
                        save_epoch10,
                        total_epoch11,
                        batch_size12,
                        if_save_latest13,
                        pretrained_G14,
                        pretrained_D15,
                        gpus16,
                        if_cache_gpu17,
                        if_save_every_weights18,
                        version19,
                        gpus_rmvpe,
                    ],
                    [info3, sid],
                    api_name="train_start_all",
                ).then(refresh_options,[sid, vc_transform, exp_sex],[sid, choice_ckpt, example_output, vc_transform, sid_output]).then(modelAnalysis,[device, sid, sid_output, choice_ckpt],[sid,sid_output])

        # iface.launch(enable_queue=True)
        iface.queue(concurrency_count=511, max_size=1022).launch(share=True)


if __name__ == '__main__':
    gradio_config = yaml.safe_load(open('settings-1.yaml'))
    g = GradioInfer(**gradio_config)
    g.run()