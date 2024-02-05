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
import gradio as gr
import yaml
from gradio.inputs import Textbox

from infer import Infer
from utils.hparams import set_hparams
from utils.hparams import hparams as hp
import numpy as np
import soundfile as sf
import importlib
import re

import shutil
import shlex
import locale
import random


import random
import aistudio
import yaml
from gradio.inputs import Textbox

from infer import Infer
from utils.hparams import set_hparams
from utils.hparams import hparams as hp
import numpy as np
import soundfile as sf

import glob

import subprocess
import sys
import time
import traceback


import librosa
import soundfile
import torch
import wave
from pydub import AudioSegment

os.environ["AISTUDIO_LOG"] = "debug"



llm = ErnieBotChat(ernie_client_id='96BMhQM5simx6R97yDl483Zm', 
                ernie_client_secret='9e05mDOjHoyXD7Sb9GA1l420uaZ6vGMo', 
                model_name='ERNIE-Bot-4',
                top_p=0.4)

Songs_CN_dict = {
    "稻香":"DAOXIANG",
    "宁夏":"NINGXIA",
    "生日快乐":"HAPPYBIRTHDAY"
}

Ins_CN_dict = {
    "稻香":"宁夏伴奏.wav",
    "宁夏":"宁夏伴奏.wav",
    "生日快乐":"宁夏伴奏.wav"
}

lyrics_dict = {
    "DAOXIANG": "\n对这个世界如果你有太多的抱怨\n跌倒了就不敢继续往前走\n为什么人要这么的脆弱堕落\n请你打开电视看看\n多少人为生命在努力勇敢的走下去我们是不是该知足\n珍惜一切就算没有拥有耶\n还记得你说家是唯一的城堡\n随着稻香河流继续奔跑\n微微笑小时候的梦我知道\n不要哭让萤火虫带着你逃跑\n乡间的歌谣永远的依靠\n回家吧回到最初的美好\n",
    "NINGXIA": "\n宁静的夏天\n天空中繁星点点\n心里头有些思念\n思念着你的脸\n我可以假装看不见\n也可以偷偷地想念\n直到让我摸到你那温暖的脸\n",
    "HAPPYBIRTHDAY": "\n对所有的烦恼说拜拜\n对所有的快乐说嗨嗨\n亲爱的亲爱的生日快乐\n每一天都精彩\n看幸福的花儿为你盛开\n听美妙的音乐为你喝彩\n亲爱的亲爱的生日快乐\n祝你幸福永远\n幸福永远"
}
 
notes_dict = {
    "DAOXIANG": "C4 | C4 | A3 | C4 | A3 | C4 | C4 | C4 | D4 | D4 | D4 | D4 | E4 | C4 | rest | C4 | A3 | C4 | C4 | C4 "
                "| D4 | D4 | D4 | D4 | E4 | C4 | rest | C4 | A3 | C4 | C4 | C4 | D4 | D4 | D4 | D4 | C4 | E4 | E4 | "
                "rest | C4 | C4 | C4 | A3 | C4 | C4 | C4 | A3 | rest | C4 | A3 | C4 | C4 | C4 | D4 | D4 | D4 | D4 | "
                "E4 | C4 | C4 | C4 | A3 | C4 | C4 | C4 | D4 | D4 | D4 | D4 | E4 | C4 | rest | C4 | F4 | E4 | D4 | D4 "
                "| C4 | D4 | C4 | D4 | C4 | A3 | rest | E4 | G4 | G4 | G4 | G4 | G4 | G4 | G4 | G4 | G4 | E4 | D4 | "
                "rest | C4 | E4 | E4 | E4 | E4 | E4 | E4 | E4 | E4 | C4 | rest | A3 | C4 | C4 | C4 | C4 | D4 | D4 | "
                "D4 | C4 | E4 | E4 | rest | E4 | G4 | G4 | G4 | G4 | G4 | G4 | G4 | G4 | G4 | E4 | D4 | rest | C4 | "
                "E4 | E4 | E4 | E4 | E4 | E4 | E4 | E4 | C4 | rest | A3 | C4 | C4 | C4 | C4 | D4 | D4 | D4 | C4 | C4",
    "NINGXIA": "E4 | G3 | A3 | C4 | C4 | rest | E4 | E4 | E4 | E4 | D4 | C4 | C4 | rest | G3 | G3 | E3 | G3 | A3 | C4 "
               "| C4 | rest | A3 | A3 | C4 | A3 | G3 | G4 | rest | E4 | E4 | D4 | C4 | D4 | E4 | A3/G3 | G3 | rest | "
               "G3 | G3 | G3 | A3 | C4 | C4 | A3 | D4 | rest | G3 | A3 | C4 | D4 | E4 | G4 | G4 | A4 | E4 | D4 | "
               "D4/A3 | C4",
    "HAPPYBIRTHDAY": "B3 | G4 | E4 | F4 | G4 | E4 | F4 | B3 | B3 | rest | G3 | E4 | C4 | D4 | E4 | C4 | D4 | B4 | B4 "
                     "| rest | A4 | A4 | G4 | F4 | G4 | A4 | G4 | F4 | E4 | C4 | rest | E4 | E4 | C4 | E4 | F4 G4 | "
                     "F4 | rest | B3 | G4 | E4 | F4 | G4 | E4 | E4 | F4 | B3 | B3 | rest | G3 | E4 | C4 | D4 | E4 | "
                     "C4 | C4 | D4 | B4 C5 | B4 | rest | A4 | A4 | G4 | F4 | G4 | A4 | G4 | F4 | E4 | C4 | rest | E4 "
                     "| E4 | C4 | E4 | F4 G4 | F4 | rest | D4 | E4 | F4 | E4 "
}
 
notes_duration_dict = {
     "DAOXIANG": "0.125 | 0.125 | 0.125 | 0.25 | 0.125 | 0.375 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.25 "
                "| 0.25 | 0.125 | 0.25 | 0.125 | 0.375 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.25 | 0.25 "
                "| 0.125 | 0.25 | 0.125 | 0.375 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.25 | 0.25 | 0.5 "
                "| 0.375 | 0.25 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.25 | 0.125 | "
                "0.375 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.25 | 0.125 | 0.125 | 0.25 | 0.125 | 0.375 "
                "| 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.125 | 0.25 | 0.25 | 0.375 | 0.125 | 0.375 | 0.25 | 0.125 "
                "| 0.125 | 0.125 | 0.125 | 0.125 | 0.5 | 0.75 | 0.125 | 0.5 | 0.25 | 0.25 | 0.25 | 0.125 | "
                "0.25 | 0.25 | 0.25 | 0.125 | 0.125 | 0.125 | 0.25 | 0.25 | 0.125 | 0.25 | 0.125 | 0.25 | 0.25 | 0.25 "
                "| 0.125 | 0.125 | 0.125 | 0.25 | 0.25 | 0.125 | 0.25 | 0.125 | 0.375 | 0.125 | 0.125 | 0.25 | 0.125 "
                "| 0.125 | 0.25 | 0.25 | 1 | 0.5 | 0.25 | 0.25 | 0.25 | 0.125 | 0.25 | 0.25 | 0.25 | 0.125 | 0.125 | "
                "0.125 | 0.25 | 0.25 | 0.125 | 0.25 | 0.125 | 0.25 | 0.125 | 0.25 | 0.125 | 0.125 | 0.125 | 0.25 | "
                "0.25 | 0.125 | 0.25 | 0.125 | 0.375 | 0.125 | 0.125 | 0.25 | 0.125 | 0.375 | 0.25 | 1",
    "NINGXIA": "0.68 | 0.68 | 0.34 | 0.34 | 0.68 | 0.05 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.68 | 0.05 | "
               "0.17 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.68 | 0.05 | 0.17 | 0.34 | 0.17 | 0.34 | 0.34 | 0.68 | "
               "0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.68 | 0.34 | 0.17 | 0.17 | 0.17 | 0.34 | "
               "0.34 | 0.34 | 0.34 | 0.68 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 0.34 | 1.2 "
               "| 0.68 | 1.8",
    "HAPPYBIRTHDAY": "0.25 | 0.5 | 0.25 | 0.125 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.25 | 0.25 | 0.5 | 0.25 | 0.125 | 0.5 | 0.5 "
                     "| 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.25 | 0.125 | 0.5 | 0.25 | 0.125 | 0.5 | 0.5 | 0.5 | 0.5 | 0.125 | 0.5 "
                     "| 0.25 | 0.125 | 0.5 | 0.25 0.25 | 1 | 0.375 | 0.25 | 0.5 | 0.25 | 0.125 | 0.5 | 0.25 | 0.25 | 0.5 | "
                     "0.5 | 0.5 | 0.125 | 0.25 | 0.5 | 0.25 | 0.125 | 0.5 | 0.25 | 0.25 | 0.5 | 0.25 0.25 | 0.5 | 0.125 | 0.5 "
                     "| 0.25 | 0.125 | 0.5 | 0.25 | 0.125 | 0.5 | 0.5 | 0.5 | 0.5 | 0.125 | 0.5 | 0.25 | 0.25 | 0.5 | 0.25 "
                     "0.25 | 0.75 | 0.125 | 0.25 | 0.5 | 0.5 | 2 "
}
 
lyrics_slot_dict = {
    "DAOXIANG": "\n[对这个世界][如果][你有太多的抱怨]\n[跌倒了]就[不敢][继续往前走]\n[为什么][人要这么的脆弱堕落]\n[请你][打开电视][看看]\n[多少人][为生命在努力][勇敢的走下去]["
                "我们是不是该知足]\n[珍惜一切][就算没有拥有耶]\n[还记得][你说家是唯一的城堡]\n[随着稻香河流][继续奔跑]\n[微微笑][小时候的梦][我知道]\n[不要哭][让萤火虫带着你逃跑]\n["
                "乡间的歌谣][永远的依靠]\n[回家吧][回到最初的美好]\n",
    "NINGXIA": "\n[宁静的][夏天]\n[天空中][繁星点点]\n[心里头][有些][思念]\n[思念着][你的][脸]\n[我可以][假装][看不见]\n[也可以][偷偷地][想念]\n[直到][让我][摸到]["
               "你那][温暖的][脸]\n",
    "HAPPYBIRTHDAY": "\n[对所有的][烦恼][说拜拜]\n[对所有的][快乐][说嗨嗨]\n[亲爱的][亲爱的][生日快乐]\n[每一天][都精彩]\n[看幸福的][花儿][为你][盛开]\n[听美妙的]["
                     "音乐][为你喝彩]\n[亲爱的][亲爱的][生日快乐]\n[祝你][幸福][永远]\n[幸福][永远]\n "
}
 
# prompt template
GEN_LYRICS_MAIN = """
以"{theme}"为主题，逐句改编下列【歌词】。
 
注意：改编后与改编前每一句的字数都完全相同, 直接返回结果。
 
【歌词】：
{source}
 
【改编】：
"""
 
WORD_REPLACE = """
请适当替换句子中的[词语]，使得贴近给定的主题，替换的词和原词长度一致，直接返回结果。
 
主题：{theme}
 
句子：{ori_sentence}
 
直接返回结果：
"""
 
WORD_INC_DEC = """
将句子{option}1个汉字，使句子意思不变。
 
注意：返回结果必须用"【】"包裹。
 
句子：{sentence}
 
返回结果：
"""
 
ONOMATOPOEIC_WORDS = ["啊", "呀", "哎", "哟", "啦", "呜", "哩"]
 
 
def gen_lyrics(theme, source, source_slot):
    # first trail of lyrics generation
    prompt = PromptTemplate(input_variables=["theme", "source"], template=GEN_LYRICS_MAIN)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    results = llm_chain.run(**{"theme": theme, "source": source})
 
    # filter results
    ori_list = source.split('\n')
    ori_line_count = len(ori_list) - 2
    punc = '[’!"#$%&\'()*+,-./:;<=>?？@[\\]^_`{|}~。！，‘’“”]+'
    result_pure = re.sub(punc, "", results).replace(" ", "")
    target_list = result_pure.split('\n')[0:ori_line_count]
    match_list = []
    correct_list = []
 
    # check words count, adjust words
    for i in range(ori_line_count):
        if len(target_list[i]) != len(ori_list[i + 1]):
            ori_slot_list = source_slot.split('\n')
            tmp_prompt = PromptTemplate(input_variables=["theme", "ori_sentence"], template=WORD_REPLACE)
            tmp_llm_chain = LLMChain(llm=llm, prompt=tmp_prompt)
            tmp_result = tmp_llm_chain.run(**{"theme": theme, "ori_sentence": ori_slot_list[i + 1]})
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
            if residual > len(lyrics[i]):
                #print(new_lyrics[i], lyrics[i])
                new_lyrics_res.append(new_lyrics[i][:len(lyrics[i])])
                notes_res.extend(notes[i])
                durations_res.extend(durations[i])
            else:
                notes_res.extend(notes[i][:-residual])
                durations_res.extend(durations[i][:-residual])
 
                while residual > 0:
                    if " " in durations[i][-residual] or " " in notes[i][-residual]:
                        #print(durations[i][-residual].split(" ")[0], durations[i][-residual].split(" ")[1])
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
    #print(len(notes_res), len(durations_res), len(new_lyrics_res.replace("SP", "|")))
    notes_res = " | ".join(notes_res)
 
    durations_res = " | ".join(durations_res)
    return notes_res, durations_res, new_lyrics_res
 
 
def gen_songs_needs(theme, song):
    target_list, match_list, correct_list, new_lyrics_tmp = gen_lyrics(theme,
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
        '### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)第一次编写|[@thestmitsuk](https://github.com/thestmitsuki)二次补完': '### 2023.7.11|[@OOPPEENN](https://github.com/OOPPEENN)第一次编写|[@thestmitsuk](https://github.com/thestmitsuki)二次补完'
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
        '一键训练': 'One-click training',
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


def merge_audio_files(file1, file2, output_file):
    audio1 = AudioSegment.from_file(file1)
    audio2 = AudioSegment.from_file(file2)

    # 将两个音频文件进行叠加
    merged_audio = audio1.overlay(audio2)

    # 导出合并后的音频文件
    merged_audio.export(output_file, format='wav')

debug = True
local_model_root = 'weights'
Author_ZH = {"专业女歌手":"专业女歌手","yuhui":"RD瑜晖","Jay_Chou": "周杰伦", "ljr": "梁静茹", "dlj":"邓丽君", "luyao":"Paddle小鹿","yujun":"Paddle小军"}
Author_EN = {"专业女歌手":"专业女歌手","RD瑜晖":"yuhui", "Paddle小军": "yujun", "周杰伦" : "Jay_Chou" ,"梁静茹": "ljr", "邓丽君":"dlj", "Paddle小鹿":"luyao"}
Author_Sing = {"专业女歌手":"宁夏-改歌词","RD瑜晖":"yuhui", "Paddle小军":"翻唱-白月光与朱砂痣", "Paddle小鹿":"翻唱周杰伦-发如雪", "周杰伦" : "翻唱女生版-粉色海洋" ,"梁静茹": "翻唱张韶涵-亲爱的那不是爱情", "邓丽君":"翻唱日语版-我只在乎你"}

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

ckpt_list = [file for file in get_file_options(os.path.join('weights',Author_EN[spks[0]],'inference'), ".pth")]
model = None

i18n = I18nAuto()


if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"
        

def modelAnalysis(device, sid, msg, choice_ckpt):
    global model, spks, Author_ZH, Author_EN

    device = cuda[device] if "CUDA" in device else device

    device=device if device != "Auto" else None

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    
    print(device)
    model_path = os.path.join(local_model_root, Author_EN[sid], "inference", choice_ckpt)
    
    device_name = torch.cuda.get_device_properties(dev).name if "cuda" in str(dev) else str(dev)
    _name = "weights/"
    msg = f"成功加载模型到设备{device_name}上\n"
    msg += f"模型{model_path.split(_name)[1]}加载成功\n"
    msg += "当前模型的可用音色：\n"
    for i in spks:
        msg += i + " "

    model = model_path

    return sid, msg



def vc_batch_fn(sid, input_audio, auto_f0, vc_transform, choice_ckpt):

    global model
    try:
        if input_audio is None:
            # return input_audio, input_audio
            return "You need to upload an audio", None
        if model is None:
            return "You need to upload an model", None
        
        if os.path.exists("test.wav"):
            os.remove("test.wav")
            print(i18n("已清理残留文件"))
        else:
            print(i18n("无需清理残留文件"))
        
        print(i18n('开始推理'))
        
        print(f"Start processing: {input_audio}")
        
        input_audio_path = input_audio
        audio, sampling_rate = soundfile.read(input_audio_path)

        if np.issubdtype(audio.dtype, np.integer):
            audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)

        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio.transpose(1, 0))
        truncated_basename = os.path.basename(input_audio).split(".wav")[0]
        processed_audio = input_audio
        output_file_path = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment, choice_ckpt)

        os.system("rm -rf {}".format(input_audio))
        # os.system("rm -rf {}".format("results"))
        # os.system("rm -rf {}".format(spleeter_path))
        
        return "音乐合成成功啦，快来听听吧!", output_file_path

    except Exception as e:
        if debug:
            traceback.print_exc()
        raise gr.Error(e)


def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment, choice_ckpt):
    global model

    print(audio_path,sid,vc_transform,slice_db,cluster_ratio,auto_f0,noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment)
    
    train_config_path = 'configs/base.yaml'
    train_config_path = shlex.quote(train_config_path)
    keychange = shlex.quote(str(int(vc_transform)))

    resume_voice = Author_EN[sid]+".spk.npy"
    # "sovits5.0.pth"
    
    model_path = os.path.join(local_model_root, Author_EN[sid], "inference", choice_ckpt)

    cmd = ["python", "-u", "svc_inference.py", "--config", train_config_path, "--model", model_path, "--spk",
            f"data_svc/singer/{resume_voice}", "--wave", audio_path, "--shift", keychange]
    train_process = subprocess.run(cmd, shell=False, capture_output=True, text=True)
    print(train_process.stdout)
    print(train_process.stderr)
    print(i18n("推理成功"))

    return "svc_out.wav"
def refresh_options(sid, vc_transform):
    global ckpt_list
    
    audio_wav_file = Author_Sing[sid] + '.wav'

    if sid == "RD瑜晖" or sid == "Paddle小军" or sid == "周杰伦":
        vc_transform = -4
    else:
        vc_transform = 0
        
    if sid == "专业女歌手":
        return gr.Dropdown.update(choices=[], value="默认"), gr.Audio.update(label=Author_Sing[sid], value=audio_wav_file, interactive=False), vc_transform
    
    ckpt_list = [file for file in get_file_options(os.path.join('weights',Author_EN[sid],'inference'), ".pth")]
    

    
    return gr.Dropdown.update(choices=ckpt_list,value=ckpt_list[0]), gr.Audio.update(label=Author_Sing[sid], value=audio_wav_file, interactive=False), vc_transform




class GradioInfer:
    def __init__(self, exp_name, inference_cls, title, description, article, example_inputs):
        self.exp_name = exp_name
        self.title = title
        self.description = description
        self.article = article
        self.example_inputs = example_inputs

        pkg = ".".join(inference_cls.split(".")[:-1])
        cls_name = inference_cls.split(".")[-1]
        self.inference_cls = getattr(importlib.import_module(pkg), cls_name)
        
    def clear(self, name, text, notes, notes_duration):
        return '','','','',''

    def select_sid(self, name):
        text = lyrics_dict[Songs_CN_dict[name]]
        return name, text
    
    
    def generate_lyrics(self, name, sid_name):
        notes, notes_duration, text   = gen_songs_needs(name, Songs_CN_dict[sid_name])
    
        len_notes = len(notes.split('|'))
        len_notes_duration = len(notes_duration.split('|'))
        len_text = len(text) - text.count("SP")
        
        print(len_notes, len_notes_duration, len_text)
        
        if len_text!=len_notes_duration:
            return notes, notes_duration, text, "歌词长度为{}个单词，需要保持{}个单词，请点击【使用主题】重新生成!".format(len_text,len_notes_duration)
        
        text = text.replace("SP", "\n")
        
        return notes, notes_duration, text, "创作成功，快点击歌声合成吧～"
    
    def greet(self, name, text, notes, notes_duration, sid, auto_f0, vc_transform, choice_ckpt, sid_name, Ins):
        if not notes:
            notes = notes_dict[Songs_CN_dict[sid_name]]
            notes_duration = notes_duration_dict[Songs_CN_dict[sid_name]]
            
        len_notes = len(notes.split('|'))
        len_notes_duration = len(notes_duration.split('|'))
        if text[0] == "\n":
            text = text[1:]
        if text[-1]=="\n":
            text = text[:-1]
        text = text.replace("\n", "SP").replace(" ", "")
        len_text = len(text) - text.count("SP")
        print(len_notes, len_notes_duration, len_text)
        
        if len_text!=len_notes_duration:
            return None, "歌词长度为{}个单词，请保持{}个单词!".format(len_text,len_notes_duration)
        
        PUNCS = '。？；：？'
        sents = re.split(rf'([{PUNCS}])', text.replace('\n', ','))
        sents_notes = re.split(rf'([{PUNCS}])', notes.replace('\n', ','))
        sents_notes_dur = re.split(rf'([{PUNCS}])', notes_duration.replace('\n', ','))
        if sents[-1] not in list(PUNCS):
            sents = sents + ['']
            sents_notes = sents_notes + ['']
            sents_notes_dur = sents_notes_dur + ['']

        audio_outs = []
        s, n, n_dur = "", "", ""
        for i in range(0, len(sents), 2):
            if len(sents[i]) > 0:
                s += sents[i] + sents[i + 1]
                n += sents_notes[i] + sents_notes[i+1]
                n_dur += sents_notes_dur[i] + sents_notes_dur[i+1]
            if len(s) >= 400 or (i >= len(sents) - 2 and len(s) > 0):
                audio_out = self.infer_ins.infer_once({
                    'text': s,
                    'notes': n,
                    'notes_duration': n_dur,
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

        if sid != "专业女歌手":
            _,  output_file_path = vc_batch_fn(sid, 'temp.wav', auto_f0, vc_transform, choice_ckpt)
        
        if Ins:
            merge_audio_files(output_file_path, Ins_CN_dict[sid_name], output_file_path)

        return output_file_path, "合成成功，快去听听吧～"
    
    def run(self):

        set_hparams(os.path.join(script_path, "model/config.yaml"))
        infer_cls = self.inference_cls
        self.infer_ins: Infer = infer_cls(hp)
        example_inputs = self.example_inputs
        for i in range(len(example_inputs)):
            text, notes, notes_dur = example_inputs[i].split('<sep>')
            example_inputs[i] = ['', text, notes, notes_dur]
        iface = gr.Blocks()

        with iface:

            gr.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>{self.title}</h1>")
            
            gr.Markdown(value="""
                ### 「声韵创颂」 推理端口 v0.1
                
                仅供个人娱乐和非商业用途，禁止用于血腥、暴力、性相关、政治相关内容
                
                联系制作者：【飞桨&百度研究院大数据实验室】（陆瑶、王硕、边江）
                
                """)
            gr.Markdown(value=self.description)
            
            with gr.Column():
                with gr.Row():
                    sid = gr.Dropdown(label="音色（歌手）",choices = spks,value=spks[0])
                    choice_ckpt = gr.Dropdown(label="模型选择", choices=ckpt_list, value="默认")
                    model_load_button = gr.Button(value="加载模型", variant="primary")
                with gr.Row():
                    example_output = gr.Audio(label="声音试听", interactive=False, value="宁夏-改歌词.wav")
                    sid_output = gr.Textbox(label="Output Message")
                auto_f0 = gr.Checkbox(label="自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）", value=False, visible=False)
                
            vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
    
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        Ins = gr.Checkbox(label="是否加入伴奏，暂时只支持《宁夏》哦！", value=False)
                        btn_nx = gr.Button(value="宁夏", variant="primary")
                        btn_dx = gr.Button(value="稻香", variant="primary")
                        btn_kl = gr.Button(value="生日快乐", variant="primary")
                        text = gr.Textbox(lines=2, placeholder=None, value="", label="input text")
                        sid_name = gr.Dropdown(label="歌曲",choices = list(Songs_CN_dict.keys()),value="", visible=False)
                    with gr.Row():
                        name = gr.Textbox(lines=2, placeholder=None, label="输入你想创作的主题，例如：回家，我们会进行AI创作歌词哦～")
                        output_audio = gr.Audio(label="Output Audio", interactive=False)
                        
                    notes = gr.Textbox(lines=2, placeholder=None, value="", label="input note", visible=False)
                    notes_duration = gr.Textbox(lines=2, placeholder=None, value="", label="input duration", visible=False)
                    
                with gr.Row():
                    clear_button = gr.Button(value="清除信息", variant="primary")
                    with gr.Row():
                        submit1_button = gr.Button(value="使用主题", variant="primary")
                        submit2_button = gr.Button(value="歌声合成", variant="primary")
                        
            
            btn_dx.click(self.select_sid, [btn_dx], [sid_name, text])
            btn_nx.click(self.select_sid, [btn_nx], [sid_name, text])
            btn_kl.click(self.select_sid, [btn_kl], [sid_name, text])

            device = gr.Dropdown(label="推理设备, 默认为自动选择CPU和GPU", choices=["Auto",*cuda.keys(),"cpu"], value="Auto", visible=False)
            
            sid.select(refresh_options,[sid, vc_transform],[choice_ckpt, example_output, vc_transform])
            
            model_load_button.click(modelAnalysis,[device, sid, sid_output, choice_ckpt],[sid,sid_output])

            submit1_button.click(self.generate_lyrics, [name, sid_name], [notes, notes_duration, text, sid_output])
            
            submit2_button.click(self.greet, [name, text, notes, notes_duration,sid, auto_f0, vc_transform, choice_ckpt, sid_name, Ins], [output_audio, sid_output])
            
            clear_button.click(self.clear, [name, text, notes, notes_duration], [name, text, notes, notes_duration])
    
        
        # iface.launch(enable_queue=True)
        iface.launch(share=True, server_name="0.0.0.0", server_port=8913)


if __name__ == '__main__':
    gradio_config = yaml.safe_load(open('settings-1.yaml'))
    g = GradioInfer(**gradio_config)
    g.run()