#!/usr/local/bin/python3
# -*- encoding: utf-8 -*-
"""
@Author : wangshuo
@Contact : wangshuo41@baidu.com
@File : rhythm_duration.py
@Time : 2023/12/22 4:00 PM
@Desc :
"""
import random
import jieba

# 二分音符、四分音符、八分音符、十六分音符
beat_type = ["minim", "quaver", "crotchet", "semiquaver"]


def get_rhythm_duration_v1(beat_count, cut_text_line_lst, songs_type):
    """
    获取节奏的持续时间
    :param beat_count: 节奏的节拍数
    :param beat_type: 节奏的节拍类型
    :return: 节奏的持续时间
    """
    quaver = round(60 / beat_count, 3)
    minim = round(2 * quaver, 3)
    crotchet = round(quaver / 2, 3)
    semiquaver = round(crotchet / 2, 3)
    values = {"minim": minim, "quaver": quaver, "crotchet": crotchet, "semiquaver": semiquaver}
    durations = []
    for cut_text_line in cut_text_line_lst:
        # if cut_text_line == cut_text_line_lst[-1]:
        #     probabilities["minim"] += probabilities["minim"] / 2
        #     probabilities["quaver"] += random.uniform(0, 1 - probabilities["quaver"] - 0.1)
        #     probabilities["crotchet"] -= random.uniform(0, 1 - probabilities["crotchet"] - 0.2)
        #     probabilities["semiquaver"] -= random.uniform(probabilities["semiquaver"] / 4, probabilities["semiquaver"])
        word_count = 0
        for idx in range(len(cut_text_line)):
            if word_count % 2 == 0:
                probabilities = {"minim": 0.05, "quaver": 0.7, "crotchet": 0.2, "semiquaver": 0.05}
            else:
                probabilities = {"minim": 0.01, "quaver": 0.09, "crotchet": 0.7, "semiquaver": 0.1}
            word_num = len(cut_text_line[idx])

            # 生成介于0和1之间的随机数
            random_number = random.uniform(0, 1)
            # 60%的概率走if分支，40%的概率走else分支
            if random_number <= 0.6:
                result = random.choices(list(probabilities.keys()), weights=list(probabilities.values()))
                result = result * word_num
            else:
                result = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=word_num)
            durations.extend(result)
            word_num += word_num
        durations.append(random.choice(beat_type))
    durations = [values[item] for item in durations[:-1]]
    return durations, list(values.values())


def get_rhythm_duration_v2(beat_count, cut_text_line_lst, songs_type):
    """
    获取节奏的持续时间
    :param beat_count: 节奏的节拍数
    :param beat_type: 节奏的节拍类型
    :return: 节奏的持续时间
    """
    quaver = round(60 / beat_count, 3)
    minim = round(2 * quaver, 3)
    crotchet = round(quaver / 2, 3)
    semiquaver = round(crotchet / 2, 3)
    values = {"minim": minim, "quaver": quaver, "crotchet": crotchet, "semiquaver": semiquaver}
    durations = []
    for cut_text_line in cut_text_line_lst:
        word_count = 0
        for idx in range(len(cut_text_line)):
            for char in cut_text_line[idx]:
                if len(cut_text_line[idx]) > 1:
                    if word_count % 2 == 1:
                        probabilities = {"minim": 0.09, "quaver": 0.5, "crotchet": 0.4, "semiquaver": 0.01}
                    else:
                        probabilities = {"minim": 0, "quaver": 0.4, "crotchet": 0.6, "semiquaver": 0}
                else:
                    if word_count % 2 == 1:
                        probabilities = {"minim": 0.09, "quaver": 0.5, "crotchet": 0.4, "semiquaver": 0.01}
                    else:
                        probabilities = {"minim": 0, "quaver": 0.4, "crotchet": 0.58, "semiquaver": 0.02}
                word_count += 1
                result = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)
                durations.extend(result)
        rest_probabilities = {"minim": 0.01, "quaver": 0.01, "crotchet": 0.18, "semiquaver": 0.8}
        durations.extend(random.choices(list(rest_probabilities.keys()), weights=list(probabilities.values()), k=1))
    durations = [values[item] for item in durations[:-1]]
    return durations, list(values.values())


def get_rhythm_duration(beat_count, cut_text_line_lst, songs_type):
    """
    获取节奏的持续时间
    :param beat_count: 节奏的节拍数
    :param beat_type: 节奏的节拍类型
    :return: 节奏的持续时间
    """
    quaver = round(60 / beat_count, 3)
    minim = round(2 * quaver, 3)
    crotchet = round(quaver / 2, 3)
    semiquaver = round(crotchet / 2, 3)
    quaver_half = round(quaver + crotchet, 3)
    crotchet_half = round(crotchet + semiquaver, 3)
    semiquaver_half = round(semiquaver + (semiquaver / 2), 3)
    values = {"minim": minim, "quaver": quaver, "crotchet": crotchet, "semiquaver": semiquaver,
              "quaver_half": quaver_half, "crotchet_half": crotchet_half, "semiquaver_half": semiquaver_half}
    durations = []
    for cut_text_line in cut_text_line_lst:
        word_count = 0
        for idx in range(len(cut_text_line)):
            for char in cut_text_line[idx]:
                if char == cut_text_line[-1][-1]:
                    probabilities = {"minim": 0, "quaver_half": 0.05, "quaver": 0.5, "crotchet_half": 0.4,
                                     "crotchet": 0.05, "semiquaver_half": 0, "semiquaver": 0}
                else:
                    if len(cut_text_line[idx]) > 1:
                        if word_count % 2 == 1:
                            probabilities = {"minim": 0.02, "quaver_half": 0.05, "quaver": 0.5, "crotchet_half": 0.3,
                                             "crotchet": 0.1, "semiquaver_half": 0.03, "semiquaver": 0}
                        else:
                            probabilities = {"minim": 0, "quaver_half": 0.03, "quaver": 0.3, "crotchet_half": 0.54,
                                             "crotchet": 0.11, "semiquaver_half": 0.02, "semiquaver": 0}
                    else:
                        if word_count % 2 == 1:
                            probabilities = {"minim": 0.02, "quaver_half": 0.01, "quaver": 0.4, "crotchet_half": 0.34,
                                             "crotchet": 0.2, "semiquaver_half": 0.02, "semiquaver": 0.01}
                        else:
                            probabilities = {"minim": 0, "quaver_half": 0, "quaver": 0.05, "crotchet_half": 0.4,
                                             "crotchet": 0.4, "semiquaver_half": 0.1, "semiquaver": 0.05}
                word_count += 1
                result = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)
                durations.extend(result)
        rest_probabilities = {"minim": 0, "quaver_half": 0, "quaver": 0.05, "crotchet_half": 0.3, "crotchet": 0.3, "semiquaver_half": 0.3, "semiquaver": 0.05}
        durations.extend(random.choices(list(rest_probabilities.keys()), weights=list(probabilities.values()), k=1))
    durations = [values[item] for item in durations[:-1]]
    return durations, list(values.values())

