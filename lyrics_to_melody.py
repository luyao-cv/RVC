#!/usr/local/bin/python3
# -*- encoding: utf-8 -*-
"""
@Author : wangshuo
@Contact : wangshuo41@baidu.com
@File : lyrics_to_melody.py.py
@Time : 2023/12/22 3:19 PM
@Desc :
"""
import re
import ast
import random
import jieba
from rhythm_duration import get_rhythm_duration
from rhythm import gen_rhythm
from langchain.chat_models import ErnieBotChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from emotional_chord import SongsType
from lyrics_to_medoly_prompt import songs_prompt, melody_prompt

# llm初始化
llm = ErnieBotChat(ernie_client_id='96BMhQM5simx6R97yDl483Zm',
                   ernie_client_secret='9e05mDOjHoyXD7Sb9GA1l420uaZ6vGMo',
                   model_name='ERNIE-Bot-4',
                   top_p=0)


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


def review_melody(lyrics, notes, durations, values, emotion):
    prompt = PromptTemplate(input_variables=["notes", "durations", "values", "lyrics", "emotion"],
                            template=melody_prompt)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    results = llm_chain.run(
        **{"notes": notes, "durations": durations, "values": values, "lyrics": lyrics, "emotion": emotion})
    results = extract_json(results)
    durations_new = results["durations"]
    notes_new = results["notes"]
    return notes_new, durations_new


def gen_lyrics(theme, emotion):
    prompt = PromptTemplate(input_variables=["theme", "emotion"], template=songs_prompt)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    results = llm_chain.run(**{"theme": theme, "emotion": emotion})
    results = extract_json(results)
    if results["lyrics"].startswith("SP"):
        results["lyrics"] = results["lyrics"][2:]
    if results["lyrics"].endswith("SP"):
        results["lyrics"] = results["lyrics"][:-2]
    song_name = results["song_name"]
    lyrics = results["lyrics"]
    print("主题：", theme)
    print("情感色彩：", emotion)
    print("创作的歌曲名：", song_name)
    print("创作的歌词：", lyrics)
    return song_name, lyrics


def get_need_lyrics(lyric_lines):
    check_need_lyrics = []
    cut_lyric_line_lst = []
    for lyric_line in lyric_lines:
        lyric_line = jieba.lcut(lyric_line)
        cut_lyric_line_lst.append(lyric_line)
        slot_line = ""
        for word in lyric_line:
            slot_line += "[" + word + "]"
        check_need_lyrics.append(slot_line)
    return cut_lyric_line_lst, check_need_lyrics


def lyrics_to_melody(theme, emotion):
    """入口函数，返回歌词、音符、音符时值"""
    songs_type = SongsType[emotion]["type"]
    beat_count = SongsType[emotion]["beat_count"]
    beat_count += random.randint(0, 10)
    print("节拍：", beat_count)
    song_name, lyrics = gen_lyrics(theme, emotion)
    lyrics = lyrics.replace("\n", "SP").replace("，", "SP").replace("。", "SP")
    while re.search(r'SPSP', lyrics):
        lyrics = re.sub(r'SPSP', "SP", lyrics)
    punc = '[’!"#$%&\'()*+,-./:;<=>?？@[\\]^_`{|}~。！，]+'
    lyrics = re.sub(punc, "", lyrics).replace(" ", "")
    lyric_lines = lyrics.split("SP")
    cut_lyric_line_lst, check_need_lyrics = get_need_lyrics(lyric_lines)
    durations, values = get_rhythm_duration(beat_count, cut_lyric_line_lst, songs_type)
    values = sorted(values, reverse=True)[2:-2]
    notes = gen_rhythm(cut_lyric_line_lst, songs_type)
    print("####" * 30)
    print("随机生成，优化前：\n{}\n{}\n{}".format(lyrics, " | ".join(notes), " | ".join([str(num) for num in durations])))
    print("####" * 30)
    # 优化
    try:
        notes_new, durations_new = review_melody(lyrics, notes, durations, values, emotion)
    except Exception as e:
        print("####" * 30)
        print(e)
        notes = " | ".join(notes)
        durations = " | ".join([str(num) for num in durations])
        return lyrics, notes, durations

    note_len, durations_len = len(notes), len(durations)
    note_new_len, durations_new_len = len(notes_new), len(durations_new)
    print("优化前：音符长度{},节奏时值长度{}\n优化后:音符长度{},节奏时值长度{}".format(note_len, durations_len,
                                                                                      note_new_len, durations_new_len))
    if note_len == note_new_len:
        if durations_len < durations_new_len:
            durations_new = durations_new[:(durations_len - durations_new_len)]

        if durations_len > durations_new_len:
            if durations_len - durations_new_len == 1:
                durations_new.append(random.choice(values))
            else:
                print("####" * 30)
                print("优化失败，【节奏时值】未对齐,使用随机生成版本")
                notes = " | ".join(notes)
                durations = " | ".join([str(num) for num in durations])
                return lyrics, notes, durations
        if durations_len == len(durations_new):
            print("####" * 30)
            notes_new = " | ".join(notes_new)
            durations_new = " | ".join([str(num) for num in durations_new])
            print("优化后：\n{}\n{}\n{}".format(lyrics, notes_new, durations_new))
            print("####" * 30)
            return lyrics, notes_new, durations_new
        print("####" * 30)
        print("优化失败，【节奏时值】未对齐,使用原始生成版本")
        notes = " | ".join(notes)
        durations = " | ".join([str(num) for num in durations])
        return lyrics, notes, durations
    else:
        print("####" * 30)
        print("优化失败，【音符】未对齐,使用原始生成版本")
        notes = " | ".join(notes)
        durations = " | ".join([str(num) for num in durations])
        return lyrics, notes, durations


if __name__ == "__main__":
    # 遗憾.
    # 抒情悲伤
    # theme = "生活的两面"
    # emotion = "振奋激励"
    # theme = "命运的安排"
    # emotion = "抒情悲伤"
    # theme = "善良与邪恶"
    # emotion = "抒情悲伤"
    theme = "小跳蛙"
    emotion = "欢快愉悦"
    lyrics, notes, durations = lyrics_to_melody(theme, emotion)
