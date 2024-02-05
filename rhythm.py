#!/usr/local/bin/python3
# -*- encoding: utf-8 -*-
"""
@Author : wangshuo
@Contact : wangshuo41@baidu.com
@File : rhythm.py
@Time : 2023/12/22 5:13 PM
@Desc :
"""
import random

from emotional_chord import EmotionalChord
from chordal import ChordalTempletes


def get_start_same_chordal(chordal_lst, end_note):
    for chordal in chordal_lst:
        if ChordalTempletes[chordal][0] == end_note:
            return chordal
    return random.choice(chordal_lst)


def gen_rhythm(cut_text_line_lst, songs_type):
    notes = []
    chordal_lst = random.choice(list(EmotionalChord[songs_type].values()))
    pre_chordal = ''
    note_set = "4"
    for cut_text_line in cut_text_line_lst:
        for word in cut_text_line:
            word_len = len(word)
            if random.choice([True, False]) and pre_chordal:
                chordal = pre_chordal
            else:
                chordal = random.choice(chordal_lst)
            chordal_note_lst = ChordalTempletes[chordal]
            if random.choice([True, False]):
                note_tmp = random.choices(chordal_note_lst, k=1)
                note_tmp = note_tmp * word_len
            else:
                note_tmp = random.choices(chordal_note_lst, k=word_len)
            notes.extend(note_tmp)
            pre_chordal = chordal
        notes.append("rest")

    notes_res = []
    for note in notes:
        if note != "rest":
            notes_res.append(note + note_set)
        else:
            notes_res.append("rest")
    return notes_res[:-1]
