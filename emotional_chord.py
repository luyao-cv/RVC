#!/usr/local/bin/python3
# -*- encoding: utf-8 -*-
"""
@Author : wangshuo
@Contact : wangshuo41@baidu.com
@File : emotional_chord.py
@Time : 2023/12/22 3:34 PM
@Desc :
"""
import chordal

# 欢快/愉悦
upbeat_and_joyful = {"小苹果": ["Am", "F", "G", "Em", "Dm7"],
                     "欢乐颂": ["C", "G", "Am", "F"],
                     "青春纪念手册": ["F", "Em", "Dm", "G", "Am", "C"],
                     "小跳蛙": ["G", "D", "A"],
                     "童年": ["C", "Am7", "F", "G", "Em", "D7"]}

# 悲伤/抒情
melancholic_and_expressive = {"遥远的她": ["G", "Am", "Em", "C", "D"],
                              "我们的爱": ["F", "Am", "G"],
                              "闹够了没有": ["Am", "D", "G", "C", "Em"],
                              "黄昏": ["B", "C", "D", "E", "F", "G", "A"]}

# 浪漫/温馨
romantic_and_heartwarming = {"灰姑娘": ["A", "C", "G", "G7", "Em"],
                             "情难自控": ["Am7", "D", "G", "Em", "C"],
                             "倾国倾城": ["C", "G", "Am", "F", "Dm", "D"]}

# 振奋/激励
uplifting_and_inspiring = {"海阔天空": ["C", "G", "Am", "F"],
                           "真心英雄": ["G", "D", "Em", "Bm", "C", "Am"],
                           "不再犹豫": ["G", "D", "Em", "Bm", "Am7", "C"],
                           "奔跑": ["G", "D", "Em", "C"]}
# 紧张/惊悚
tense_and_thrilling = {}
# 放松/平静
relaxing_and_calm = {}
# 愤怒/激昂
angry_and_intense = {}
# 思考/沉思
reflective_and_thoughtful = {}

SongsType = {"欢快愉悦": {"type": "upbeat_and_joyful", "beat_count": 125},
             "浪漫温馨": {"type": "romantic_and_heartwarming", "beat_count": 110},
             "振奋激励": {"type": "melancholic_and_expressive", "beat_count": 95},
             "抒情悲伤": {"type": "melancholic_and_expressive", "beat_count": 85}}

EmotionalChord = {"upbeat_and_joyful": upbeat_and_joyful, "melancholic_and_expressive": melancholic_and_expressive,
                  "relaxing_and_calm": relaxing_and_calm, "romantic_and_heartwarming": romantic_and_heartwarming,
                  "uplifting_and_inspiring": uplifting_and_inspiring}
