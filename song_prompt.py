#!/usr/local/bin/python3
# -*- encoding: utf-8 -*-


THEME_EXTEND = """
请你根据你的知识，扩展【主题】与相关的十个关键词，要求关键词积极向上，直接返回以逗号分割的关键词。
返回```json```格式：```json{{"keywords": ""}}```

【主题】：儿时记忆
返回：```json{{"keywords": "玩具，动画片，童话，糖果，游戏"}}```

【主题】：{theme}
返回：
"""

SUMMARY = """
请你根据你的知识，根据【主题】，写一段100字左右的介绍，要求与主题契合。
返回```json```格式：```json{{"summary": ""}}```

【主题】：{theme}
返回：
"""


GEN_LYRICS_MAIN = """
你是一个写歌词的专家，请以参考【主题】、【主题摘要】，模仿【歌词】的内容逐句生成与【歌词】长度一致（字数相同）的句子。
返回```json```格式：```json{{"lyrics": ""}}```
要求：
1、生成的句子语义连贯，契合主题
2、生成的句子必须和【歌词】中对应句子（序号相同）的字数完全相同
3、逐句改编歌词，根据原歌词的长度和句式，生成与原句长度相同、句式相同的新歌词

【歌词】：'\n我恭喜你发财\n我恭喜你精彩\n'
【主题】：科技创新
【主题摘要】：科技创新是推动人类社会进步的重要动力，它涵盖了众多领域，如人工智能、生物技术、新能源等。通过不断的研究与探索，科技创新为我们带来了更高效、更便捷、更安全的现代化生活。
返回：```json{{"lyrics": "我赞你创新威SP我赞你科技辉"}}```

【歌词】：{source}
【主题】：{theme}
【主题摘要】：{theme_summary}
返回：
"""


WORD_REPLACE = """
你是一个写歌词的专家，能够理解长度计算的约束，请以"{theme}"为主题，模仿【歌词】的内容生成与【歌词】长度一致（字数相同）的句子。
返回```json```格式：```json{{"line": ""}}```
要求：
1、生成的句子语义通顺合理
2、生成的句子必须和【歌词】的字数长度完全相同。

【歌词】：{ori_sentence}
返回：
"""

# 似乎没用上这个
WORD_INC_DEC = """
将句子{option}1个汉字，使句子意思不变。

注意：返回结果必须用"【】"包裹。

句子：{sentence}

返回结果：
"""
