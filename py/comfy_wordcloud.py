import os
import glob
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image, ImageOps
import jieba

def log(message):
    name = 'WordCloud'
    print(f"# 😺dzNodes: {name} ->  {message}")

COLOR_MAP = ['viridis', 'Accent', 'Blues', 'BrBG', 'BuGn', 'BuPu', 'CMRmap','Dark2', 'GnBu',
             'Grays', 'Greens', 'OrRd', 'Oranges', 'PRGn', 'Paired', 'Pastel1',
             'Pastel2', 'PiYG', 'PuBu', 'PuBuGn', 'PuOr', 'PuRd', 'Purples', 'RdBu', 'RdGy',
             'RdPu', 'RdYlBu', 'RdYlGn', 'Reds', 'Set1', 'Set2', 'Set3', 'Spectral', 'Wistia',
             'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd', 'afmhot', 'autumn', 'binary', 'bone',
             'brg', 'bwr', 'cividis', 'cool', 'coolwarm', 'copper', 'cubehelix', 'flag',
             'gist_earth', 'gist_gray', 'gist_grey', 'gist_heat', 'gist_ncar', 'gist_rainbow',
             'gist_stern', 'gist_yarg', 'gist_yerg', 'gnuplot', 'gnuplot2',
             'hot', 'hsv', 'inferno', 'jet', 'magma', 'nipy_spectral', 'ocean', 'pink', 'plasma',
             'prism', 'rainbow', 'seismic', 'spring', 'summer', 'tab10', 'tab20', 'tab20b', 'tab20c',
             'terrain', 'turbo', 'twilight', 'twilight_shifted', 'winter'
             ]

default_text = 'demo of word cloud for ComfyUI by dzNodes'
font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'font')
ini_file = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), "font_dir.ini")

try:
    with open(ini_file, 'r') as f:
        ini = f.read()
        dir = ini[ini.find('=') + 1:].rstrip().lstrip()
        if os.path.exists(dir):
            font_dir = dir
        else:
            log(f'ERROR: invalid dir, default to be used. check {ini_file}')
except Exception as e:
    log(f'ERROR: {ini_file} ' + repr(e))

file_list = glob.glob(font_dir + '/*.ttf')
file_list.extend(glob.glob(font_dir + '/*.otf'))
font_dict = {}
for i in range(len(file_list)):
    _, filename =  os.path.split(file_list[i])
    font_dict[filename] = file_list[i]
font_list = list(font_dict.keys())
log(f'find {len(font_list)} fonts in {font_dir}')

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def img_whitebackground(image):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    width = image.width
    height = image.height
    img_new = Image.new('RGB',size=(width,height),color=(255,255,255))
    img_new.paste(image,(0,0),mask=image)

    return img_new

class ComfyWordCloud:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),  # 文本内容
                ## size
                "width": ("INT", {"default": 512}),  # 画幅宽
                "height": ("INT", {"default": 512}),  # 画幅高
                "scale": ("FLOAT", {"default": 1, "min": 0.1, "max": 1000.0, "step": 0.01}),  # 放大倍数
                "margin": ("INT", {"default": 0}),  # 空白边界

                ## font
                "font_path": (font_list,),  # 字体文件
                "min_font_size": ("INT", {"default": 4}),  # 单词最小size
                "max_font_size": ("INT", {"default": 128}),  # 单词最大size
                # "font_step": ("INT", {"default": 1}),  # 字体迭代步长，大于1时计算速度加快但易导致错误
                "relative_scaling": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),  # 单词大小离散度
                ## color control
                "colormap": (COLOR_MAP,),  # 文字颜色
                "background_color": ("STRING", {"default": "#FFFFFF"}),  # 背景颜色
                "transparent_background": ("BOOLEAN", {"default": False}),  # 是否透明，如果是则需要background_color强制为None
                ## word control
                "prefer_horizontal": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),  # 横排比例
                "max_words": ("INT", {"default": 200}),  # 最大单词数量
                "repeat": ("BOOLEAN", {"default": False}),  # 允许重复单词直到最大单词数量
                # "min_word_length": ("INT", {"default": 0}),  # 最小单词长度
                "include_numbers": ("BOOLEAN", {"default": False}),  # 是否包含数字
                # "collocations": ("BOOLEAN", {"default": False}),  # 词组关联开关
                # "collocation_threshold": ("INT", {"default": 30}),  # 词组关联度
                # "normalize_plurals": ("BOOLEAN", {"default": True}),  # 复数单词转单数
                # "ranks_only": ("BOOLEAN", {"default": False}),  # 仅显示高频词
                "random_state": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),  # 固定随机值，-1时强制转为None（随机）
                "stopwords": ("STRING", {"default": ""}),  # 排除词，用中英文逗号或空格分开
                # "regexp": ("STRING", {"default": "", "multiline": True}),  # 正则表达式 string or None
            },
            "optional": {
                ## recolor refrence image
                "color_ref_image": ("IMAGE", ),
                ## mask image 白底或带alpha通道
                "mask_image": ("IMAGE", ),  # 有输入mask则强制使用该图尺寸
                "contour_width": ("FLOAT", {"default": 0, "min": 0, "max": 9999, "step": 0.1}),
                "contour_color": ("STRING", {"default": "#000000"}),
                "keynote_words": ("STRING", {"default": ""}),  # 重点词，用中英文逗号或空格分开
                "keynote_weight": ("INT", {"default": 60}),  # 重点词加权
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'wordcloud'
    CATEGORY = '😺dzNodes'
    OUTPUT_NODE = True

    def wordcloud(self, text, width, height, margin, scale, font_path,
                  min_font_size, max_font_size, relative_scaling,
                  colormap, background_color, transparent_background,
                  prefer_horizontal, max_words, repeat,
                  include_numbers, random_state, stopwords,
                  color_ref_image=None, mask_image=None,
                  contour_width=None, contour_color=None,
                  keynote_words=None, keynote_weight=None,
                  ):

        # parameter preprocessing
        if text == '':
            text = default_text
            log(f"text input not found, use demo string.")

        freq_dict = WordCloud().process_text(' '.join(jieba.cut(text)))
        if not keynote_words == '':
            keynote_list = list(re.split(r'[，,\s*]', keynote_words))
            keynote_list = [x for x in keynote_list if x != '']  # 去除空字符
            keynote_dict = {keynote_list[i]: keynote_weight + max(freq_dict.values()) for i in range(len(keynote_list))}
            freq_dict.update(keynote_dict)
        log(f"word frequencies dict generated, include {len(freq_dict)} words.")


        font_path = font_dict[font_path]
        if not os.path.exists(font_path):
            font_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'font'),
                                     'Alibaba-PuHuiTi-Heavy.ttf')
            log(f"font_path not found, use {font_path}")
        else:
            log(f"font_path = {font_path}")

        stopwords_set = set("")
        if not stopwords == "":
            stopwords_list = re.split(r'[，,\s*]', stopwords)
            stopwords_set= set([x for x in stopwords_list if x != ''])  # 去除空字符
            
            # 同时在词典中删除（stopwords之bug）
            for item in stopwords_set:
                if item in freq_dict.keys():
                    del freq_dict[item]

        mode = 'RGB'
        if transparent_background:
            background_color = None
            mode = 'RGBA'

        if random_state == -1:
            random_state = None

        mask = None
        image_width = width
        image_height = height
        if not mask_image == None:
            p_mask = tensor2pil(mask_image)
            mask = np.array(img_whitebackground(p_mask))
            image_width = p_mask.width
            image_height = p_mask.height


        # set wordcloud parameters
        wc = WordCloud(width=width, height=height, scale=scale, margin=margin,
                       font_path=font_path, min_font_size=min_font_size, max_font_size=max_font_size,
                       relative_scaling=relative_scaling, colormap=colormap, mode=mode,
                       background_color=background_color, prefer_horizontal=prefer_horizontal,
                       max_words=max_words, repeat=repeat, include_numbers=include_numbers,
                       random_state=random_state, stopwords=stopwords_set,
                       mask=mask, contour_width=contour_width, contour_color=contour_color,
                       )

        # generate wordcloud
        wc.generate_from_frequencies(freq_dict)

        # generate recolor
        if not color_ref_image == None:
            p_color_ref_image = tensor2pil(color_ref_image)
            p_color_ref_image = p_color_ref_image.resize((image_width, image_height))
            image_colors = ImageColorGenerator(np.array(p_color_ref_image))
            wc.recolor(color_func=image_colors)

        return (pil2tensor(wc.to_image()),)


NODE_CLASS_MAPPINGS = {
    "ComfyWordCloud": ComfyWordCloud
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyWordCloud": "Word Cloud"
}