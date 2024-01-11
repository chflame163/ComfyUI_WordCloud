import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image, ImageOps
import jieba


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


COLOR_MAP = ['viridis', 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r',
             'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r',
             'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges',
             'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
             'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r',
             'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r',
             'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r',
             'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral',
             'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr',
             'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r',
             'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis',
             'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r',
             'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r',
             'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar',
             'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r',
             'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r',
             'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno',
             'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r',
             'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r',
             'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
             'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c',
             'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
             'twilight_shifted', 'twilight_shifted_r', 'viridis_r', 'winter', 'winter_r']

DEFAULT_FONT = os.path.join(os.path.dirname(os.path.dirname(os.path.normpath(__file__))), 'font')
DEFAULT_FONT = os.path.join(DEFAULT_FONT,'Alibaba-PuHuiTi-Heavy.ttf')
DEFAULT_TEXT = 'Word Cloud for ComfyUI by dzNodes'


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
                "font_path": ("STRING", {"default": "c:\\font.ttf"}),  # 字体文件
                "min_font_size": ("INT", {"default": 4}),  # 单词最小size
                "max_font_size": ("INT", {"default": 128}),  # 单词最大size
                # "font_step": ("INT", {"default": 1}),  # 字体迭代步长，大于1时计算速度加快但易导致错误
                "relative_scaling": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),  # 单词大小（按频率）离散度
                ## color control
                "colormap": (COLOR_MAP,),  # 文字颜色
                "background_color": ("STRING", {"default": "#FFFFFF"}),  # 背景颜色
                "transparent_background": ("BOOLEAN", {"default": False}),  # 是否透明，如果是则background_color强制为None
                ## word control
                "prefer_horizontal": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),  # 横排比例
                "max_words": ("INT", {"default": 200}),  # 最大单词数量
                "repeat": ("BOOLEAN", {"default": False}),  # 允许重复单词直到最大单词数量
                # "min_word_length": ("INT", {"default": 0}),
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
        if text == None:
            text = DEFAULT_TEXT
        freq_dict = WordCloud().process_text(' '.join(jieba.cut(text)))

        if not keynote_words == None:
            keynote_list = list(re.split(r'[，,\s*]', keynote_words))
            keynote_dict = {keynote_list[i]: keynote_weight + max(freq_dict.values()) for i in range(len(keynote_list))}
            freq_dict.update(keynote_dict)
        print(f"# 😺dzNodes: WordCloud:  -> word frequencies dict is {freq_dict}")

        if not os.path.exists(os.path.normpath(font_path)):
            print(f"# 😺dzNodes: WordCloud:  -> {font_path} font_path is invalid, use default font.")
            font_path = DEFAULT_FONT

        stopwords_set = set(STOPWORDS).union(set(re.split(r'[，,\s*]', stopwords)))

        mode = 'RGB'
        if transparent_background:
            background_color = None
            mode = 'RGBA'

        if random_state == -1:
            random_state = None

        mask = None
        if not mask_image == None:
            mask = np.array(img_whitebackground(tensor2pil(mask_image)))

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
            image_colors = ImageColorGenerator(np.array(tensor2pil(color_ref_image)))
            wc.recolor(color_func=image_colors)

        return (pil2tensor(wc.to_image()),)


NODE_CLASS_MAPPINGS = {
    "ComfyWordCloud": ComfyWordCloud
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyWordCloud": "Word Cloud"
}