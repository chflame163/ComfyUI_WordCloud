
mode_list = ['HEX', 'DEC']

def hex_to_dec(inhex):
    rval = inhex[1:3]
    gval = inhex[3:5]
    bval = inhex[5:]
    rgbval = (int(rval, 16), int(gval, 16), int(bval, 16))
    return rgbval

class RGB_Picker:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "color": ("COLOR", {"default": "white"},),
                "mode": (mode_list,),  # 输出模式
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = 'picker'
    CATEGORY = '😺dzNodes/WordCloud'
    OUTPUT_NODE = True

    def picker(self, color, mode,):
        ret = color
        if mode == 'DEC':
            ret = hex_to_dec(color)
        return (ret,)


NODE_CLASS_MAPPINGS = {
    "RGB_Picker": RGB_Picker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGB_Picker": "RGB Color Picker"
}