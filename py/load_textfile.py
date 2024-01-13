import os

class LoadTextFile:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": 'c:\\text.txt'}),
            },
            "optional": {
            },
        }


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Text",)
    FUNCTION = "load_text_file"
    OUTPUT_NODE = True
    CATEGORY = '😺dzNodes'

    def load_text_file(self, path):

        text_content = ""
        try:
            with open(os.path.normpath(path), 'r',  encoding="utf-8") as f:
                text_content = ''.join(str(l) for l in f.read())
            print("# 😺dzNodes: Load Text File -> " + path + " success.")
        except Exception as e:
            print("# 😺dzNodes: Load Text File -> ERROR, " + path + ", " + repr(e))

        return {"ui": {"text":text_content}, "result": (text_content,)}


NODE_CLASS_MAPPINGS = {
    "LoadTextFile": LoadTextFile
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadTextFile": "Load Text File"
}