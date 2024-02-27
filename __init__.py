import importlib.util
import glob
import os
import sys
import filecmp
import shutil
import __main__
from .dzNodes import init, get_ext_dir, log

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

python = sys.executable
extentions_folder = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)),
                                 "web" + os.sep + "extensions" + os.sep + "mtb")
javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mtb")
outdate_file_list = ['comfy_shared.js', 'debug.js', 'mtb_widgets.js', 'parse-css.js', 'dz_widgets.js']

if not os.path.exists(extentions_folder):
    log('Making the "web\extensions\mtb" folder')
    os.mkdir(extentions_folder)
else:
    for i in outdate_file_list:
        outdate_file = os.path.join(extentions_folder, i)
        if os.path.exists(outdate_file):
            os.remove(outdate_file)

result = filecmp.dircmp(javascript_folder, extentions_folder)

if result.left_only or result.diff_files:
    log('Update to javascripts files detected')
    file_list = list(result.left_only)
    file_list.extend(x for x in result.diff_files if x not in file_list)

    for file in file_list:
        log(f'WordCloud: Copying {file} to extensions folder')
        src_file = os.path.join(javascript_folder, file)
        dst_file = os.path.join(extentions_folder, file)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        #log("disabled")
        shutil.copy(src_file, dst_file)

if init():
    py = get_ext_dir("py")
    files = glob.glob("*.py", root_dir=py, recursive=False)
    for file in files:
        name = os.path.splitext(file)[0]
        spec = importlib.util.spec_from_file_location(name, os.path.join(py, file))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
