import os
import re
import subprocess
import platform

INCLUDE_RE = re.compile(r'^#include "(.*?)".*$')

class CompileError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)

cache = {}

# Vulkan GLSL preprocessor
os_name = platform.system()
if os_name == "Windows":
    # D:\Programs\Vulkan\Bin\glslc.exe
    glslc_path = 'D:/Programs/Vulkan/Bin/glslc.exe'
elif os_name == "Linux":
    glslc_path = '/usr/bin/glslc'
else:
    assert False, "Platform isn't supported."


# !!! this is the root of shaders relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
REL_ROOT = '../../engine/shader/'
# REL_ROOT = './glsl-precompiler-test/'

# the root of shaders
shaders_root = os.path.join(script_dir, REL_ROOT)
# the output dir
output_dir = os.path.join(shaders_root, './build/')

EXTS = ['.glsl', '.frag', '.vert']

if __name__ == '__main__':
    for root, dirs, files in os.walk(shaders_root):
        if os.path.relpath(root, shaders_root).startswith('build'):
            continue

        for file in files:
            ext = os.path.splitext(file)[-1]
            if ext not in EXTS:
                continue

            srcpath = os.path.normpath(os.path.join(root, file))
            relpath = os.path.relpath(srcpath, shaders_root)

            dstpath = os.path.normpath(os.path.join(output_dir, relpath))
            # dstpath = f"{dstpath}.spv"
            dstdir = os.path.dirname(dstpath)
            print(f'processing {relpath} to {os.path.relpath(dstpath, shaders_root)}...')


            if not os.path.exists(dstdir):
                os.makedirs(dstdir)

            command = [glslc_path, srcpath, '-o', dstpath]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise CompileError(f'Failed to compile {srcpath}:\n{result.stderr.decode()}')
            else:
                print(f'{relpath} compiled successfully.')
