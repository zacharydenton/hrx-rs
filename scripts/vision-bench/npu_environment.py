"""Locate libraries from an explicitly selected Ryzen AI Python installation."""
import json
from pathlib import Path
import subprocess


def environment(base, python, output, xrt_root, extra_dirs):
    # This child only uses Python's standard library. No model package discovery.
    probe = ('import json,site,sys;print(json.dumps(dict(prefix=sys.prefix,'
             'base=sys.base_prefix,sites=site.getsitepackages())))')
    info = json.loads(subprocess.check_output([str(python), '-c', probe], text=True))
    paths = [Path(p) for p in extra_dirs]
    paths.extend(xrt_root / name for name in ('lib', 'lib64', 'lib/x86_64-linux-gnu'))
    for site in info['sites']:
        paths.extend(Path(site) / name for name in ('lib/lnx64.o', 'lib/lnx64.o/tools/peano/lib',
                     'ryzenai_dynamic_dispatch/lib', 'voe/lib'))
    paths.extend(Path(info['prefix']) / name for name in ('deployment/lib', 'onnxruntime/lib'))
    paths.append(Path(info['base']) / 'lib')
    paths = list(dict.fromkeys(p.resolve() for p in paths if p.is_dir()))
    if not any((p / 'libxrt_coreutil.so.2').is_file() for p in paths):
        raise RuntimeError('system XRT not found; set --xrt-root or --npu-lib-dir')
    # Arch provides the wide-character ncurses ABI; keep its loader alias local.
    if not any((p / 'libncurses.so.6').exists() for p in paths):
        wide = next((p / 'libncursesw.so.6' for p in paths if (p / 'libncursesw.so.6').exists()), None)
        if wide is None:
            raise RuntimeError('libncurses.so.6 missing; provide its directory with --npu-lib-dir')
        compatibility = output / 'runtime-libs'
        compatibility.mkdir()
        (compatibility / 'libncurses.so.6').symlink_to(wide.resolve())
        paths.insert(0, compatibility.resolve())
    return dict(base, XILINX_XRT=str(xrt_root), LD_LIBRARY_PATH=':'.join(map(str, paths)))
