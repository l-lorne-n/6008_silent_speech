"""Exercise current publishable sources in a new path without private inputs.

This checks path independence using the already installed interpreter/dependencies;
it is not a substitute for a fresh online dependency installation.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT/'tests/output')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='portable_', dir=args.output_root))
    checkout = out/'clone check 中文'
    checkout.mkdir()
    files = subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',
        '-c','core.excludesFile=.git/info/exclude','ls-files','-co','--exclude-standard','-z'], cwd=ROOT)
    names = sorted(set(s.decode('utf-8') for s in files.split(b'\0') if s))
    for name in names:
        source = ROOT/name
        if source.is_file():
            dest = checkout/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source,dest)
    for name in ('.runtime','.python','recordings','models','vocabulary.json'):
        assert not (checkout/name).exists(), name
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONUTF8='1')
    logs = {}
    def run(name, args):
        completed = subprocess.run(args,cwd=checkout,env=env,text=True,encoding='utf-8',
                                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=150)
        logs[name] = dict(returncode=completed.returncode, output=completed.stdout)
        (out/(name+'.log')).write_text(completed.stdout,encoding='utf-8')
        assert completed.returncode == 0, completed.stdout
        print(name+': PASS',flush=True)
    run('environment', [sys.executable, 'check_environment.py'])
    run('unit_tests', [sys.executable, '-m','pytest','-q'])
    run('dark_theme', [sys.executable, '-c', '''
from pathlib import Path
import time
from PySide6 import QtWidgets as W, QtGui
from semg.app import MainWindow, ROOT, configure_fonts
assert ROOT == Path.cwd()
app=W.QApplication([])
configure_fonts(app)
p=QtGui.QPalette()
for group in (p.ColorGroup.Active,p.ColorGroup.Inactive,p.ColorGroup.Disabled):
    for role in (p.ColorRole.Window,p.ColorRole.Base,p.ColorRole.Button): p.setColor(group,role,QtGui.QColor('#202020'))
    for role in (p.ColorRole.WindowText,p.ColorRole.Text,p.ColorRole.ButtonText): p.setColor(group,role,QtGui.QColor('#eeeeee'))
app.setPalette(p)
w=MainWindow();w.resize(1280,900);w.show();app.processEvents()
side=w.findChild(W.QWidget,'sidebar')
assert side.palette().color(p.ColorRole.Window).name() == '#f3f5f4'
assert side.grab().toImage().pixelColor(side.width()-3,10).name() == '#f3f5f4'
out=ROOT/'tests/output';out.mkdir(parents=True,exist_ok=True)
w.grab().save(str(out/'dark_main.png'))
w.resize(1000,720);app.processEvents();w.grab().save(str(out/'dark_main_small.png'))
w.open_lda()
end=time.monotonic()+10
while w.lda_window.busy and time.monotonic()<end:
    app.processEvents();time.sleep(.01)
assert not w.lda_window.busy
assert '扫描完成' in w.lda_window.status.text(),w.lda_window.status.text()
assert w.lda_window.train_picker.catalog == []
w.lda_window.close();w.close()
print('PASS: dark sidebar, small window scroll, relocated root, empty clone LDA')
'''])
    run('vocabulary_gui', [sys.executable,'tests/vocabulary_gui_smoke.py'])
    result = dict(status='PASS', checkout=str(checkout), files=len(names),
                  private_inputs_copied=False, environment_reused=str(sys.executable),
                  fresh_online_install_tested=False, checks=logs)
    (out/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(out),flush=True)


if __name__=='__main__':main()
