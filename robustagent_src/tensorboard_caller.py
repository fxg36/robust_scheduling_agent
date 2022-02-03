from tensorboard import program
import os
import hyperparam


path = os.path.join(os.path.dirname(__file__), hyperparam.TENSORBOARD_LOG_PATH)
subdirs = list(os.walk(path))
subdirs.pop(0)
subdirs = sorted(subdirs, key=lambda x: x[0])

sublog = subdirs[-1][0]
logpath = sublog #path + '1'
print(f'\n\n\nOpen tensorboard log: {logpath}\n==============================')
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logpath])
#tb.configure(argv=[None])

url = tb.launch()
print(f"\n\nTensorflow listening on: {url}")
input('\n\n\nPress anything to stop')