from tensorboard import program
import os
import hyperparam


def load():
    path = os.path.join(os.path.dirname(__file__), hyperparam.TENSORBOARD_LOG_PATH)
    subdirs = list(map(lambda x: x[0],os.walk(path)))
    subdirs.pop(0)

    i = 0
    for d in subdirs:
        print(f"[{i}] {d.split('/')[-1]}")
        i = i+1
    print(f"[x] REFRESH")

    n = str(input('\n\n\nLog numer to open: '))
    if n.lower() == "x":
        load()
        
    logdir = subdirs[int(n)]

    print(f'\n\n\nOpen tensorboard log: {logdir}\n==============================')
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir])
    #tb.configure(argv=[None])

    url = tb.launch()
    print(f"\n\nTensorflow listening on: {url}")

    n = str(input('\n\n\n[x] to reload or anything else to stop'))
    if n.lower() == "x":
        load()



load()