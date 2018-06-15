

import matplotlib.pyplot as plt
import time
import tqdm


def test():
    # total = 10
    p = progress()
    for i in range(10):
        time.sleep(0.1)
        print(i)
        p(i)


def progress():
    fg = plt.figure('progress')
    ax = fg.add_axes([0, 0, 1, 1])
    ax.hist([50]*100, bins=20, range=(1, 100), orientation=u'horizontal')
    # ax.get_yaxis().set_visible(False)
    fg.show()

    def move(step):
        ax.hist([50] * step, bins=20, range=(1, 100), orientation=u'horizontal')
        # ax.set_title(str(step) + '%')
        fg.text(0.1, 0.2, 'progress...')
        fg.show()

    return move


def test_tqdm():
    text = ''
    for char in tqdm.tqdm([chr(x) for x in range(ord('a'), ord('a')+26)]):
        text = text + char
        time.sleep(0.1)
        print(text)
