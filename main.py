# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import csv

import numpy as np

from dtm import Dtm
from image import Image


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    with open("imageinfo.csv", "r") as f:
        reader = csv.reader(f, delimiter="\t")
        keys, vals = reader
        img = Image({k: v for k, v in zip(keys, vals)})
        dtm = Dtm()
        dtm.image = img

        dtm.process_dtm()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
