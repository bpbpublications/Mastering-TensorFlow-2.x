# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import tensorflow as tf

def run_mirrorstrategy():
    # Use a breakpoint in the code line below to debug your script.

    strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    with strategy.scope():
        x = tf.Variable(1.)
        print(x)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run_mirrorstrategy()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
