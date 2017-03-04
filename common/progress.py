import sys
import time

def progressbar(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()

if __name__ == '__main__' :
    for i in range(1000):
        time.sleep(0.01)

        j = i - (int(i/ 500)*500)
        progressbar(j, (500-1))
        
        if i == 499 :
            print('')
