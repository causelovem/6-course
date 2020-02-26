import sys
import subprocess
import pika

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('ERROR: Check your command string')
        print('Usage: python3 start.py <hostCnt> <sleepSec>')
        sys.exit(-1)

    hostCnt = int(sys.argv[1])
    sleepSec = int(sys.argv[2])

    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    procList = []
    for i in range(hostCnt):
        proc = subprocess.Popen('python3 host.py {} {} {}'.format(hostCnt, i, sleepSec), shell=True)
        procList.append(proc)

    for i in range(len(procList)):
        print('[Start] Host {} started'.format(i))

    for i in range(len(procList)):
        procList[i].wait()

    connection.close()
