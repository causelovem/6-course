import sys
import subprocess
import pika

if __name__ == '__main__':
    # проверка командной строки
    if len(sys.argv) != 3:
        print('ERROR: Check your command string')
        print('Usage: python3 start.py <hostCnt> <sleepSec>')
        sys.exit(-1)

    # параметры программы
    hostCnt = int(sys.argv[1])
    sleepSec = int(sys.argv[2])

    # подключение к кролику
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()

    # запускаем процессы и инициализируем очереди
    procList = []
    for i in range(hostCnt):
        channel.queue_declare(queue='queue{}{}'.format(i, (i + 1) % hostCnt), auto_delete=True)
        channel.queue_declare(queue='queue{}{}'.format((i + 1) % hostCnt, i), auto_delete=True)

        proc = subprocess.Popen('python3 host.py {} {} {}'.format(hostCnt, i, sleepSec), shell=True)
        procList.append(proc)

    # выводим, что всё ок
    for i in range(len(procList)):
        print('[Start] Host {} started'.format(i))

    # ждём завершения
    for i in range(len(procList)):
        procList[i].wait()

    # закрываем соединение
    connection.close()
