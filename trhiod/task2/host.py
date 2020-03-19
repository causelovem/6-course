import time
import pika
import sys
import random as rnd

if __name__ == '__main__':
    # проверка командной строки
    if len(sys.argv) != 4:
        print('ERROR: Check your command string')
        print('Usage: python3 host.py <hostCnt> <hostId> <sleepSec>')
        sys.exit(-1)

    # параметры программы
    hostCnt = int(sys.argv[1])
    hostId = int(sys.argv[2])
    sleepSec = int(sys.argv[3])
    queueTemp = 'queue{}{}'

    # рассчитываем номера соседей
    rightNeighbourID = (hostId + 1) % hostCnt
    leftNeighbourID = (hostId - 1) % hostCnt

    # формируем списки очередей откуда принимать и куда отправлять
    inQueues = [queueTemp.format(rightNeighbourID, hostId), queueTemp.format(leftNeighbourID, hostId)]
    outQueues = [queueTemp.format(hostId, rightNeighbourID), queueTemp.format(hostId, leftNeighbourID)]

    # список для маркировки очередей, откуда пришло сообщение 'snapshot'
    inQueuesEmpty = []
    # список для хранения сообщений, которые пришли во время создания снимка
    recordQueue = []

    # подключение к кролику
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    # начальная сумма
    # localSum = rnd.randint(15, 500)
    localSum = 100
    # сам снимок (без каналов)
    snap = False

    print('[Host {}] Started I have {}'.format(hostId, localSum))

    # основной цикл работы
    while 1:
        time.sleep(sleepSec)

        # отправка
        if snap is False:
            # по всем исходящим каналам отправляем случайную сумму
            for que in outQueues:
                msg = rnd.randint(0, localSum)
                localSum -= msg
                channel.basic_publish(exchange='', routing_key=que, body=str(msg))
                print('[Host {}] Paid {} to {}. Now I have {}.'.format(hostId, int(msg), que[-1:], localSum))

        time.sleep(sleepSec)

        # приём
        for que in inQueues:
            # если канал не 'пустой'
            if que not in inQueuesEmpty:
                method, _, msg = channel.basic_get(queue=que)
                if method:
                    channel.basic_ack(delivery_tag=method.delivery_tag)

                    # обработка управляющего приложения, которое запускает процедуру создания снимка
                    if msg.upper() == b'MAKESNAP':
                        print('[Host {}] Starting snapshot.'.format(hostId))
                        # сохраняем своё состояние
                        snap = localSum
                        # по всем исходящим каналам посылаем сообщение 'snapshot'
                        for qu in outQueues:
                            channel.basic_publish(exchange='', routing_key=qu, body='SNAPSHOT')
                    elif msg.upper() == b'SNAPSHOT':
                        # если сообщение 'snapshot' пришло впервый раз
                        if snap is False:
                            print('[Host {}] Received first snapshot message.'.format(hostId))
                            # сохраняем своё состояние
                            snap = localSum
                            # помечаем канал, как пустой
                            inQueuesEmpty.append(que)
                            # по всем исходящим каналам посылаем сообщение 'snapshot'
                            for qu in outQueues:
                                channel.basic_publish(exchange='', routing_key=qu, body='SNAPSHOT')
                        else:
                            # если сообщение 'snapshot' пришло не первый раз
                            print('[Host {}] Received not first snapshot message.'.format(hostId))
                            # помечаем канал, как пустой
                            inQueuesEmpty.append(que)
                    else:
                        # если сейчас создаётся снимок, то сохраняем все пришедшие сообщения и не учитываем их
                        if snap is not False:
                            print('[Host {}] Recorder not snapshot message while snapshot {}.'.format(hostId, int(msg)))
                            recordQueue.append(int(msg))
                        else:
                            # иначе, учитываем
                            localSum += int(msg)
                            print('[Host {}] Get {} from {}. Now I have {}.'.format(hostId, int(msg), que[-2:-1], localSum))

        # если все входящие каналы помечены, как пустые, то снимок создан
        if len(inQueues) == len(inQueuesEmpty):
            # учитываем все сохранённые сообщения
            localSum += sum(recordQueue)
            # очищаем системные переменные
            recordQueue = []
            inQueuesEmpty = []
            print('[Host {}] My snapshot is {}. Now I have {}.'.format(hostId, snap, localSum))
            snap = False

    # закрываем соединение
    connection.close()
