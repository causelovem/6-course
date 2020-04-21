import pika
import sys
import config as cfg

if __name__ == '__main__':
    # проверка командной строки
    if len(sys.argv) != 2:
        print('ERROR: Check your command string')
        print('Usage: python3 host.py <hostId>')
        sys.exit(-1)

    # параметры программы
    hostId = int(sys.argv[1])
    queue = 'queueSch'

    # подключение к кролику
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()

    myNeed = cfg.maxNeed[hostId]

    channel.basic_publish(exchange='', routing_key=queue, body=str(myNeed) + '#' + str(hostId))

    connection.close()
