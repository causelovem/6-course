import pika
import sys

# проверка командной строки
if len(sys.argv) < 3:
    print('ERROR: Check your command string')
    print('Usage: python3 send.py <hostNumber> <message>')
    sys.exit(-1)

# подключение к кролику
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

# очередь, в котору будем посылать сообщение
channel.queue_declare(queue='queue{}'.format(sys.argv[1]), auto_delete=True)

# формируем сообщение
message = ' '.join(sys.argv[2:])

# отправляем
channel.basic_publish(
    exchange='',
    routing_key='queue{}'.format(sys.argv[1]),
    body=message)
print(" [TO NODE {}] Sent {}".format(sys.argv[1], message))

# закрываем соединение
connection.close()
