import pika
import sys

if len(sys.argv) < 3:
    print('ERROR: Check your command string')
    print('Usage: python3 send.py <hostNumber> <message>')
    sys.exit(-1)

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='queue{}'.format(sys.argv[1]), auto_delete=True)

message = ' '.join(sys.argv[2:]) or "Hello World!"
channel.basic_publish(
    exchange='',
    routing_key='queue{}'.format(sys.argv[1]),
    body=message)
print(" [TO NODE {}] Sent {}".format(sys.argv[1], message))
connection.close()
