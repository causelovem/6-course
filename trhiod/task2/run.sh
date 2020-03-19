# если сервис не включён
# sudo service rabbitmq-server start

# проверить очереди
# sudo rabbitmqctl list_queues


# перезагружаем сервер
sudo rabbitmqctl stop_app
sudo rabbitmqctl reset
sudo rabbitmqctl start_app

# запускаем программу
python3 start.py 3 2


# послать сообщение 'snapshot' процессу 0
# python3 send.py 10 SNAPSHOT
