import time
import pika
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('ERROR: Check your command string')
        print('Usage: python3 host.py <hostCnt> <hostId> <sleepSec>')
        sys.exit(-1)

    hostCnt = int(sys.argv[1])
    hostId = int(sys.argv[2])
    sleepSec = int(sys.argv[3])
    queueTemp = 'queue{}'

    leaderId = 0
    neighbourID = (hostId + 1) % hostCnt

    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=queueTemp.format(hostId), auto_delete=True)
    channel.confirm_delivery()

    print('[Host {}] Started'.format(hostId))

    def publish(sendId, msg):
        try:
            global neighbourID
            if sendId != hostId:
                channel.basic_publish(exchange='', routing_key=queueTemp.format(sendId),
                                      body=msg, mandatory=True)
                neighbourID = sendId
            else:
                print('[Host {}] I can not send message to myself :('.format(hostId))
        except pika.exceptions.UnroutableError:
            publish((sendId + 1) % hostCnt, msg)

    def beginElection(sendId):
        try:
            global neighbourID
            if sendId != hostId:
                channel.basic_publish(exchange='', routing_key=queueTemp.format(sendId),
                                      body='Election [{}]'.format(hostId), mandatory=True)
                neighbourID = sendId
            else:
                print('[Host {}] All other hosts killed :('.format(hostId))
                sys.exit(-1)
        except pika.exceptions.UnroutableError:
            beginElection((sendId + 1) % hostCnt)

    def callback(ch, method, properties, body):
        global leaderId
        time.sleep(sleepSec)

        print('[Host {}] Received {}'.format(hostId, body))
        ch.basic_ack(delivery_tag=method.delivery_tag)

        bodySplit = body.split()
        msgType = bodySplit[0].upper()

        if msgType == b'SHOW':
            print('[Host {}] Leader Id is {}'.format(hostId, leaderId))
            return
        elif msgType == b'BEGIN':
            beginElection(neighbourID)
            return
        elif msgType == b'STOP':
            sys.exit(-1)
        elif msgType == b'ELECTION':
            electionList = eval(bodySplit[1])

            if hostId in electionList:
                leaderId = min(electionList)
                publish(neighbourID, 'Leader {}'.format(leaderId))
            else:
                electionList.append(hostId)
                publish(neighbourID, 'Election {}'.format(str(electionList).replace(' ', '')))
            return
        elif msgType == b'LEADER':
            newLeaderId = int(bodySplit[1])

            if leaderId != newLeaderId:
                leaderId = newLeaderId
                publish(neighbourID, 'Leader {}'.format(newLeaderId))
                return

        # time.sleep(1)
        # try:
        #     channel.basic_publish(exchange='', routing_key=queueTemp.format(neighbourID),
        #                             body='Ping', mandatory=True)
        # except pika.exceptions.UnroutableError:
        #     print('Message was returned')
        #     print('[Host {}] Begin ELECTION'.format(hostId))
        #     beginElection((neighbourID + 1) % hostCnt)

    channel.basic_consume(queue=queueTemp.format(hostId), on_message_callback=callback)

    channel.start_consuming()

    connection.close()
