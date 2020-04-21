import pika
import sys
import numpy as np
import config as cfg


if len(sys.argv) != 2:
        print('ERROR: Check your command string')
        print('Usage: python3 schedulerRabbit.py <hostCnt>')
        sys.exit(-1)

hostCnt = int(sys.argv[1])

# подключение к кролику
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()
queue = 'queueSch'


maxNeedTmp = []
i = 0
while i != hostCnt:
    method, _, msg = channel.basic_get(queue=queue)
    if method:
        channel.basic_ack(delivery_tag=method.delivery_tag)
        i += 1
        tmp = msg.split(b'#')
        maxNeedTmp.append((eval(tmp[0]), int(tmp[1])))

maxNeedTmp = sorted(maxNeedTmp, key=lambda x: x[1])

maxNeed = []
for row in maxNeedTmp:
    maxNeed.append(row[0])
maxNeed = np.array(maxNeed)

maxAvailable = np.array(cfg.maxAvailable)

allocalion = None
if len(cfg.initialAllocalion):
    allocalion = np.array(cfg.initialAllocalion)
else:
    allocalion = np.random.randint(low=0, high=3, size=maxNeed.shape)

available = maxAvailable - allocalion.sum(axis=0)

if np.all(available >= 0) == False:
    print('Wrong available:', available)
    sys.exit(-1)

if np.all(maxNeed.max(axis=0) <= maxAvailable) == False:
    print('Wrong maxNeed:\n', maxNeed)
    sys.exit(-1)

need = maxNeed - allocalion

if need.min() < 0:
    print('Wrong need:\n', need)
    sys.exit(-1)

priority = cfg.priority

if priority > hostCnt:
    print('Wrong priority:', priority)
    sys.exit(-1)

print('Available:', available)
print('Allocalion\n', allocalion)
print('Need\n', need)
print('Priority:', priority)
print('\n\n')


# def findNext(done, available):
def findNext():
    global done, available
    availableStart = available.copy()
    doneStart = done.copy()

    for i in range(need.shape[0]):
        if i in done:
            continue
        row = need[i]
        if np.all(row <= available) == True:
            done.append(i)
            available += allocalion[i]
            # findNext(done, available)
            findNext()

        if len(done) == hostCnt:
            doneAll.append(done)
            # print(availableStart, doneStart, done)

        done = doneStart.copy()
        available = availableStart.copy()
    return None


doneAll = []
done = []
availableStart = available.copy()

for i in range(need.shape[0]):
    row = need[i]
    if np.all(row <= available) == True:
        done.append(i)
        available += allocalion[i]
        # findNext(done, available)
        findNext()

    if len(done) == hostCnt:
        doneAll.append(done)

    done = []
    available = availableStart.copy()

print()
print('Safe! :)' if len(doneAll) else 'Not safe! :(')
if len(doneAll):
    minIndex = hostCnt
    for seq in doneAll:
        newMin = seq.index(priority)
        if newMin < minIndex:
            minIndex = newMin

    for seq in doneAll:
        if seq.index(priority) == minIndex:
            print('>', seq, '<')
        else:
            print(seq)
    # print(doneAll)
