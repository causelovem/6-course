import sys
import numpy as np
import config as cfg


maxAvailable = np.array(cfg.maxAvailable)
maxNeed = np.array(cfg.maxNeed)

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

print('Available', available)
print('Allocalion\n', allocalion)
# print(maxNeed)
print('Need\n', need)
print('\n\n')

done = []

safe = False

while True:
    safe = False

    for i in range(need.shape[0]):
        if i in done:
            continue
        row = need[i]
        if np.all(row <= available) == True:
            safe = True
            done.append(i)
            available += allocalion[i]
            print('Available', available)

    if (len(done) == maxNeed.shape[0]) or (not safe):
        break

print()
print('Safe! :)' if safe else 'Not safe! :(')
if safe:
    print(done)
