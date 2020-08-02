import os

with open('noise-list.txt', 'r') as f:
    lines = f.readlines()
num = 0
for l in lines:
    fn = l.strip('\n')
    os.remove(fn)
    num += 1
    print('{} - removed!'.format(fn))
print('Noise Removal Done! {} images are removed!'.format(num))
