import os
counter=0
names=[]
for subdir, dirs, files in os.walk('aData'):
    if counter == 0:
        counter+=1
    else:
        names.append(subdir[6:])

for name in names:
    print(name)
