import os

data_size='4_new'

for dataset in ['train','val','test']:
    path='data/data_'+data_size+'/'+dataset+'/Healthy'
    fileObject = open('datatxt/'+dataset+'_'+data_size+'.txt', 'a+')
    for file in os.listdir(path):
        combine = path+'/'+file+ ' '+'0'
        fileObject.write(combine)
        fileObject.write('\n')
    fileObject.close()
    path='data/data_'+data_size+'/'+dataset+'/Unhealthy'
    fileObject = open('datatxt/'+dataset+'_'+data_size+'.txt', 'a+')
    for file in os.listdir(path):
        combine = path+'/'+file+ ' '+'1'
        fileObject.write(combine)
        fileObject.write('\n')
    fileObject.close()

