import os
from pandaset import DataSet
import sys
init = 0
print('the transfered data would be stored in ../"root"/padsToKit dir')
print('please input the pandaset root path:')
pd_root = input()
dataset = DataSet(pd_root)
sequence = dataset.sequences()
script = os.path.join(os.getcwd(),'transferPdToKit.py') 
for i in sequence:
    os.system(sys.executable+' '+script+' %s %s %s %s' %(pd_root,i,init,init*80))
    init = init+1