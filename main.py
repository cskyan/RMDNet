import subprocess
import sys


sys.path.append('.../RMDNet.py')

class RunCmd(object):
  def cmd_run(self, cmd):
    self.cmd = cmd
    subprocess.call(self.cmd, shell=True)

for i in range(1):
    a = RunCmd()
    a.cmd_run('CUDA_VISIBLE_DEVICES=0 python RMDNet.py \
    --posi=.../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa \
    --nega=.../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa \
    --train=True --n_epochs=1 ')

    for j in range(1):
        a = RunCmd()
        a.cmd_run('CUDA_VISIBLE_DEVICES=0 python RMDNet.py \
        --testfile=.../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa \
        --nega=.../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.negatives.fa \
        --predict=True ')

