import subprocess
import sys


sys.path.append('/home/.../DBODL.py')

class RunCmd(object):
  def cmd_run(self, cmd):
    self.cmd = cmd
    subprocess.call(self.cmd, shell=True)

for i in range(1):
    a = RunCmd()
    a.cmd_run('CUDA_VISIBLE_DEVICES=0 python DBODL.py \
    --posi=/home/.../DBODL/GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa \
    --nega=/home/.../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa \
    --train=True --n_epochs=1 ')

    for j in range(1):
        a = RunCmd()
        a.cmd_run('CUDA_VISIBLE_DEVICES=0 python DBODL.py \
        --testfile=/home/.../DBODL/GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa \
        --nega=/home/.../DBODL/GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.negatives.fa \
        --predict=True ')

