import numpy as np
import random

clust_script = "/home/yalavarthi/gratif/gratif/sbatch_file.sh"
run_hp = open('run_hp.sh','w')

ct = 36
ft = 12
for dataset in ['mimiciv']:
	basefolder = '/home/yalavarthi/gratif/gratif/'
	for i in range(100, 105):
		n_layers = random.choice([1,2,3])
		imab_dim = random.choice([32, 64, 128])
		enc_num_heads = random.choice([1,2, 4])
		for fold in [0, 1, 2, 3, 4]:
			batch = 'sbatch --job-name='+dataset+'-gratif --output='+basefolder+'results/'+dataset+'/'+dataset+'-gratif-'+'ct-'+str(ct)+'-ft-'+str(ft)+'-'+str(fold)+'-'+str(i)+'-%A.log --error='+basefolder+'results/'+dataset+'/'+dataset+'-gratif-'+'ct-'+str(ct)+'-ft-'+str(ft)+'-'+str(fold)+'-'+str(i)+'-%A.err --mail-type=FAIL --mail-user=yalavarthi@ismll.de '+clust_script+' train_gratif.py --epochs 200 --learn-rate 0.001 --batch-size 64 --attn-head '+str(enc_num_heads)+' --latent-dim '+str(imab_dim)+' --nlayers '+str(n_layers)+' --dataset '+dataset+'  --fold '+ str(fold) +' -ct '+str(ct)+' -ft '+ str(ft) +'\n'
			run_hp.write(batch)
run_hp.close()