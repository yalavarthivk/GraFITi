import numpy as np
import random

clust_script = "/home/yalavarthi/gratif/mTAN/src/sbatch_file.sh"
run_hp = open('run_hp.sh','w')

ct = 36
ft = 0
for dataset in ['ushcn','mimiciii','physionet2012','mimiciv']:
	basefolder = '/home/yalavarthi/gratif/mTAN/src/'
	for i in range(100, 105):
		latent_dim = random.choice([16])
		rec_hidden = random.choice([32,64,128])
		gen_hidden = random.choice([50])
		num_ref_points = random.choice([8, 16, 32, 64, 128])
		for fold in [0, 1, 2, 3, 4]:
			batch = 'sbatch --job-name='+dataset+'-mTAN --output='+basefolder+'results/'+dataset+'/'+dataset+'-mTAN-'+'ct-'+str(ct)+'-ft-'+str(ft)+'-'+str(fold)+'-'+str(i)+'-%A.log --error='+basefolder+'results/'+dataset+'/'+dataset+'-mTAN-'+'ct-'+str(ct)+'-ft-'+str(ft)+'-'+str(fold)+'-'+str(i)+'-%A.err --mail-type=FAIL --mail-user=yalavarthi@ismll.de '+clust_script+' tan_interpolation.py --niters 200 --lr 0.001 --batch-size 64 --latent-dim '+str(latent_dim)+' --rec-hidden '+str(rec_hidden)+' --gen-hidden '+str(gen_hidden)+' --num-ref-points '+str(num_ref_points)+' --dataset '+dataset+'  --fold '+ str(fold) +' -ct '+str(ct)+' -ft '+ str(ft) +'\n'
			run_hp.write(batch)
run_hp.close()