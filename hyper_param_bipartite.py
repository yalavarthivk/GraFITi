import numpy as np
import random

clust_script = "/home/yalavarthi/gratif/gratif/sbatch_file.sh"
run_hp = open('run_hp.sh','w')

ct = 24
ft = 24
for dataset in ['physionet2012', 'mimiciii', 'mimiciv']:
	basefolder = '/home/yalavarthi/gratif/gratif/'
	for i in range(100, 105):
		n_layers = random.choice([1,2,3,4])
		attn_head = random.choice([1,2,4])
		n_ref_points = random.choice([8,16,32,64])
		# tim_dims = random.choice([16,32,64,128,256])
		latent_dim = random.choice([16,32,64,128])
		for fold in [0, 1, 2, 3, 4]:
			batch = 'sbatch --job-name='+dataset+'-bipartitegraph --output='+basefolder+'results/'+dataset+'/'+dataset+'-bipartitegraph-'+'ct-'+str(ct)+'-ft-'+str(ft)+'-'+str(fold)+'-'+str(i)+'-%A.log --error='+basefolder+'results/'+dataset+'/'+dataset+'-bipartitegraph-'+'ct-'+str(ct)+'-ft-'+str(ft)+'-'+str(fold)+'-'+str(i)+'-%A.err --mail-type=FAIL --mail-user=yalavarthi@ismll.de '+clust_script+' train_bipartitegraph.py --epochs 200 --learn-rate 0.001 --batch-size 128 --attn-head '+str(attn_head)+' --latent-dim '+str(latent_dim)+' --n-ref-points '+str(n_ref_points)+' --tim-dims '+str(latent_dim)+' --nlayers '+str(n_layers)+' --dataset '+dataset+'  --fold '+ str(fold) +' -ct '+str(ct)+' -ft '+ str(ft) +'\n'
			run_hp.write(batch)
run_hp.close()