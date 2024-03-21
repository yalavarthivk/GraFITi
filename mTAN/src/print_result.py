import numpy as np
import glob
import pdb
import re
import math

data = ['physionet2012']
fold = [0, 1, 2, 3, 4]
mode = ['mTAN']
hypers = [100,101, 102, 103, 104]
for data1 in data:
	for mode1 in mode:
		best_val = 10e8
		test_result = dict()
		val_result = dict()
		for runs in hypers:
			if mode1 == 'gratif':
				trainfn = 'train_gratif.py'
			elif mode1 == 'hetvae':
				trainfn = 'train.py'

			dir = '/home/yalavarthi/gratif/mTAN/src/results/'
			p = 0
			test_result[runs] = []
			val_result[runs] = []
			for fold in [0,1,2,3,4]:
				#pdb.set_trace()
				name = glob.glob(dir +data1+"/"+data1+'-'+mode1+"*ct-36-ft-0*"+str(fold)+"-"+str(runs)+"*.log")
				for coeff in name:
					c = []
					for line in open(coeff,'r'):
						c.append(line)
					# pdb.set_trace()
					temp = re.compile(r'\d+(?:\.\d*)')
					res = [ele for ele in c[-1].split(' ') if temp.match(ele)]
					if res == []:
						val_result[runs].append(np.nan)
						test_result[runs].append(np.nan)
					else:
						val_result[runs].append(float(res[0]))
						test_result[runs].append(float(res[1]))
			print(np.mean(val_result[runs]), np.mean(test_result[runs]))
			# pdb.set_trace()
			if (best_val > np.mean(val_result[runs])) & (val_result[runs] != None) & (np.nan not in val_result[runs]):
				if np.nanmean(val_result[runs]) >= 0:
					best_val = np.nanmean(val_result[runs])
					best_test = np.nanmean(test_result[runs])
					best_setup = runs
		print(data1, mode1, best_setup, str('$'+str(np.around(np.mean(test_result[best_setup]), 3))+' \pm '+str(np.around(np.std(test_result[best_setup]), 3))+'$'))
