import os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=="__main__":
	checkpoint_dir = sys.argv[1]
	check_g = [x for x in os.listdir(checkpoint_dir) if 'G' in x]
	check_g = sorted(check_g ,key= lambda x: int(x.split('.')[0].split('_')[-1]))
	print check_g
	check_d = [x for x in os.listdir(checkpoint_dir) if 'D' in x]
	check_d = sorted(check_d ,key= lambda x: int(x.split('.')[0].split('_')[-1]))

	#check_n = [x for x in os.listdir(checkpoint_dir) if 'ntalk' in x]
	#check_n = sorted(check_n ,key= lambda x: int(x.split('.')[0].split('_')[-1]))
	
	total_g = np.array([])
	total_d = np.array([])
	total_n = np.array([])
	
	content = open(checkpoint_dir+"errG_experiment1_100.txt").read().rstrip()
	data = content.split(',')[:-1]
	data = np.array([float(x) for x in data])
	total_g = np.concatenate((total_g,data))	

	content = open(checkpoint_dir+check_g[-1]).read().rstrip()
	data = content.split(',')[:-1]
	data = np.array([float(x) for x in data])
	total_g = np.concatenate((total_g,data))
        
	content = open(checkpoint_dir+"errD_experiment1_100.txt").read().rstrip()
        data = content.split(',')[:-1]
        data = np.array([float(x) for x in data])
        total_d = np.concatenate((total_d,data)) 
      
	content = open(checkpoint_dir+check_d[-1]).read().rstrip()
	data = content.split(',')[:-1]
	data = np.array([float(x) for x in data])
	total_d = np.concatenate((total_d,data))
	
	#content = open(checkpoint_dir+check_n[-1]).read().rstrip()
	#data = content.split(',')[:-1]
	#data = np.array([float(x) for x in data])
	#total_n = np.concatenate((total_n,data))

	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	#plt.title('Loss over Time G & D & N')
	plt.title('Loss over Time G & D')
	l1, = plt.plot(np.arange(len(total_g)),total_g)
	l2, = plt.plot(np.arange(len(total_d)),total_d)
	#l3, = plt.plot(np.arange(len(total_n)),total_n)
	#plt.legend([l1,l2,l3],['G Loss','D Loss', 'N loss'])
	plt.legend([l1,l2],['G Loss','D Loss'])
	#plt.show()
	plt.savefig(checkpoint_dir + "plot.png")
