import data
import pdb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import pyplot


out_folder = "out_test"
    
# little test/example
if __name__ == "__main__":

    tot=0
    for x,y in data.data_iterator(batch_size=1, time_chunks_per_example=8, img_size=-1, data_dir="/storeSSD/cbeckham/nersc/big_images/"):
        # x is of the shape (1, 8, 16, 768, 1152)
        fm_idx = 1
        # get rid of the batch index, and choose a specific feature map
        img = x[0][:,fm_idx,:]
        for t in range(0, img.shape[0]):
            plt.imshow(img[t])
            plt.axis('off')
            filename = "%s/%i/%i.png" % (out_folder, fm_idx, tot)
            plt.savefig(filename.zfill(4))
            pyplot.clf()
            tot += 1
    print i
