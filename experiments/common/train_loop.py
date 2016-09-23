import os
from helper_fxns import *
from time import time
import sys
from data import *

def iterate(X_train, bs=32):
    b = 0
    while True:
        if b*bs >= X_train.shape[0]:
            break
        yield X_train[b*bs:(b+1)*bs]
        b += 1
    
        
  

        
    

    
def train(cfg,
        num_epochs,
        out_folder,
        sched={},
        batch_size=128,
        model_folder="/storeSSD/cbeckham/nersc/models/",
        tmp_folder="tmp",
        training_days=[1,20],
        validation_days=[345,365],
        time_chunks_per_example=1,
        step_size=20,
        data_dir="/storeSSD/cbeckham/nersc/big_images/",
        dataset="climate",
        img_size=128,
        resume=None,
        debug=True,
        time_steps=8):
    


    def get_iterator(name, batch_size, data_dir, start_day, end_day, img_size, time_chunks_per_example, step_size, time_steps):
        # for stl10, 'days' and 'data_dir' does not make
        # any sense
        assert name in ["climate", "stl10"]
        if name == "climate":
            return data_iterator(batch_size, data_dir, start_day=start_day, end_day=end_day, img_size=img_size, time_chunks_per_example=time_chunks_per_example, step_size=step_size, time_steps=time_steps)
        elif name == "stl10":
            return stl10.data_iterator(batch_size)


    def prep_batch(X_batch):
            if dataset == "climate":
                if time_chunks_per_example == 1:
                    # shape is (32, 1, 16, 128, 128), so collapse to a 4d tensor
                    X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[2], X_batch.shape[3], X_batch.shape[4])
                else:
                    # right now it is: (bs, time, nchannels, height, width)
                    # needs to be: (bs, nchannels, time, height, width)
                    X_batch = X_batch.reshape(X_batch.shape[0], X_batch.shape[2], X_batch.shape[1], X_batch.shape[3], X_batch.shape[4])
            else:
                pass # nothing needs to be done for stl-10
            return X_batch


    def plot_image(img_composite):
            if dataset == "climate":
                for j in range(0,32):
                    plt.subplot(8,4,j+1)
                    if time_chunks_per_example > 1:
                        plt.imshow(img_composite[j][0])
                    else:
                        plt.imshow(img_composite[j])
                    plt.axis('off')
            elif dataset == "stl10":
                for j in range(0,6):
                    plt.subplot(3,2,j+1)
                    plt.imshow(img_composite[j])
                    plt.axis('off')
            plt.savefig('%s/%i.png' % (out_folder, epoch))
            pyplot.clf()  
        
    # extract methods
    train_fn, loss_fn, out_fn, l_out = cfg["train_fn"], cfg["loss_fn"], cfg["out_fn"], cfg["l_out"]
    lr = cfg["lr"]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    #if resume != None:
    #    

    
    logger = get_logger(out_folder)
    num_train = (training_days[1] - training_days[0] + 1) * time_steps * (((1152 - img_size) / step_size) + 1) * (((768 - img_size) / step_size) + 1)
    logger.info("train size: %i"%(num_train))
    for layer in get_all_layers(l_out):
        logger.info(str(layer) + ' ' +  str(layer.output_shape))
    for epoch in range(0, num_epochs):
        t0 = time()
        # learning rate schedule
        if epoch+1 in sched:
            lr.set_value( floatX(sched[epoch+1]) )
            logger.info("changing learning rate to: %f\n" % sched[epoch+1])
        train_losses = []
        train_losses_det = []
        valid_losses = []
        first_minibatch = True
        # TRAINING LOOP
        for X_train, y_train in get_iterator(dataset, batch_size, data_dir, start_day=training_days[0], end_day=training_days[1],
                                        img_size=img_size,step_size=step_size,time_chunks_per_example=time_chunks_per_example, time_steps=time_steps):
            X_train = prep_batch(X_train)
            if first_minibatch:
                X_train_sample = X_train[0:1]
                first_minibatch = False
            this_loss, this_loss_det = train_fn(X_train)
            if debug:
                logger.info("iteration loss: %f" %this_loss)
            train_losses.append(this_loss)
            train_losses_det.append(this_loss_det)
        # if debug:
        #     mem = virtual_memory()
        #     print mem
        # VALIDATION LOOP
        for X_valid, y_valid in get_iterator(dataset, batch_size, data_dir, start_day=validation_days[0], end_day=validation_days[1],
                                              img_size=img_size, time_chunks_per_example=time_chunks_per_example, step_size=step_size, time_steps=time_steps):
            X_valid = prep_batch(X_valid)
            val_loss = loss_fn(X_valid)
            if debug:
                logger.info("iteration val loss: %f" %val_loss)
            valid_losses.append(val_loss)
        # DEBUG: visualise the reconstructions
        img_orig = X_train_sample
        img_reconstruct = out_fn(img_orig)
        img_composite = np.vstack((img_orig[0],img_reconstruct[0]))
        plot_image(img_composite)
        # STATISTICS
        time_taken = time() - t0

        logger.info("epoch %i of %i \n time: %f \n train loss: %6.3f \n val loss: %6.3f" % (epoch+1, num_epochs, time_taken, np.mean(train_losses), np.mean(valid_losses)))

        # save model at each epoch
        if not os.path.exists("%s/models/" % (out_folder)):
            os.makedirs("%s/models/" % (out_folder))
        with open("%s/models/%i.model" % (out_folder, epoch), "wb") as g:
            pickle.dump( get_all_param_values(cfg["l_out"]), g, pickle.HIGHEST_PROTOCOL )