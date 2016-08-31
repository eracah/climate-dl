# climate-dl

Rough notes:
Semi-Supervised
diff)
* Baseline (patches)
    * Hidden code is vector
      * Unlabelled examples -> we try to reconstruct patch from code
      * Labelled examples -> we pass code thru classification network and penalize via cross entropy
  * Training Data
     * features: num_time_chunks x 16 x 128 x 128 patches 
     * labels (for labeled data): what type of event is in the patch (we assume one or none per patch?)
     * labels (for unlabeled data): features
  * Network Architecture
     * 3D fully convolutional autoencoder
     * Filters are 16x k x k and convolved in the x,y as well as z (time) direction
      * I-C-C-C-..-FC-DC-DC-DC-,.. -> mean-squared-error
     * FC classifier
       * FC-FC-Softmax -> cross entropy
     * Layer Types:
       * 3D Convolutional (C)
       * 3D Deconvolutional (DC)
       * FC (only at bottleneck) (F)
     * Training Algo
       * train as standard autoencoder for a while to get decent encoder
       * train as semi-sup ae
         * for labelled images, extract feature vector, put thru classifier and backprop using crossentropy loss
          * for unlabelled images (and maybe labelled as well), put thru whole autoencoder and backprop using mean squared error
   * Potential Bells and Whistles to Increase Complexity
     * use unsupervised criterion on labelled data as well
     * exploit some temporal coherence prior
       * assume features of interest change slowly in time?
     * add in spatial transformer units
     * put reconstructions thru discriminator of a GAN to get loss instead of mean squared error
     * 
  
        





* End goal (full image)
  * Semi-supervised autoencoder on full image w/ time
    * Hidden code is a spatial map
      * Unlabelled examples -> we try to reconstruct full image from code
      * Labelled examples -> we compare to real semantic segmentation mask and penalize based on how far the code is from that

