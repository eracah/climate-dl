# climate-dl

Rough notes:
Semi-Supervised
* End goal (full image)
  * Semi-supervised autoencoder on full image w/ time
    * Hidden code is a spatial map
      * Unlabelled examples -> we try to reconstruct full image from code
      * Labelled examples -> we compare to real semantic segmentation mask and penalize based on how far the code is from that (squared diff)
* Baseline (patches)
  * Semi-sup ae on 128x128 patches
    * Hidden code is vector
      * Unlabelled examples -> we try to reconstruct patch from code
      * Labelled examples -> we pass code thru classification network and penalize via cross entropy

