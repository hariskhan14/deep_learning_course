
## Images to add in report

### Attention maps
The transformer module (Fig. 1b) uses a dual-convolutional window self-attention (Conv-wSA) mechanism that operates on both high- and low-resolution versions of the original feature maps to produce an attention map. 


### Evidence Map
To enhance interpretability, we replaced the FCL with a convolutional classifier, referred to as the class evidence layer. This layer leverages spatial information to produce class-wise evidence maps (Fig. 1), where each pixel reflects the local contribution of input regions to the final prediction. Following classification, the evidence maps are upsampled and overlaid on the input image for visualization purposes (Fig. 1d).


