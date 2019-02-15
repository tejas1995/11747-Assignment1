The project consists of the following code files:

1. classifier.py
   In order to run the learned embeddings model, uncomment line 13, and comment line 14 and 15.
   In order to run the pretrained, single channel model, uncomment line 14, and comment line 13 and 15. If you wish to fix the embedings, when the variable model=CNN(...) is created, it requires an additional parameter 'fixed_embeds=True'
   In order to run the multichannel model, uncomment line 15, and comment line 13 and 14.

2. cnn_model.py
   Defines the CNN model for the learned embeddings model

3. cnn_pretrained.py
   Defines the CNN model for the pretrained, single channel model

4. cnn_pretrained_multichannel.py
   Defines the CNN model for the multichannel model
