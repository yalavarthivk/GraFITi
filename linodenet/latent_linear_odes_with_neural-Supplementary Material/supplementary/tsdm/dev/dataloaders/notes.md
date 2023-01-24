The whole start-to-end data pipeline looks as follows:

Modus Operandi: small data-set, everything in-memory

1. Raw Dataset (numpy/pandas/xarray/etc.)
2. Perform task specific pre-processing.
3. Perform the model specific encoding
4. Sample a batch
   - batch should consist of model_inputs, model_targets, true_targets
   - targets are model/loss specific preprocessed targets from raw dataset, for instance
     one-hot encoded for classification task
5. compute outputs `outputs = model(*inputs)`
   - outputs should have the same shape as targets ?!?
   - but then the model needs to know task specific things!
   - but most models need to know these during initialization anyway.
   - but our model here doesn't and that makes it special.
   - alternatively another encoding/decoding layer is necessary.
6. compute predictions
   - `predictions = ???(outputs)`
   - convert to "true" targets
   - i.e. revert any encoding/preprocessing steps
7. Question.
   - can we make sure that the model doesn't see any leaked information?
   - Since the encoder is regular

Pre-Encoder:

- namedtuple[TimeTensor] -> namedtuple[TimeTensor]
- tuple[StaticTensor] -> tuple[StaticTensor]

Problem:

- Certain encodings like One-Hot Endoding might create more columns in the target.
- we still need to find the target column name, to mask it for future time stamps!

But even this is kind of cheating, since models that do not act auto-regressively might
far future datapoints to increase accuracy!
=> would need to feed things 1 by 1, slow!
=> Instead: Trust the user?!?

Modus Operandi: large data-set, stream from disk
