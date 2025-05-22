datasets
========

Mental Model
------------


Datasets vs Tasks
~~~~~~~~~~~~~~~~~

We consider a dataset to be just that - data.
A Task on the other hand is a dataset equipped with a problem formulation. Often times,
datasets are shipped with default tasks; for example in the famous MNIST dataset, a prescribed split
of the data is given into a training set and a test set and the goal is to train a classifier on the
train split with high classification accuracy on the test set.

Here, **we do not split datasets a-priori**. Instead, datasets may implement a `default_task` method
that returns a Task object.


TimeSeries specifics
~~~~~~~~~~~~~~~~~~~~

- TimeSeries -- Single Tensor

  - univariate: ℝ×𝓢
  - multivariate: (ℝ×𝓢₁)⊕(ℝ×𝓢₂)⊕…⊕(ℝ×𝓢ₙ) ≃ ℝ×(𝓢₁⊕𝓢₂⊕…⊕𝓢ₙ)

- TimeSeries -- Multiple **Aligned** Tensors

  - Observations/controls:  ℝ×(𝓞₁⊕𝓞₂⊕…⊕𝓞ₙ) ⊕ ℝ×(𝓒₁⊕𝓒₂⊕…⊕𝓒ₙ) ≃ ℝ×(𝓞⊕𝓒)
  - Video: Image + Audio + Text: (ℝ×𝐈) ⊕ (ℝ×𝐀) ⊕ (ℝ×𝐓) ≃ ℝ×(𝐈⊕𝐀⊕𝐓)
  - Generally cannot be represented by a single tensor since 𝐈≃ℝ⊗ℝ, but 𝐀≃ℝ
  - Can be represented as an xarray.Dataset (except for duplicate indices.)

- TimeSeries -- Static Data.

  - Example: MetaData attached to a TimeSeries.
  - Possibly Multiple Tensors that represent static data
  - This is guaranteed to not change with time and hence different from an observable
    that remains constant but could *potentially* change its value.

- Collections of TimeSeries with same modality

  - Example: repetition of same experiment.
  - batching can be done naturally ((concat) / list / padded / packed)

- Collections of TimeSeries with different modality

  - Example: collection of results from different experiments
  - Hard MetaLearning problem
  - Batching: only (list).

Consequences for representing TimeSeries Data
---------------------------------------------

- Collect all aligned TS data in a `tuple[tensor]`.

  - another advantage of this is that one could potentially split a sparse multivariate series into
    multiple univariate series which can be stored in a dense manner.

- Collect all static data in a `tuple[tensor]`.
- Collect all collections in a dictionary.
- Q: Where should single tensor meta-data be stored? in the tensor itself?

  - For example: Value Range for columns, measurement uncertainty per column, etc.
  - → use DataFrame.attr / DataArray.attr

Consequences for loading TimeSeries data
----------------------------------------

- When sampling a TimeSlice, we should return a tuple of tensors from all Time-Aligned tensors
  in the dataset, corresponding to the same TimeSlice.
- Time-Independent Tensors should not be sampled / returned.
- Q: Should the DataLoader be responsible for splitting interval data into observation/prediction?

  - Split obligations into multiple parts:
  - Sampler: responsible for sampling a single example
  - Collator: responsible for collating multiple samples into a batch
  - Encoder: responsible for transforming data from numpy/pandas/xarray into FloatTensor for NN
  - Splitter: responsible for splitting the data into observable/target.

- Q: How to best distinguish between static tensor and TimeTensors?

  - Either: two separate tuples
  - Or: select by tensor attributes.

    - Time tensors should be indexed by a "time" axis.
    - Or define a special type of tensor "TimeTensor", that must have a TimeLike index.


Proposal
--------

We should consider the following data-structures

- TimeSeries-Tensor: a Tensor whose main index is TimeLike
- TimeSeries-Dataset: a Tuple of TimeSeriesTensors together with a tuple of regular tensors.
- MetaDataSet (equimodal/non-equimodal): a Collection of Dataset objects with the same/different modality over some index

Example
-------

We want to show a complete example for a case where we both need a `tuple[TimeTensor]` and a `tuple[Tensor]`
to represent the dataset. Consider a Video then we have

TimeTensors:

- picture signal: time×width×height
- audio signal: time×frequency
- subtitles: time×variable_length

Static Tensors:

- preview image: width×height
- genre: class
- language: class
- runtime: timedelta
- year: datetime
- title: string

Seeing the variety of data possible, we see that likely things cannot be encoded in a single string.


Mental Model h5 files
---------------------

All datasets are stored in `Hierarchical Data Format <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_, based on

This format knows two things:

- Datasets, which are multidimensional arrays of a homogeneous type
- Groups, which are container structures which can hold datasets and other groups

For example, pandas will store a single DataFrame with different data type columns as
multiple Series objects of homogeneous data-type, collected in a group.


Supported DataTypes:

For Datasets: Only nullable types.

Dtypes:

- `pandas.BooleanDtype`
- `pandas.CategoricalDtype`
- `pandas.DatetimeTZDtype`
- `pandas.Float32Dtype`
- `pandas.Float64Dtype`
- `pandas.Int16Dtype`
- `pandas.Int32Dtype`
- `pandas.Int64Dtype`
- `pandas.Int8Dtype`
- `pandas.IntervalDtype`
- `pandas.PeriodDtype`
- `pandas.SparseDtype`
- `pandas.StringDtype`
- `pandas.UInt16Dtype`
- `pandas.UInt32Dtype`
- `pandas.UInt64Dtype`
- `pandas.UInt8Dtype`

Index Types:

- `pandas.CategoricalIndex`
- `pandas.DatetimeIndex`
- `pandas.Float64Index`
- `pandas.Index`
- `pandas.IndexSlice`
- `pandas.Int64Index`
- `pandas.IntervalIndex`
- `pandas.MultiIndex`
- `pandas.PeriodIndex`
- `pandas.RangeIndex`
- `pandas.TimedeltaIndex`
- `pandas.UInt64Index`

Arrays:

- `pandas.ArrowStringArray`
- `pandas.BooleanArray`
- `pandas.Categorical`
- `pandas.DatetimeArray`
- `pandas.FloatingArray`
- `pandas.IntegerArray`
- `pandas.IntervalArray`
- `pandas.PandasArray`
- `pandas.PeriodArray`
- `pandas.SparseArray`
- `pandas.StringArray`
- `pandas.TimedeltaArray`
