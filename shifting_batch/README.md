# Shifting Batch mechanism

Problem: we cannot store all the batches. We need to generate them, consume (training) and delete them.

Solution: have a buffer of batches that are generated dynamically, just before training.
At each time, we have the current batches (used in training) and the next batches (being generated) so that we can minimize waiting time.


| stage          | time 0 | time 1 | time 2  | time 3  |
|----------------|--------|--------|---------|---------|
| create_batch   | [0:k]  | [k:2k] | [2k:3k] |         |
| train + delete |        | [0:k]  | [k:2k]  | [2k:3k] |
