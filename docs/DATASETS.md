# Dataset Policy

This repository does not redistribute raw CAN-v1.5, ROAD, or MIRGU data.

Automotive CAN datasets often include separate citation, access, and redistribution conditions. 

## CAN-v1.5 / can-train-and-test

Use the official project/source provided by the dataset authors. Cite the dataset paper or README as requested by the authors.

Local expected layout:

```text
data/can-v1.5/
```

## ROAD

ROAD is the Real ORNL Automotive Dynamometer CAN Intrusion Dataset. The local README requests citation of:

```bibtex
@article{verma2020road,
  title={ROAD: the real ORNL automotive dynamometer controller area network intrusion detection dataset (with a comprehensive CAN IDS dataset survey & guide)},
  author={Verma, Miki E and Iannacone, Michael D and Bridges, Robert A and Hollifield, Samuel C and Kay, Bill and Combs, Frank L},
  journal={arXiv preprint arXiv:2012.14600},
  year={2020}
}
```

Local expected layout:

```text
data/road/
```

## MIRGU

Use the official VehicleSec/MIRGU release source and follow the authors' citation and redistribution terms.

Local expected layout:

```text
data/mirgu/
```

## Public Repo Rule

Do not commit:

- raw dataset files
- full preprocessed caches
- labels copied from restricted metadata
- checkpoints trained on datasets whose terms do not permit redistribution
- local absolute paths

