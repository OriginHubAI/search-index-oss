# MyScale Search-Index

The MyScale Search-Index library encapsulates vector indexing algorithms used in MyScale and provides a unified interface for operations such as index building, loading, serialization, and vector search. While primarily intended for use as a module within MyScale, it also offers methods for standalone building and running unit tests. The library supports various vector index algorithms, including Flat, IVF, HNSW (with optimized HNSWfast), and ScaNN (with automatic parameter tuning).

## Building from source code

### Install dependencies

Similar to MyScale, Search-Index requires `clang-15` with `c++20` support to build from source:

```bash
sudo apt install clang-15 libc++abi-15-dev libc++-15-dev -y
sudo apt install libboost-all-dev libmkl-dev -y
```

### Build the project

After installing the dependencies, use `cmake` to build the project. The static library and unit test programs will be generated under the `build/` folder.

```bash
mkdir build && cd build
CC=clang-15 CXX=clang++-15 cmake .. && make -j
```

### Run Unit Tests

Execute the following commands under the `build` folder. The `run_tests.sh` script contains commands for testing vector indexes under various configurations:

```bash
cd build
bash ../scripts/run_tests.sh
```

## Credits

This project utilizes the following open-source vector search libraries:

- [Faiss](https://github.com/facebookresearch/faiss) - A library for efficient similarity search and clustering of dense vectors, by Meta's Fundamental AI Research.
- [hnswlib](https://github.com/nmslib/hnswlib) - Header-only C++/python library for fast approximate nearest neighbors.
- [ScaNN](https://github.com/google-research/google-research/tree/master/scann) - Scalable Nearest Neighbors library by Google Research.
