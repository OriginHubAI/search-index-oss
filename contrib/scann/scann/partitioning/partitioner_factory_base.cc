// Copyright 2022 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scann/partitioning/partitioner_factory_base.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/random/random.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "scann/distance_measures/distance_measure_factory.h"
#include "scann/proto/distance_measure.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace research_scann {

namespace {

float ComputeSamplingFraction(const PartitioningConfig& config,
                              const Dataset* dataset) {
  return (config.has_expected_sample_size())
             ? std::min(1.0,
                        static_cast<double>(config.expected_sample_size()) /
                            dataset->size())
             : config.partitioning_sampling_fraction();
}

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactoryNoProjection(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<ThreadPool> pool, SampleDatasetCreator sample_dataset_creator) {
  const TypedDataset<T>* sampled;
  unique_ptr<TypedDataset<T>> sampled_mutable;
  Deleter sample_data_deleter = nullptr;
  Search::OnExit on_exit([&]() {
    // release sample data resource on exit
    sampled_mutable.reset();
    if (sample_data_deleter) sample_data_deleter();
  });

  const float sampling_fraction = ComputeSamplingFraction(config, dataset);
  // always put sampled dataset in memory
  if (sampling_fraction < 1.0 || dataset->needDataPrefetching()) {
    sampled_mutable.reset(
        (dataset->IsSparse())
            ? absl::implicit_cast<TypedDataset<T>*>(new SparseDataset<T>)
            : absl::implicit_cast<TypedDataset<T>*>(new DenseDataset<T>));
    MTRandom rng(kDeterministicSeed + 1);
    vector<DatapointIndex> sample;
    for (DatapointIndex i = 0; i < dataset->size(); ++i) {
      if (absl::Uniform<float>(rng, 0, 1) < sampling_fraction) {
        sample.push_back(i);
      }
    }

    bool sample_dataset_created = false;
    if constexpr (std::is_same_v<T, float>) {
      if (!dataset->IsSparse() && sample_dataset_creator) {
        // allocate dataset on the fly
        auto ptr = sample_dataset_creator(sample.size(), dataset->dimensionality(), sample_data_deleter);
        if (ptr) {
          // use the new dataset if it's successfully created
          sampled_mutable.reset(static_cast<TypedDataset<float>*>(ptr.release()));
          sample_dataset_created = true;
        }
      }
    }
    if (!sample_dataset_created) {
      // reserve in-memory dataset
      sampled_mutable->Reserve(sample.size());
    }
    SCANN_RETURN_IF_ERROR(
        sampled_mutable->NormalizeByTag(dataset->normalization()));
    sampled = sampled_mutable.get();

    size_t sample_idx = 0;
    size_t batch_size = dataset->needDataPrefetching() ?
            dataset->prefetchSizeLimit() : sample.size();
    for (size_t st=0; st<sample.size(); st+=batch_size) {
      size_t len = std::min(batch_size, sample.size()-st);
      Search::PrefetchInfo* prefetch_info{nullptr};
      if (dataset->needDataPrefetching()) {
          std::vector<int64_t> prefetch_list;
          for (auto i : absl::MakeSpan(sample).subspan(st, len))
              prefetch_list.push_back(i);
          prefetch_info = dataset->prefetchData(std::move(prefetch_list));
          dataset->setThreadPrefetchInfo(prefetch_info);
      }
      for (DatapointIndex i: absl::MakeSpan(sample).subspan(st, len)) {
        if (sample_dataset_created) {
          auto p = dataset->at(i);
          const T* new_values = sampled_mutable->at(sample_idx++).values();
          std::copy(p.values(), p.values() + p.dimensionality(), (T*) new_values);
        }
        else sampled_mutable->AppendOrDie(dataset->at(i), "");
      }
      dataset->releasePrefetch(prefetch_info);
      dataset->setThreadPrefetchInfo(nullptr);
    }
  } else {
    sampled = dataset;
  }
  LOG(INFO) << "Size of sampled dataset for training partition: "
            << sampled->size();

  return PartitionerFactoryPreSampledAndProjected(sampled, config, pool);
}

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactoryWithProjection(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<ThreadPool> pool, SampleDatasetCreator sample_dataset_creator) {
  const TypedDataset<float>* sampled;
  unique_ptr<TypedDataset<float>> sampled_mutable;
  MTRandom rng(kDeterministicSeed + 1);
  vector<DatapointIndex> sample;
  const float sampling_fraction = ComputeSamplingFraction(config, dataset);
  for (DatapointIndex i = 0; i < dataset->size(); ++i) {
    if (absl::Uniform<float>(rng, 0, 1) < sampling_fraction) {
      sample.push_back(i);
    }
  }

  auto append_to_sampled = [&](const DatapointPtr<float>& dptr) -> Status {
    if (ABSL_PREDICT_FALSE(!sampled_mutable)) {
      if (dptr.IsSparse()) {
        sampled_mutable = make_unique<SparseDataset<float>>();
      } else {
        sampled_mutable = make_unique<DenseDataset<float>>();
      }
      sampled_mutable->Reserve(sample.size());
      SCANN_RETURN_IF_ERROR(
          sampled_mutable->NormalizeByTag(dataset->normalization()));
      sampled = sampled_mutable.get();
    }
    return sampled_mutable->Append(dptr, "");
  };
  TF_ASSIGN_OR_RETURN(unique_ptr<Projection<T>> projection,
                      ProjectionFactory(config.projection(), dataset));
  Datapoint<float> projected;
  for (DatapointIndex i : sample) {
    SCANN_RETURN_IF_ERROR(projection->ProjectInput(dataset->at(i), &projected));
    SCANN_RETURN_IF_ERROR(append_to_sampled(projected.ToPtr()));
  }
  LOG(INFO) << "Size of sampled dataset for training partition: "
            << sampled->size();
  TF_ASSIGN_OR_RETURN(
      auto raw_partitioner,
      PartitionerFactoryPreSampledAndProjected(sampled, config, pool));
  return MakeProjectingDecorator<T, float>(std::move(projection),
                                           std::move(raw_partitioner));
}
}  // namespace

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactory(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<ThreadPool> pool, SampleDatasetCreator sample_dataset_creator) {
  auto fp = (config.has_projection()) ? (&PartitionerFactoryWithProjection<T>)
                                      : (&PartitionerFactoryNoProjection<T>);
  return (*fp)(dataset, config, pool, sample_dataset_creator);
}

template <typename T>
StatusOr<unique_ptr<Partitioner<T>>> PartitionerFactoryPreSampledAndProjected(
    const TypedDataset<T>* dataset, const PartitioningConfig& config,
    shared_ptr<ThreadPool> training_parallelization_pool) {
  if (config.tree_type() == PartitioningConfig::KMEANS_TREE) {
    return KMeansTreePartitionerFactoryPreSampledAndProjected(
        dataset, config, training_parallelization_pool);
  } else {
    return InvalidArgumentError("Invalid partitioner type.");
  }
}

SCANN_INSTANTIATE_PARTITIONER_FACTORY(, int8_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, uint8_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, int16_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, int32_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, uint32_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, int64_t);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, float);
SCANN_INSTANTIATE_PARTITIONER_FACTORY(, double);

}  // namespace research_scann
