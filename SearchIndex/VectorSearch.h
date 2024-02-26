#pragma once

#include <SearchIndex/Config.h>
#include <SearchIndex/LocalDiskFileStore.h>
#include <SearchIndex/VectorIndex.h>
#ifdef ENABLE_DISKANN
#    include <SearchIndex/DiskANNIndex.h>
#endif
#ifdef ENABLE_FAISS
#    include <SearchIndex/FaissIndex.h>
#endif
#ifdef ENABLE_SCANN
#    include <SearchIndex/MSTGIndex.h>
#    include <SearchIndex/MultiPartMSTGIndex.h>
#endif

namespace Search
{

// db depend on this variable
// BinaryIVF and BInaryHNSW are disabled, and only BinaryMSTG is enabled for binary indices.
inline static const std::vector<IndexType> BINARY_VECTOR_INDEX_TYPES = {
    IndexType::BinaryFLAT,
    // IndexType::BinaryIVF,
    // IndexType::BinaryHNSW,
    IndexType::BinaryMSTG,
};

inline static const std::vector<IndexType> BINARY_VECTOR_INDEX_TEST_TYPES = {
    IndexType::BinaryFLAT,
    IndexType::BinaryIVF,
    IndexType::BinaryHNSW,
};

// db depend on this variable
inline static const std::vector<IndexType> FLOAT_VECTOR_INDEX_TEST_TYPES = {
#ifdef ENABLE_FAISS
    IndexType::IVFFLAT,
    IndexType::IVFPQ,
    IndexType::IVFSQ,
    IndexType::FLAT,
    IndexType::HNSWfastFLAT,
    IndexType::HNSWfastPQ,
    IndexType::HNSWfastSQ,
    IndexType::HNSWFLAT,
    IndexType::HNSWPQ,
    IndexType::HNSWSQ,
#endif
#ifdef ENABLE_DISKANN
    IndexType::VAMANA,
    IndexType::DISKANN,
#endif
#ifdef ENABLE_SCANN
    IndexType::MSTG,
    IndexType::MultiPartMSTG
#endif
};

/**
 * \brief create VectorIndex index
 *
 * \param name name of the name of index
 * \param metric metric of index
 * \param data_dim data dim of index
 *
 * \return shared pointer VectorIndex instance
 */
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
std::shared_ptr<VectorIndex<IS, OS, IDS, dataType>> createVectorIndex(
    const std::string & name,
    IndexType index_type,
    Metric metric,
    size_t data_dim,
    size_t max_points,
    const Parameters & params,
    [[maybe_unused]] bool load_diskann_after_build = true,
    [[maybe_unused]] const std::string & file_store_prefix = "",
#ifdef ENABLE_SCANN
    [[maybe_unused]] std::shared_ptr<DiskIOManager> io_manager = nullptr,
#endif
    [[maybe_unused]] bool use_file_checksum = false,
    [[maybe_unused]] bool manage_cache_folder = false)
{
    switch (index_type)
    {
#ifdef ENABLE_FAISS
        case IndexType::FLAT:
        case IndexType::BinaryFLAT:
            return std::make_shared<FaissFlatIndex<IS, OS, IDS, dataType>>(
                name, index_type, metric, data_dim, max_points, params);
        case IndexType::IVFFLAT:
        case IndexType::IVFSQ:
        case IndexType::IVFPQ:
        case IndexType::BinaryIVF:
            return std::make_shared<FaissIVFIndex<IS, OS, IDS, dataType>>(
                name, index_type, metric, data_dim, max_points, params);
        case IndexType::HNSWfastFLAT:
        case IndexType::HNSWfastSQ:
        case IndexType::HNSWfastPQ:
        case IndexType::HNSWFLAT:
        case IndexType::HNSWSQ:
        case IndexType::HNSWPQ:
        case IndexType::BinaryHNSW:
            return std::make_shared<FaissHNSWIndex<IS, OS, IDS, dataType>>(
                name, index_type, metric, data_dim, max_points, params);
#endif
#ifdef ENABLE_DISKANN
        case IndexType::VAMANA:
            return std::make_shared<DiskANNMemoryIndex<IS, OS, IDS, dataType>>(
                name, metric, data_dim, max_points, params);
        case IndexType::DISKANN: {
            std::shared_ptr<FileStore<IS, OS>> file_store
                = std::make_shared<LocalDiskFileStore<IS, OS>>(
                    file_store_prefix, use_file_checksum);
            return std::make_shared<DiskANNFlashIndex<IS, OS, IDS, dataType>>(
                name,
                metric,
                data_dim,
                max_points,
                file_store,
                load_diskann_after_build,
                params);
        }
#endif
#ifdef ENABLE_SCANN
        case IndexType::MSTG: {
            bool disk_mode = params.getParam<int>(
                "disk_mode",
                MSTGIndex<IS, OS, IDS, dataType>::DEFAULT_DISK_MODE);
            // NOTE: we always use checksum in MSTG
            auto file_store = disk_mode > 0
                ? std::make_shared<LocalDiskFileStore<IS, OS>>(
                    file_store_prefix, use_file_checksum, manage_cache_folder)
                : nullptr;

            if constexpr (dataType == DataType::FloatVector)
            {
                return std::make_shared<MSTGIndex<IS, OS, IDS, dataType>>(
                    name,
                    metric,
                    data_dim,
                    max_points,
                    file_store,
                    io_manager,
                    params);
            }
        }
        case IndexType::BinaryMSTG: {
            if constexpr (dataType == DataType::BinaryVector)
            {
                return std::make_shared<MSTGBinaryIndex<IS, OS, IDS>>(
                    name,
                    IndexType::BinaryIVF,
                    metric,
                    data_dim,
                    max_points,
                    params);
            }
        }
        case IndexType::MultiPartMSTG: {
            bool disk_mode = params.getParam<int>(
                "disk_mode",
                MSTGIndex<IS, OS, IDS, dataType>::DEFAULT_DISK_MODE);
            // NOTE: we always use checksum in MSTG
            auto file_store = disk_mode > 0
                ? std::make_shared<LocalDiskFileStore<IS, OS>>(
                    file_store_prefix, use_file_checksum, manage_cache_folder)
                : nullptr;

            if constexpr (dataType == DataType::FloatVector)
            {
                return std::make_shared<
                    MultiPartMSTGIndex<IS, OS, IDS, dataType>>(
                    name,
                    metric,
                    data_dim,
                    max_points,
                    file_store,
                    io_manager,
                    params);
            }
        }
#endif
        default:
            SI_THROW_MSG(
                ErrorCode::UNSUPPORTED_PARAMETER,
                "Unsupported IndexType: " + enumToString(index_type));
    }
}

/// Implement adaptive vector indexing algorithm with few lines of code
template <typename IS, typename OS, IDSelector IDS, DataType dataType>
std::shared_ptr<VectorIndex<IS, OS, IDS, dataType>>
createFlatAdaptiveVectorIndex(
    const std::string & name,
    IndexType index_type,
    size_t flat_cutoff,
    Metric metric,
    size_t data_dim,
    size_t max_points,
    const Parameters & params)
{
#ifdef ENABLE_FAISS
    if (max_points <= flat_cutoff)
    {
        auto flat_index
            = std::make_shared<FaissFlatIndex<IS, OS, IDS, dataType>>(
                name + "_flat", metric, data_dim, max_points, params);
        // clear the search parameters for FlatIndex
        auto adapter
            = [](Parameters & search_params) { search_params.clear(); };
        flat_index->setSearchParamsAdapter(adapter);
    }
#endif
    return createVectorIndex<IS, OS, IDS, dataType>(
        name, index_type, metric, data_dim, max_points, params);
}

}
