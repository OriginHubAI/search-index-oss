#include <SearchIndex/VectorIndex.h>

namespace Search
{

template <typename IS, typename OS, IDSelector IDS, DataType dataType>
int VectorIndex<IS, OS, IDS, dataType>::computeFirstStageNumCandidates(
    IndexType index_type,
    bool disk_mode,
    int64_t num_data,
    int64_t data_dim,
    int32_t topK,
    Parameters params)
{
    SI_THROW_FMT(
        ErrorCode::BAD_ARGUMENTS,
        "Unsupported index type for computing num_candidates: %s",
        enumToString(index_type).c_str());
}

// instantiate the template class
template class VectorIndex<
    AbstractIStream,
    AbstractOStream,
    DenseBitmap,
    DataType::FloatVector>;

}
