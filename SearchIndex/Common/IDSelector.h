#include <cstdlib>

namespace Search
{

/**
 * @brief Abstract class for selecting IDs.
 *
 * This is used for filter search and lightweight deletes.
 */
class AbstractIDSelector
{
public:
    using idx_t = int64_t;

    virtual bool is_member(idx_t id) const = 0;

    virtual ~AbstractIDSelector() = default;
};

}
