/*
 * Copyright © 2024 MOQI SINGAPORE PTE. LTD.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 3.0 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston MA 02110-1301, USA.
 */

#pragma once
#include <bit>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include "Utils.h"

namespace Search
{

class DenseBitmap;
using DenseBitmapPtr = std::shared_ptr<DenseBitmap>;

/**
 * @brief A dense bitmap for lightweight delete and filter search.
*/
class DenseBitmap
{
public:
    /// @brief Align the byte size of DenseBitmap to be multiple of 8.
    static const int BYTE_ALIGNMENT;

    DenseBitmap() = default;

    /// @brief Construct a bitmap with given size and value.
    explicit DenseBitmap(size_t size_, bool value = false, bool fill_data = true) : size(size_)
    {
        // bitmap = new uint8_t[byte_size()];
        bitmap = static_cast<uint8_t *>(
            std::aligned_alloc(BYTE_ALIGNMENT, byte_size()));
        if (!bitmap)
            throw std::bad_alloc();

        if (fill_data)
            memset(bitmap, value ? 255 : 0, byte_size());
    }

    DenseBitmap(const DenseBitmap & other)
    {
        size = other.size;
        bitmap = static_cast<uint8_t *>(
            std::aligned_alloc(BYTE_ALIGNMENT, byte_size()));
        if (!bitmap)
            throw std::bad_alloc();

        memcpy(bitmap, other.bitmap, byte_size());
    }

    /// @brief Return number elements in the bitmap
    size_t get_size() const { return size; }
    /**
     * @brief Return the byte size of the bitmap.
     * @note The byte size is aligned to be multiple of BYTE_ALIGNMENT.
     */
    inline size_t byte_size() const
    {
        return DIV_ROUND_UP(size, 8 * BYTE_ALIGNMENT) * BYTE_ALIGNMENT;
    }

    /// @brief Return the raw bitmap data.
    uint8_t * get_bitmap() { return bitmap; }

    /// @brief Check whether `id` is in the bitmap.
    /// @note This function will throw if `id` is out of range.
    inline bool is_member(size_t id) const
    {
        SI_THROW_IF_NOT(id < size, ErrorCode::LOGICAL_ERROR);
        return unsafe_test(id);
    }

    /// @brief Check whether `id` is in the bitmap without boundary check.
    inline bool unsafe_test(size_t id) const
    {
        return (bitmap[id >> 3] & (0x1 << (id & 0x7)));
    }

    /**
     * @brief Approximately count the number of bits set to 1 around `id`.
     * 
     * This function is used to estimate the cardinality of the bitmap for 
     * filter search parameter auto-tuning.
     */
    inline std::pair<size_t, size_t> unsafe_approx_count_member(size_t id) const
    {
        SI_THROW_IF_NOT_FMT(
            id < get_size(),
            ErrorCode::LOGICAL_ERROR,
            "id must be smaller than bitmap size: %lu vs. %lu",
            id,
            get_size());
        /// Check 8 bytes at a time
        size_t idx = (id >> 3) & ~7ULL;
        SI_THROW_IF_NOT_FMT(
            idx + 7 < byte_size(),
            ErrorCode::LOGICAL_ERROR,
            "idx+7 exceededs byte_size: %lu vs. %lu",
            idx + 7,
            byte_size());
        size_t count
            = std::popcount(*reinterpret_cast<uint64_t *>(&bitmap[idx]));
        return std::make_pair(64, count);
    }

    /// @brief set bit value at `id` to 1 in the bitmap
    /// @note this function will throw if `id` is out of range
    inline void set(size_t id)
    {
        SI_THROW_IF_NOT(id < size, ErrorCode::LOGICAL_ERROR);
        size_t byte_index = id >> 3;
        uint8_t bit_mask = 0x1 << (id & 0x7);

        /// Use std::atomic_fetch_or_explicit to perform an atomit OR operation
        /// Ensure thread-safety while maintaining good performance
        std::atomic_fetch_or_explicit(
            reinterpret_cast<std::atomic<uint8_t> *>(&bitmap[byte_index]),
            bit_mask,
            std::memory_order_relaxed);
    }

    /// @brief set bit value at `from begin_id to end_id` to 1 in the bitmap
    void batch_set(const size_t begin_id, const size_t end_id);

    /// @brief set bit value at `id` to 0 in the bitmap
    /// @note this function will throw if `id` is out of range
    inline void unset(size_t id)
    {
        SI_THROW_IF_NOT(id < size, ErrorCode::LOGICAL_ERROR);
        bitmap[id >> 3] &= ~(0x1 << (id & 0x7));
    }

    /// @brief Check whether all bits are set to 1 in the bitmap.
    bool all() const;
    /// @brief Check whether any bit is set to 1 in the bitmap.
    bool any() const;

    /// @brief Count the number of bits set to 1 in the bitmap.
    size_t count() const
    {
        size_t count = 0;
        const uint64_t * bitmap64 = reinterpret_cast<const uint64_t *>(bitmap);
        const size_t n_chunks = byte_size() / sizeof(uint64_t);

        for (size_t i = 0; i < n_chunks; ++i)
        {
            count += __builtin_popcountll(bitmap64[i]);
        }

        const uint8_t * remaining
            = reinterpret_cast<const uint8_t *>(bitmap64 + n_chunks);
        for (size_t i = 0; i < byte_size() % sizeof(uint64_t); ++i)
        {
            count += __builtin_popcount(remaining[i]);
        }

        return count;
    }

    /// @brief Free the memory allocated by the bitmap.
    ~DenseBitmap()
    {
        // delete null pointer has no effect
        free(bitmap);
    }

    /// @brief Return the raw bitmap data.
    uint8_t * data() { return bitmap; }

    /// @brief Return indices of all the bits set to 1 as a vector.
    std::vector<size_t> to_vector(const bool need_reserve = true) const
    {
        std::vector<size_t> result;
        if (need_reserve)
            result.reserve(count());

        const uint64_t * bitmap64 = reinterpret_cast<const uint64_t *>(bitmap);
        const size_t n_chunks = size / 64;

        for (size_t chunk = 0; chunk < n_chunks; ++chunk)
        {
            uint64_t word = bitmap64[chunk];
            const size_t base = chunk * 64;

            while (word)
            {
                unsigned int pos = __builtin_ctzll(word);
                result.push_back(base + pos);
                word &= (word - 1);
            }
        }

        const size_t remaining_start = n_chunks * 64;
        for (size_t i = remaining_start; i < size; ++i)
        {
            if (unsafe_test(i))
            {
                result.push_back(i);
            }
        }

        return result;
    }

    /// @brief Intersect elements of two dense bitmaps.
    static DenseBitmapPtr
    intersectDenseBitmaps(DenseBitmapPtr left, DenseBitmapPtr right);

private:
    /// @brief The raw bitmap data.
    uint8_t * bitmap;

    /// @brief Number of elements in the bitmap.
    size_t size;
};
}
