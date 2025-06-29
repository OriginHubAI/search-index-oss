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

#include "DenseBitmap.h"
#include <simde/x86/sse2.h>

namespace Search
{
const int DenseBitmap::BYTE_ALIGNMENT = sizeof(simde__m128i);

bool DenseBitmap::all() const
{
    // Use SIMDE for cross-platform SIMD support
    const size_t simde_chunks = byte_size() / sizeof(simde__m128i);
    const simde__m128i* bitmap_simde = reinterpret_cast<const simde__m128i*>(bitmap);
    const simde__m128i all_ones = simde_mm_set1_epi8(-1);

    // Process 16-byte chunks using SIMDE
    for (size_t i = 0; i < simde_chunks; ++i)
    {
        simde__m128i current = simde_mm_loadu_si128(&bitmap_simde[i]);  // Using loadu for potentially unaligned data
        simde__m128i cmp = simde_mm_cmpeq_epi8(current, all_ones);
        if (simde_mm_movemask_epi8(cmp) != 0xFFFF)
        {
            return false;
        }
    }

    const size_t processed_bytes = simde_chunks * sizeof(simde__m128i);
    const uint8_t* remaining = bitmap + processed_bytes;
    for (size_t i = 0; i < byte_size() - processed_bytes; ++i)
    {
        if (remaining[i] != 0xFF)
        {
            return false;
        }
    }
    return true;
}

bool DenseBitmap::any() const
{
    // Use SIMDE for cross-platform SIMD support
    const size_t simde_chunks = byte_size() / sizeof(simde__m128i);
    const simde__m128i* bitmap_simde = reinterpret_cast<const simde__m128i*>(bitmap);
    const simde__m128i all_zeros = simde_mm_setzero_si128();

    // Process 16-byte chunks using SIMDE
    for (size_t i = 0; i < simde_chunks; ++i)
    {
        simde__m128i current = simde_mm_loadu_si128(&bitmap_simde[i]);  // Using loadu for potentially unaligned data
        simde__m128i cmp = simde_mm_cmpeq_epi8(current, all_zeros);
        if (simde_mm_movemask_epi8(cmp) != 0xFFFF)
        {
            return true;
        }
    }

    const size_t processed_bytes = simde_chunks * sizeof(simde__m128i);
    const uint8_t * remaining = bitmap + processed_bytes;
    for (size_t i = 0; i < byte_size() - processed_bytes; ++i)
    {
        if (remaining[i] != 0)
            return true;
    }
    return false;
}

DenseBitmapPtr
DenseBitmap::intersectDenseBitmaps(DenseBitmapPtr left, DenseBitmapPtr right)
{
    if (left == nullptr)
        return right;
    else if (right == nullptr)
        return left;
    SI_THROW_IF_NOT_FMT(
        left->get_size() == right->get_size(),
        ErrorCode::LOGICAL_ERROR,
        "left size %zu != right size %zu",
        left->get_size(),
        right->get_size());

    DenseBitmapPtr result
        = std::make_shared<DenseBitmap>(left->get_size(), false, false);

    // Use SIMD for cross-platform SIMD support
    const size_t simde_chunks = left->byte_size() / sizeof(simde__m128i);
    auto* result_sse = reinterpret_cast<simde__m128i*>(result->bitmap);
    const auto* left_sse = reinterpret_cast<const simde__m128i*>(left->bitmap);
    const auto* right_sse = reinterpret_cast<const simde__m128i*>(right->bitmap);

    for (size_t i = 0; i < simde_chunks; ++i)
    {
        result_sse[i] = simde_mm_and_si128(left_sse[i], right_sse[i]);
    }

    const size_t remaining_offset = simde_chunks * sizeof(simde__m128i);
    const size_t remaining_bytes = left->byte_size() - remaining_offset;
    auto* result_bytes = reinterpret_cast<uint8_t*>(result->bitmap) + remaining_offset;
    const auto* left_bytes = reinterpret_cast<const uint8_t*>(left->bitmap) + remaining_offset;
    const auto* right_bytes = reinterpret_cast<const uint8_t*>(right->bitmap) + remaining_offset;

    for (size_t i = 0; i < remaining_bytes; ++i)
    {
        result_bytes[i] = left_bytes[i] & right_bytes[i];
    }

    return result;
}

}
