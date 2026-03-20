/*
 * Copyright © 2024 ORIGINHUB SINGAPORE PTE. LTD.
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

#include <SearchIndex/VectorSearch.h>
#include "SearchIndexCommon.h"

namespace Search
{

namespace ParameterSets
{
    // Float vector metric types
    inline const ParameterSpec FLOAT_METRIC_TYPE = ParameterSpec::makeString({"L2", "Cosine", "IP"}, false);

    // Binary vector metric types
    inline const ParameterSpec BINARY_METRIC_TYPE = ParameterSpec::makeString({"Hamming", "Jaccard"}, false);

    // IVF base parameters
    inline const ParameterSpec NCENTROIDS = ParameterSpec::makeInt(1, 1048576);
    inline const ParameterSpec NPROBE = ParameterSpec::makeInt(1, 1048576);

    // HNSW base parameters
    inline const ParameterSpec M = ParameterSpec::makeInt(8, 128);
    inline const ParameterSpec EF_C = ParameterSpec::makeInt(16, 1024);
    inline const ParameterSpec EF_S = ParameterSpec::makeInt(16, 1024);

    // Alpha parameter for search
    inline const ParameterSpec ALPHA = ParameterSpec::makeFloat(1.0, 4.0);

    // M parameter for IVF
    inline const ParameterSpec IVF_M = ParameterSpec::makeInt(0, 2147483647);

    // PQ bit size
    inline const ParameterSpec PQ_BIT_SIZE = ParameterSpec::makeInt(2, 12);

    // SQ bit size (string type with specific candidates)
    inline const ParameterSpec SQ_BIT_SIZE = ParameterSpec::makeString(
        {"4bit", "6bit", "8bit", "8bit_uniform", "8bit_direct", "4bit_uniform", "QT_fp16"}, true);
}

/// MyScale Valid Index Parameters MAP
inline const ValidIndexParametersMap MYSCALE_VALID_INDEX_PARAMETERS_MAP = {

    // SCANN index
    {"SCANN",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE}},
      // Search parameters
      {{"alpha", ParameterSets::ALPHA}}}},

    // FLAT index
    {"FLAT",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE}},
      // Search parameters
      {}}},

    // IVFFLAT index
    {"IVFFLAT",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE},
       {"ncentroids", ParameterSets::NCENTROIDS}},
      // Search parameters
      {{"nprobe", ParameterSets::NPROBE}}}},

    // IVFPQ index
    {"IVFPQ",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE},
       {"ncentroids", ParameterSets::NCENTROIDS},
       {"M", ParameterSets::IVF_M},
       {"bit_size", ParameterSets::PQ_BIT_SIZE}},
      // Search parameters
      {{"nprobe", ParameterSets::NPROBE}}}},

    // IVFFASTPQ index
    {"IVFFASTPQ",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE},
       {"ncentroids", ParameterSets::NCENTROIDS},
       {"M", ParameterSets::IVF_M},
       {"bit_size", ParameterSets::PQ_BIT_SIZE}},
      // Search parameters
      {{"nprobe", ParameterSets::NPROBE}}}},

    // IVFSQ index
    {"IVFSQ",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE},
       {"ncentroids", ParameterSets::NCENTROIDS},
       {"bit_size", ParameterSets::SQ_BIT_SIZE}},
      // Search parameters
      {{"nprobe", ParameterSets::NPROBE}}}},

    // HNSWFLAT index
    {"HNSWFLAT",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE},
       {"m", ParameterSets::M},
       {"ef_c", ParameterSets::EF_C}},
      // Search parameters
      {{"ef_s", ParameterSets::EF_S}}}},

    // HNSWFASTFLAT index
    {"HNSWFASTFLAT",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE},
       {"m", ParameterSets::M},
       {"ef_c", ParameterSets::EF_C}},
      // Search parameters
      {{"ef_s", ParameterSets::EF_S}}}},

    // HNSWSQ index
    {"HNSWSQ",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE},
       {"m", ParameterSets::M},
       {"ef_c", ParameterSets::EF_C},
       {"bit_size", ParameterSets::SQ_BIT_SIZE}},
      // Search parameters
      {{"ef_s", ParameterSets::EF_S}}}},

    // HNSWFASTSQ index
    {"HNSWFASTSQ",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE},
       {"m", ParameterSets::M},
       {"ef_c", ParameterSets::EF_C},
       {"bit_size", ParameterSets::SQ_BIT_SIZE}},
      // Search parameters
      {{"ef_s", ParameterSets::EF_S}}}},

    // HNSWPQ index
    {"HNSWPQ",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE},
       {"m", ParameterSets::M},
       {"ef_c", ParameterSets::EF_C},
       {"bit_size", ParameterSets::PQ_BIT_SIZE}},
      // Search parameters
      {{"ef_s", ParameterSets::EF_S}}}},

    // HNSWFASTPQ index
    {"HNSWFASTPQ",
     {// Build parameters
      {{"metric_type", ParameterSets::FLOAT_METRIC_TYPE},
       {"m", ParameterSets::M},
       {"ef_c", ParameterSets::EF_C},
       {"bit_size", ParameterSets::PQ_BIT_SIZE}},
      // Search parameters
      {{"ef_s", ParameterSets::EF_S}}}},

    // Binary vector indices
    {"BINARYFLAT",
     {// Build parameters
      {{"metric_type", ParameterSets::BINARY_METRIC_TYPE}},
      // Search parameters
      {}}},

    {"BINARYIVF",
     {// Build parameters
      {{"metric_type", ParameterSets::BINARY_METRIC_TYPE},
       {"ncentroids", ParameterSets::NCENTROIDS},
       {"M", ParameterSets::IVF_M}},
      // Search parameters
      {{"nprobe", ParameterSets::NPROBE}}}},

    {"BINARYHNSW",
     {// Build parameters
      {{"metric_type", ParameterSets::BINARY_METRIC_TYPE},
       {"m", ParameterSets::M},
       {"ef_c", ParameterSets::EF_C}},
      // Search parameters
      {{"ef_s", ParameterSets::EF_S}}}},
};

std::string getDefaultIndexType(const DataType &search_type)
{
    switch (search_type)
    {
        case DataType::FloatVector:
        {
            return "SCANN";
        }
        case DataType::BinaryVector:
        {
            return "BinaryIVF";
        }
        default:
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unsupported vector search type: %s", enumToString(search_type).c_str());
    }
}

/// The HNSW algorithm uses the corresponding HNSWFAST series algorithms by default
IndexType getVectorIndexType(std::string type, const DataType &search_type)
{
    std::transform(type.begin(), type.end(), type.begin(), [](unsigned char c) { return std::toupper(c); });

    switch (search_type)
    {
        case DataType::FloatVector:
        {
            if (type == "IVFFLAT")
                return IndexType::IVFFLAT;
            else if (type == "IVFPQ")
                return IndexType::IVFPQ;
            else if (type == "IVFSQ")
                return IndexType::IVFSQ;
            else if (type == "FLAT")
                return IndexType::FLAT;
            else if (type == "HNSWFLAT" || type == "HNSWFASTFLAT")
                return IndexType::HNSWfastFLAT;
            else if (type == "HNSWPQ" || type == "HNSWFASTPQ")
                return IndexType::HNSWPQ;
            else if (type == "HNSWSQ" || type == "HNSWFASTSQ")
                return IndexType::HNSWfastSQ;
            else if (type == "SCANN")
                return IndexType::SCANN;
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unknown index type for Float32 Vector: %s", type.c_str());
        }
        case DataType::BinaryVector:
        {
            if (type == "BINARYIVF")
                return IndexType::BinaryIVF;
            else if (type == "BINARYHNSW")
                return IndexType::BinaryHNSW;
            else if (type == "BINARYFLAT")
                return IndexType::BinaryFLAT;
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unknown index type for Binary Vector: %s", type.c_str());
        }
        default:
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unsupported vector search type: %s", enumToString(search_type).c_str());
    }
}

Metric getMetricType(std::string metric, const DataType &search_type)
{
    std::transform(metric.begin(), metric.end(), metric.begin(), [](unsigned char c) { return std::toupper(c); });

    switch (search_type)
    {
        case DataType::FloatVector:
            if (metric == "L2")
                return Metric::L2;
            else if (metric == "IP")
                return Metric::IP;
            else if (metric == "COSINE")
                return Metric::Cosine;
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unknown metirc type for Float32 Vector: %s", metric.c_str());
        case DataType::BinaryVector:
            if (metric == "HAMMING")
                return Metric::Hamming;
            else if (metric == "JACCARD")
                return Metric::Jaccard;
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unknown metirc type for Binary Vector: %s", metric.c_str());
        default:
            SI_THROW_FMT(ErrorCode::BAD_ARGUMENTS, "Unsupported vector search type: %s", enumToString(search_type).c_str());
    }
}

}
