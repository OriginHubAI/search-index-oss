#pragma once
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <span>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <SearchIndex/Common/DenseDataLayer.h>
#include <SearchIndex/Common/IndexSourceData.h>
#include <SearchIndex/Common/Utils.h>
#include <faiss/utils/distances.h>

namespace Search
{

/// @brief Dense data layer backed by memory.
template <typename T>
class DenseMemoryDataLayer : public DenseDataLayer<T>
{
public:
    using DataChunk = DataSet<T>;
    using IStreamPtr = typename DenseDataLayer<T>::IStreamPtr;
    using OStreamPtr = typename DenseDataLayer<T>::OStreamPtr;
    bool use_lazy_decode = false;
    MemoryManager * mem_manager;

    DenseMemoryDataLayer(
        size_t max_data_,
        size_t data_dim_,
        bool use_fp16_storage_ = false,
        bool init_data = false) :
        DenseDataLayer<T>(max_data_, data_dim_, use_fp16_storage_)
    {
        SI_LOG_INFO(
            "Creating DenseMemoryDataLayer, use_fp16_storage={} data_dim={} "
            "use_lazy_decode={}",
            use_fp16_storage_,
            data_dim_,
            use_lazy_decode);

        if (use_fp16_storage_)
        {
            mem_manager = getMemoryManager();
            checkAvailableMemory(
                this->max_data * this->data_dim * sizeof(uint16_t) / sizeof(T));
        }
        else
        {
            mem_manager = nullptr;
            checkAvailableMemory(this->max_data * this->data_dim * sizeof(T));
        }
        if (init_data)
        {
            SI_THROW_IF_NOT_MSG(
                !use_fp16_storage_,
                ErrorCode::LOGICAL_ERROR,
                "init_data not support for DenseMemoryDataLayer with "
                "use_fp16_storage");
            data.resize(this->max_data * this->data_dim);
            // the data has already been initialized
            this->data_num = max_data_;
        }
    }

    virtual void remove() override
    {
        data.clear();
        this->max_data = 0;
        this->data_num = 0;
    }

    const T * getDataPtrImpl(idx_t idx) const
    {
        if (this->use_fp16_storage)
        {
            auto p_info = this->thread_prefetch_info;
            auto * block
                = reinterpret_cast<MemFp32Block *>(p_info->read_buffer);
            auto * ret_ptr = block->getDataPtr(idx);
            if (use_lazy_decode)
            {
                auto * encoded_ptr
                    = reinterpret_cast<char *>(const_cast<T *>(data.data()))
                    + idx * this->dataSize();
                this->decodeFp16(
                    reinterpret_cast<uint8_t *>(encoded_ptr),
                    reinterpret_cast<float *>(ret_ptr));
            }
            return reinterpret_cast<const T *>(ret_ptr);
        }
        else
        {
            return &data[idx * this->data_dim];
        }
    }

    virtual const T * getDataPtr(idx_t idx) const override
    {
        return getDataPtrImpl(idx);
    }

    virtual T * getDataPtr(idx_t idx) override
    {
        return const_cast<T *>(getDataPtrImpl(idx));
    }

    virtual bool needDataPrefetching() const override
    {
        return this->use_fp16_storage;
    }

    const size_t fp16_prefetch_size_limit = 0x4000;
    virtual int prefetchSizeLimit() const override
    {
        if (this->use_fp16_storage)
            return fp16_prefetch_size_limit;
        else
            return 0;
    }

    virtual PrefetchInfo *
    prefetchData(std::vector<idx_t> idx_list) const override
    {
        if (this->use_fp16_storage)
        {
            auto * p_info = new PrefetchInfo();
            auto * block = mem_manager->getFp32Block();
            block->setDim(this->data_dim);
            p_info->read_buffer = reinterpret_cast<void *>(block);
            block->initializeDecodeBlock(idx_list);
            if (!use_lazy_decode)
            {
                block->decodeFp16(
                    reinterpret_cast<uint8_t *>(const_cast<T *>(data.data())),
                    this->quantizer);
            }
            return p_info;
        }
        return nullptr;
    }

    virtual void releasePrefetch(PrefetchInfo * p_info) const override
    {
        if (this->use_fp16_storage)
        {
            auto * block
                = reinterpret_cast<MemFp32Block *>(p_info->read_buffer);
            mem_manager->releaseFp32Block(block);
            delete p_info;
        }
    }

    virtual void load(IStreamPtr reader, size_t /* num_data */ = -1) override
    {
        size_t npts, dim;
        std::visit(
            [&](auto && r) { load_bin_from_reader(*r, data, npts, dim); },
            reader);
        SI_THROW_IF_NOT_FMT(
            dim << this->scale == this->dataDimension(),
            ErrorCode::LOGICAL_ERROR,
            "Dimension doesn't match: dim=%lu scale=%d "
            "DataLayer::dataDimension=%lu",
            dim,
            this->scale,
            this->dataDimension());
        this->data_num = npts;
        SI_LOG_INFO(
            "DenseMemoryDataLayer::load data_num={} data_dim={} data_size={} "
            "scale={}",
            npts,
            this->dataDimension(),
            this->dataSize(),
            this->scale);
    }

    virtual size_t serialize(OStreamPtr writer) override
    {
        size_t checksum;
        std::visit(
            [&](auto && w)
            {
                save_bin_with_writer(
                    *w,
                    data.data(),
                    this->dataNum(),
                    this->dataDimension() >> this->scale,
                    &checksum);
            },
            writer);
        SI_LOG_INFO(
            "DenseMemoryDataLayer::serialize checksum={} dataDimension={}",
            checksum,
            this->dataDimension());
        return checksum;
    }

protected:
    virtual void addDataImpl(DataChunk & chunk) override
    {
        if (this->use_fp16_storage)
        {
            size_t new_data_size = data.size()
                + (chunk.numData() * chunk.dimension() * sizeof(uint16_t))
                    / sizeof(float);

            checkAvailableMemory(new_data_size);
            data.resize(new_data_size);

            for (int i = 0; i < chunk.numData(); i++)
            {
                auto * current_fp32 = reinterpret_cast<const float *>(
                    reinterpret_cast<const char *>(chunk[i]));
                auto * current_fp16 = reinterpret_cast<uint8_t *>(data.data())
                    + (this->data_num + i) * sizeof(uint16_t) * this->data_dim;
                this->fp16Encode(
                    current_fp32, reinterpret_cast<uint8_t *>(current_fp16), 1);
            }
        }
        else
        {
            checkAvailableMemory(
                chunk.numData() * chunk.dimension() * sizeof(T));
            data.insert(data.end(), chunk.getData(), chunk.dataEnd());
        }

        this->data_num += chunk.numData();
        if (this->use_fp16_storage)
        {
            SI_THROW_IF_NOT(
                this->data_num * this->data_dim * sizeof(uint16_t)
                    == data.size() * sizeof(float),
                ErrorCode::LOGICAL_ERROR);
        }
        else
        {
            SI_THROW_IF_NOT(
                this->data_num * this->data_dim * sizeof(float)
                    == data.size() * sizeof(float),
                ErrorCode::LOGICAL_ERROR);
        }
    }

    std::vector<T> data;
};

}
