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
#include <SearchIndex/DiskIOManager.h>
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

/// @brief Dense data layer backed by local disk-based files.
template <typename T>
class DenseMemoryMappedDataLayer : public DenseDataLayer<T>
{
public:
    using DataChunk = DataSet<T>;
    using IStreamPtr = typename DenseDataLayer<T>::IStreamPtr;
    using OStreamPtr = typename DenseDataLayer<T>::OStreamPtr;

    DenseMemoryMappedDataLayer(
        size_t max_data_,
        size_t data_dim_,
        const std::string & file_name_,
        bool reuse_data_file_ = false,
        bool use_fp16_storage_ = false,
        bool init_num_data_ = false) :
        DenseDataLayer<T>(max_data_, data_dim_, use_fp16_storage_),
        file_name(file_name_),
        reuse_data_file(reuse_data_file_)
    {
        SI_THROW_IF_NOT_MSG(
            !use_fp16_storage_,
            ErrorCode::BAD_ARGUMENTS,
            "fp16_storage not support for DenseMemoryMappedDataLayer");
        SI_LOG_INFO(
            "Creating DenseMemoryMappedDataLayer, file_name={}, reuse={}",
            file_name,
            reuse_data_file);
        checkAndOpenFile();
        if (init_num_data_)
            this->data_num = max_data_;
    }

    void munmapAndClose()
    {
        /// release mmap data, no exceptions are thrown
        /// and other resources can be released normally.
        if (data && 0 != munmap(data, file_size))
            SI_LOG_ERROR(
                "DenseMemoryMappedDataLayer: Cannot munmap, file {}, "
                "file_size={}",
                file_name,
                file_size);
        data = nullptr;

        /// Release file descriptors for read and write
        if (fd != -1)
        {
            if (close(fd) == -1)
            {
                SI_LOG_ERROR(
                    "DenseMemoryMappedDataLayer: close "
                    "file {}, "
                    "file_size={} fail, error: {}",
                    file_name.data(),
                    file_size,
                    strerror(errno));
            }
        }
        fd = -1;
    }

    virtual ~DenseMemoryMappedDataLayer() override { munmapAndClose(); }

    /// Remove the file backing the data layer
    virtual void remove() override
    {
        munmapAndClose();
        std::error_code ec;
        std::filesystem::remove(file_name, ec);
        if (ec)
        {
            SI_THROW_FMT(
                ErrorCode::LOGICAL_ERROR,
                "DenseMemoryMappedDataLayer: std::fs::remove Cannot remove, "
                "file %s, "
                "file_size=%lu, ec.value=%u, ec.message=%s",
                file_name.data(),
                file_size,
                ec.value(),
                ec.message().data());
        }
    }

    virtual const T * getDataPtr(idx_t idx) const override
    {
        return &data[idx * this->data_dim * sizeof(T)];
    }

    virtual T * getDataPtr(idx_t idx) override
    {
        return &data[idx * this->data_dim * sizeof(T)];
    }

    virtual bool needDataPrefetching() const override { return false; }

    virtual int prefetchSizeLimit() const override { return 0; }

    virtual PrefetchInfo *
    prefetchData(std::vector<idx_t> /* idx_list */) const override
    {
        return nullptr;
    }

    virtual void releasePrefetch(PrefetchInfo * /* info */) const override { }

    virtual void load(IStreamPtr reader, size_t /* num_data */ = -1) override
    {
        SI_THROW_IF_NOT(!reuse_data_file, ErrorCode::LOGICAL_ERROR);
        std::function<std::span<T>(size_t, size_t)> get_data
            = [&](size_t npts, size_t dim) -> std::span<T>
        {
            SI_THROW_IF_NOT(npts == this->max_data, ErrorCode::LOGICAL_ERROR);
            SI_THROW_IF_NOT(dim == this->data_dim, ErrorCode::LOGICAL_ERROR);
            return std::span<T>(this->getDataPtr(0), npts * dim);
        };
        size_t npts, dim;
        std::visit(
            [&](auto && r) { load_bin_from_reader(*r, get_data, npts, dim); },
            reader);
        this->data_num = npts;
        SI_THROW_IF_NOT(
            this->data_num == this->max_data, ErrorCode::LOGICAL_ERROR);
        SI_LOG_INFO(
            "DenseMemoryMappedDataLayer::load data_num={} dim={} dataSize={}",
            npts,
            this->dataDimension(),
            this->dataSize());
    }

    virtual size_t serialize(OStreamPtr writer) override
    {
        SI_THROW_IF_NOT_FMT(
            this->data_num == this->max_data,
            ErrorCode::LOGICAL_ERROR,
            "DenseMemoryMappedDataLayer data_num %lu max_data %lu must match",
            this->data_num,
            this->max_data);
        size_t checksum;
        std::visit(
            [&](auto && w)
            {
                save_bin_with_writer(
                    *w,
                    data,
                    this->dataNum(),
                    this->dataDimension(),
                    &checksum);
            },
            writer);
        SI_LOG_INFO(
            "DenseMemoryMappedDataLayer::serialize checksum={} "
            "dataDimension={}",
            checksum,
            this->dataDimension());
        return checksum;
    }

protected:
    void checkAndOpenFile()
    {
        SI_LOG_INFO(
            "DenseMemoryMappedDataLayer::checkAndOpenFile {}", file_name);
        int prot;
        if (reuse_data_file)
        {
            // open file for reading
            file_size = std::filesystem::file_size(file_name);
            SI_THROW_IF_NOT(
                file_size % this->dataSize() == 0, ErrorCode::LOGICAL_ERROR);
            this->data_num = file_size / this->dataSize();
            fd = open(file_name.c_str(), O_RDONLY);
            if (fd == -1)
            {
                SI_THROW_FMT(
                    ErrorCode::CANNOT_OPEN_FILE,
                    "DenseMemoryMappedDataLayer::%s open filename %s fd %d "
                    "strerror: %s",
                    __func__,
                    file_name.data(),
                    fd,
                    strerror(errno));
            }
            prot = PROT_READ;
            SI_LOG_INFO(
                "Loading DenseMemoryMappedDataLayer {}, data_num={}",
                file_name,
                this->data_num);
        }
        else
        {
            // create file for reading & writing
            fd = open(file_name.c_str(), O_RDWR | O_CREAT, 0644);
            if (fd == -1)
            {
                SI_THROW_FMT(
                    ErrorCode::CANNOT_OPEN_FILE,
                    "DenseMemoryMappedDataLayer::%s open filename %s fd %d "
                    "strerror: %s",
                    __func__,
                    file_name.data(),
                    fd,
                    strerror(errno));
            }
            SI_THROW_IF_NOT(fd != -1, ErrorCode::CANNOT_OPEN_FILE);
            file_size = this->dataSize() * this->max_data;
            // truncate file to the right size
            auto ret = ftruncate(fd, file_size);
            if (ret == -1)
                perror("ftruncate");
            SI_THROW_IF_NOT_FMT(
                ret != -1,
                ErrorCode::CANNOT_TRUNCATE_FILE,
                "ftruncate ret %d file_size %lu",
                ret,
                file_size);
            prot = PROT_READ | PROT_WRITE;
        }

        for (int retry = 0; retry < 5; ++retry)
        {
            data = reinterpret_cast<T *>(
                mmap(nullptr, file_size, prot, MAP_SHARED, fd, 0));
            if (MAP_FAILED == reinterpret_cast<void *>(data))
            {
                SI_LOG_ERROR(
                    "DenseMemoryMappedDataLayer init data by mmap error, retry "
                    "{}",
                    retry);
                continue;
            }
            SI_LOG_INFO(
                "DenseMemoryMappedDataLayer open file {} read_only={} "
                "file_size={} "
                "fd={} data={:p}",
                file_name,
                reuse_data_file,
                file_size,
                fd,
                reinterpret_cast<void *>(data));
            return;
        }
        /// If the number of retries is exceeded, an exception is thrown.
        SI_THROW_FMT(
            ErrorCode::CANNOT_ALLOCATE_MEMORY,
            "DenseMemoryMappedDataLayer: Cannot mmap, file %s, file_size=%lu, "
            "error %s",
            file_name.c_str(),
            file_size,
            strerror(errno));
    }

    virtual void addDataImpl(DataChunk & chunk) override
    {
        SI_THROW_IF_NOT_MSG(
            this->data_num + chunk.numData() <= this->max_data,
            ErrorCode::LOGICAL_ERROR,
            "DenseMemoryMappedDataLayer exceeding initialization size");
        std::copy(
            chunk.getData(),
            chunk.dataEnd(),
            data + this->dataNum() * this->dataDimension());
        this->data_num += chunk.numData();
    }

private:
    std::string file_name;
    size_t file_size;
    bool reuse_data_file;
    T * data;
    int fd{-1};
};

template <typename T>
class DenseDiskDataLayer : public DenseDataLayer<T>
{
public:
    using DataChunk = DataSet<T>;
    using IStreamPtr = typename DenseDataLayer<T>::IStreamPtr;
    using OStreamPtr = typename DenseDataLayer<T>::OStreamPtr;

    static const int MIN_PAGE_SIZE = 4096;
    static const size_t IO_BLOCK_SIZE = (8UL << 20);

    /** Create a DenseDiskLayer to hold vector data
     *
     * @param max_data_ maximal data to put in data layer
     * @param data_dim_ data dimension
     * @param file_name_ file name of the data layer
     * @param io_manager_ io manager to issue disk IO and prefetch
     * @param page_size_ page size to put the vector data (-1 for automatic),
     *                   no single vector should span two pages.
     */
    DenseDiskDataLayer(
        size_t max_data_,
        size_t data_dim_,
        const std::string & file_name_,
        std::shared_ptr<DiskIOManager> io_manager_,
        bool reuse_data_file_ = false,
        bool use_fp16_storage_ = false,
        int page_size_ = -1) :
        DenseDataLayer<T>(max_data_, data_dim_, use_fp16_storage_),
        file_name(file_name_),
        io_manager(io_manager_),
        page_size(page_size_),
        reuse_data_file(reuse_data_file_)
    {
        SI_LOG_INFO(
            "Creating DenseDiskDataLayer, use_fp16_storage={}",
            use_fp16_storage_);

        if (page_size == -1)
        {
            page_size
                = DIV_ROUND_UP(this->dataSize(), MIN_PAGE_SIZE) * MIN_PAGE_SIZE;
            if (use_fp16_storage_)
                data_per_page = ((page_size / this->dataSize()) / 2) * 2;
            else
                data_per_page = (page_size / this->dataSize());
        }

        SI_LOG_INFO(
            "page_size_: {}, page_size: {}, data_per_page: {}, "
            "this->dataSize(): {}",
            page_size_,
            page_size,
            data_per_page,
            this->dataSize());
        // buffer data in a page before writing out
        page_buffer.resize(page_size * 2, 0);
        max_pages = DIV_ROUND_UP(this->max_data, data_per_page);
        max_file_size = page_size * max_pages;

        // buffer data before writing out
        // TODO further tune buffer size to achieve best performance and minimize overhead
        write_buffer_size = std::max(
            page_size, static_cast<int>(IO_BLOCK_SIZE) / page_size * page_size);
        max_buffer_data = write_buffer_size / page_size * data_per_page;

        if (!reuse_data_file)
        {
            // create a new file to write the data
            file = fopen(file_name.c_str(), "wb");
            if (!file)
            {
                SI_THROW_FMT(
                    ErrorCode::CANNOT_OPEN_FILE,
                    "DenseMemoryMappedDataLayer::%s fopen filename %s strerror "
                    "%s",
                    __func__,
                    file_name.c_str(),
                    strerror(errno));
            }
        }
        else
        {
            // reuse existing file, no need to open a new file for writing
            SI_THROW_IF_NOT_FMT(
                std::filesystem::exists(file_name),
                ErrorCode::LOGICAL_ERROR,
                "DenseDiskDataLayer file %s not exist!",
                file_name.c_str());
        }

        SI_LOG_INFO(
            "Building DenseDiskDataLayer, max_data={} data_dim={} page_size={} "
            "buffer_size={} reuse_data_file={} data_per_page={} file_name={} "
            "max_file_size={}",
            this->max_data,
            this->data_dim,
            page_size,
            write_buffer_size,
            reuse_data_file,
            data_per_page,
            file_name,
            max_file_size);
    }

    virtual ~DenseDiskDataLayer() override { closeFile(); }

    /// Remove the file backing the data layer
    virtual void remove() override
    {
        closeFile();
        std::error_code ec;
        std::filesystem::remove(file_name, ec);
        if (ec)
        {
            SI_THROW_FMT(
                ErrorCode::LOGICAL_ERROR,
                "DenseDiskDataLayer: std::fs::remove Cannot remove, "
                "file %s, "
                "max_file_size=%lu, ec.value=%u, ec.message=%s",
                file_name.data(),
                max_file_size,
                ec.value(),
                ec.message().data());
        }
    }

    void closeFile()
    {
        /// Release file descriptors for read and write
        if (fd != -1)
        {
            if (close(fd) == -1)
            {
                /// c++ not suggest throw exception in destructor,
                /// https://cplusplus.com/forum/general/253853/#:~:text=You%20can%20throw%20an%20exception,language%20itself%20will%20be%20violated.

                SI_LOG_ERROR(
                    "DenseDiskDataLayer::{} close filename {} fd {} "
                    "strerror: {}",
                    __func__,
                    file_name,
                    fd,
                    strerror(errno));
            }
        }
        if (file)
        {
            if (fclose(file))
            {
                SI_LOG_ERROR(
                    "DenseDiskDataLayer::{} fclose filename {}"
                    "strerror: {}",
                    __func__,
                    file_name,
                    strerror(errno));
            }
        }
    }

    virtual bool needDataPrefetching() const override { return true; }

    virtual int prefetchSizeLimit() const override
    {
        return io_manager->BUFFER_SIZE / page_size;
    }

    // must prefetch data before reading them
    virtual PrefetchInfo *
    prefetchData(std::vector<idx_t> idx_list) const override
    {
        return prefetchDataImpl(idx_list, true);
    }

    PrefetchInfo *
    prefetchDataImpl(std::vector<idx_t> idx_list, bool should_decode_fp16) const
    {
        checkAndOpenFile();
        if (idx_list.empty())
        {
            // skip when no data needs to be prefetched
            return nullptr;
        }
        SI_LOG_DEBUG(
            "DiskLayer prefetchData, idx_list.size={}", idx_list.size());
        // how many items to prefetch at once
        SI_THROW_IF_NOT_FMT(
            idx_list.size() <= prefetchSizeLimit(),
            ErrorCode::LOGICAL_ERROR,
            "Prefetch data size %lu exceeds limit %d",
            idx_list.size(),
            prefetchSizeLimit());

        auto t0 = std::chrono::high_resolution_clock::now();
        auto num_prefetch = idx_list.size();
        auto p_info = std::make_shared<PrefetchInfo>();
        // this might block for a while
        p_info->read_buffer = io_manager->takeBuffer();
        auto t1 = std::chrono::high_resolution_clock::now();
        p_info->prefetch_id_list = std::move(idx_list);
        performBatchPrefetch(*p_info);

        auto * uring_buffer
            = reinterpret_cast<IOUringReadBuffer *>(p_info->read_buffer);

        std::unordered_set<size_t> id_sets;
        for (const auto & item : p_info->page_to_read_idx)
        {
            auto page_id = item.first;
            id_sets.insert(page_id);
        }
        int page_number = id_sets.size();

        if (this->use_fp16_storage && should_decode_fp16)
        {
            this->decodeFp16Vectors(page_number, uring_buffer);
        }

        std::lock_guard<std::mutex> prefetch_info_lock(prefetch_mutex);
        prefetch_info_map[p_info.get()] = p_info;
        auto t2 = std::chrono::high_resolution_clock::now();
        auto wait_ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0)
                  .count();
        auto time_ms
            = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t0)
                  .count();
        SI_LOG_DEBUG(
            "DenseDiskDataLayer::prefetchData() num_data={} wait_time={} "
            "total_time={}",
            num_prefetch,
            wait_ms,
            time_ms);
        return p_info.get();
    }

    void
    decodeFp16Vectors(int page_number, IOUringReadBuffer * uring_buffer) const
    {
        std::vector<char> tmp_vec_buffer(page_number * page_size);
        uint8_t * tmp_buffer
            = reinterpret_cast<uint8_t *>(tmp_vec_buffer.data());
        memcpy(
            tmp_vec_buffer.data(),
            uring_buffer->data(),
            page_number * page_size);

        for (int i = 0; i < page_number; i++)
        {
            size_t before_page_offset = i * page_size;
            char * after_page_offset1
                = uring_buffer->data() + (i << 1) * page_size;
            char * after_page_offset2
                = uring_buffer->data() + ((i << 1) + 1) * page_size;

            for (int j = 0; j < data_per_page / 2; j++)
            {
                float * addr1 = reinterpret_cast<float *>(
                    after_page_offset1 + j * this->data_dim * sizeof(float));
                uint8_t * org_addr1 = tmp_buffer + before_page_offset
                    + j * this->data_dim * sizeof(uint16_t);
                this->decodeFp16(reinterpret_cast<uint8_t *>(org_addr1), addr1);
            }
            for (int j = 0; j < data_per_page / 2; j++)
            {
                float * addr2 = reinterpret_cast<float *>(
                    after_page_offset2 + j * this->data_dim * sizeof(float));
                uint8_t * org_addr2 = tmp_buffer + before_page_offset
                    + (data_per_page / 2 + j) * this->data_dim
                        * sizeof(uint16_t);
                this->decodeFp16(org_addr2, addr2);
            }
        }
    }

    virtual void releasePrefetch(PrefetchInfo * info) const override
    {
        // nothing to do here
        if (info == nullptr)
            return;
        // TODO simplify this logic
        std::lock_guard<std::mutex> prefetch_info_lock(prefetch_mutex);
        auto it = prefetch_info_map.find(info);
        SI_THROW_IF_NOT_MSG(
            it != prefetch_info_map.end(),
            ErrorCode::LOGICAL_ERROR,
            "IOManager must contain prefetch_info for current thread to "
            "release the prefetched data");
        auto read_buffer
            = reinterpret_cast<IOUringReadBuffer *>(it->second->read_buffer);
        io_manager->putBuffer(read_buffer);
        prefetch_info_map.erase(it);
    }

    virtual const T * getDataPtr(idx_t idx) const override
    {
        return getDataPtrImpl(idx, this->scale);
    }

    virtual T * getDataPtr(idx_t idx) override
    {
        return const_cast<T *>(getDataPtrImpl(idx, this->scale));
    }

    const T * getDataPtrImpl(idx_t idx, int scale_) const
    {
        // TODO might have NPE error here if not careful
        auto p_info = this->thread_prefetch_info;
        if (!p_info)
        {
            SI_LOG_FATAL(
                "DiskLayer getDataPtr without prefetch, idx={} type={} "
                "dimension={} size={}",
                idx,
                typeid(T).name(),
                this->dataDimension(),
                this->dataNum());
        }
        SI_THROW_IF_NOT(p_info, ErrorCode::LOGICAL_ERROR);
        // the data must have been prefetched already
        auto it = p_info->page_to_read_idx.find(idx / data_per_page);
        SI_THROW_IF_NOT(
            it != p_info->page_to_read_idx.end(), ErrorCode::LOGICAL_ERROR);
        auto read_buffer
            = reinterpret_cast<IOUringReadBuffer *>(p_info->read_buffer);
        if (this->use_fp16_storage && scale_)
        {
            int vec_index = 0;
            int page_index = 0;
            int org_vec_index = idx % data_per_page;
            page_index = it->second << 1;
            vec_index = org_vec_index;
            if (org_vec_index >= (data_per_page >> 1))
            {
                vec_index -= (data_per_page >> 1);
                page_index++;
            }
            char * data = read_buffer->data() + (page_size)*page_index
                + (vec_index) * (this->dataSize() << this->scale);
            return reinterpret_cast<T *>(data);
        }
        else
        {
            int real_page_size = data_per_page;
            int vec_index = idx % real_page_size;
            char * data = read_buffer->data() + page_size * it->second
                + vec_index * this->dataSize();
            return reinterpret_cast<T *>(data);
        }
    }

    virtual void load(IStreamPtr istream, size_t num_data = -1) override
    {
        if (reuse_data_file)
        {
            // skip loading
            SI_THROW_IF_NOT(num_data > 0, ErrorCode::LOGICAL_ERROR);
            SI_LOG_INFO(
                "DenseDiskDataLayer.reuse_data_file=true, skip loading, "
                "num_data={}",
                num_data);
            this->data_num = num_data;

            auto actual_file_size = std::filesystem::file_size(file_name);
            auto expected_file_size
                = DIV_ROUND_UP(num_data, max_buffer_data) * write_buffer_size;
            SI_THROW_IF_NOT_FMT(
                actual_file_size == expected_file_size,
                ErrorCode::LOGICAL_ERROR,
                "DenseDiskDataLayer %s file_size %lu != %lu",
                file_name.c_str(),
                actual_file_size,
                expected_file_size);
            return;
        }
        auto load_func = [&](auto * in_stream)
        {
            using IS = std::remove_pointer_t<decltype(in_stream)>;
            StreamBinReader<T, IS> data_reader(*in_stream);
            SI_THROW_IF_NOT_FMT(
                data_reader.ndimsValue() << this->scale
                    == this->dataDimension(),
                ErrorCode::LOGICAL_ERROR,
                "data_reader.ndimsValue %lu dataDimension %lu",
                data_reader.ndimsValue(),
                this->dataDimension());
            SI_LOG_INFO("data_reader.nptsValue {}", data_reader.nptsValue());
            size_t batch_size = (data_reader.IO_BLOCK_SIZE) / this->dataSize();
            std::vector<T> buffer(batch_size * this->dataDimension());
            for (int st = 0; st < data_reader.nptsValue(); st += batch_size)
            {
                size_t num_read = data_reader.loadData(
                    buffer.data(),
                    ((batch_size * this->dataDimension()) >> this->scale));
                SI_THROW_IF_NOT_FMT(
                    num_read % (this->dataDimension() >> this->scale) == 0,
                    ErrorCode::LOGICAL_ERROR,
                    "num_read %lu this->dataDimension %lu ",
                    num_read,
                    this->dataDimension());
                auto chunk = std::make_shared<DataChunk>(
                    buffer.data(),
                    num_read / (this->dataDimension() >> this->scale),
                    this->dataDimension() >> this->scale);
                this->performAddingData(*chunk, true);
            }
            this->seal();

            // check that the number of data matches
            SI_THROW_IF_NOT_FMT(
                data_reader.nptsValue() == this->dataNum(),
                ErrorCode::LOGICAL_ERROR,
                "data_reader npts=%lu, data_num=%lu",
                data_reader.nptsValue(),
                this->dataNum());
        };
        std::visit(load_func, istream);
    }

    virtual size_t serialize(OStreamPtr ostream) override
    {
        auto serialize_func = [&](auto * out_stream) -> size_t
        {
            using OS = std::remove_pointer_t<decltype(out_stream)>;
            StreamBinWriter<T, OS> data_writer(
                *out_stream,
                this->dataNum(),
                this->dataDimension() >> this->scale,
                true);
            SI_THROW_IF_NOT(
                this->dataSize() <= this->page_size, ErrorCode::LOGICAL_ERROR);
            size_t batch_size = this->prefetchSizeLimit();

            for (int st = 0; st < this->dataNum(); st += batch_size)
            {
                // issue prefetch
                std::vector<idx_t> prefetch_list;
                for (int i = st; i < this->dataNum() && i < st + batch_size;
                     ++i)
                    prefetch_list.push_back(i);
                auto prefetch_info = prefetchDataImpl(prefetch_list, false);
                this->setThreadPrefetchInfo(prefetch_info);
                // write to ostream
                for (auto i : prefetch_list)
                    data_writer.writeData(
                        getDataPtrImpl(i, 0),
                        this->dataDimension() >> this->scale);
                this->setThreadPrefetchInfo(nullptr);
                this->releasePrefetch(prefetch_info);
            }

            data_writer.finish();
            size_t checksum = data_writer.hashValue();
            SI_LOG_INFO("DenseDiskDataLayer::serialize checksum={}", checksum);
            return checksum;
        };
        return std::visit(serialize_func, ostream);
    }

    virtual void seal() override
    {
        // release the write buffer
        write_buffer.clear();
    }

protected:
    void checkAndOpenFile() const
    {
        std::lock_guard<std::mutex> prefetch_info_lock(prefetch_mutex);
        if (fd < 0)
        {
            // use for async reading
            auto file_size = std::filesystem::file_size(file_name);
            fd = open(file_name.c_str(), O_RDONLY | O_DIRECT);
            SI_LOG_INFO(
                "Open file {} for prefetching, file_size={}, fd={}",
                file_name,
                file_size,
                fd);
        }
    }

    void performBatchPrefetch(PrefetchInfo & p_info) const
    {
        // prefetch one page at a time
        std::vector<FileReadReq> reads;
        p_info.page_to_read_idx.clear();
        for (auto idx : p_info.prefetch_id_list)
        {
            size_t page_idx = idx / data_per_page;
            if (!p_info.page_to_read_idx.contains(page_idx))
            {
                reads.push_back(FileReadReq{fd, page_idx * page_size});
                p_info.page_to_read_idx[page_idx] = reads.size() - 1;
            }
        }
        auto * buffer
            = reinterpret_cast<IOUringReadBuffer *>(p_info.read_buffer);
        buffer->batchRead(this->page_size, reads);
    }


    virtual void addDataImpl(DataChunk & chunk) override
    {
        performAddingData(chunk);
    }

    void performAddingData(DataChunk & chunk, bool chunk_is_fp16 = false)
    {
        SI_THROW_IF_NOT_FMT(
            !reuse_data_file && file != nullptr,
            ErrorCode::LOGICAL_ERROR,
            "can't add data when %s is not open for writing",
            file_name.c_str());
        // allocate write_buffer on the fly
        if (write_buffer.empty())
            write_buffer.resize(write_buffer_size, 0);
        for (int i = 0; i < chunk.numData();)
        {
            size_t batch_size = std::min(
                static_cast<size_t>(max_buffer_data - num_buffer_data),
                chunk.numData() - i);
            for (size_t j = 0; j < batch_size; ++j)
            {
                const char * cur_data
                    = reinterpret_cast<const char *>(chunk[i + j]);
                size_t buffer_offset
                    = (num_buffer_data + j) / data_per_page * page_size
                    + ((num_buffer_data + j) % data_per_page)
                        * this->dataSize();
                char fp16_buffer[this->dataSize()];
                const char * data_ptr;
                if (this->use_fp16_storage && !chunk_is_fp16)
                {
                    this->fp16Encode(
                        reinterpret_cast<const float *>(cur_data),
                        reinterpret_cast<uint8_t *>(fp16_buffer),
                        1);
                    data_ptr = fp16_buffer;
                }
                else
                {
                    data_ptr = reinterpret_cast<const char *>(cur_data);
                }
                std::copy(
                    data_ptr,
                    data_ptr + this->dataSize(),
                    write_buffer.data() + buffer_offset);
            }
            num_buffer_data += batch_size;
            i += batch_size;

            if (num_buffer_data == max_buffer_data)
            {
                // flush the write buffer
                auto ret
                    = fwrite(write_buffer.data(), 1, write_buffer.size(), file);
                SI_THROW_IF_NOT(
                    ret == write_buffer_size,
                    ErrorCode::CANNOT_WRITE_TO_OSTREAM);
                num_buffer_data = 0;
                num_completed_buffer += 1;
            }
        }
        if (num_buffer_data > 0)
        {
            // write out the remaining data with an incomplete write buffer
            SI_THROW_IF_NOT(
                num_buffer_data < max_buffer_data, ErrorCode::LOGICAL_ERROR);
            auto ret
                = fwrite(write_buffer.data(), 1, write_buffer.size(), file);
            SI_THROW_IF_NOT(
                ret == write_buffer_size, ErrorCode::CANNOT_WRITE_TO_OSTREAM);
            // seek to previous location to finish writing the page later
            fseek(file, num_completed_buffer * write_buffer_size, SEEK_SET);
        }
        fflush(file);
        this->data_num += chunk.numData();
        //        SI_LOG_INFO("data_num {} chunk.numData {} chunk.dimension {} this->dataSize {} chunk_is_fp16 {} this->dataDimension {}",
        //                    this->data_num, chunk.numData(), chunk.dimension(), this->dataSize(), chunk_is_fp16, this->dataDimension());
    }

    void dumpVectors(float * vec_ptr, int vector_number)
    {
        std::string print_str;
        for (int vec_index = 0; vec_index < vector_number; vec_index++)
        {
            float * buffer = reinterpret_cast<float *>(vec_ptr)
                + this->data_dim * vec_index;
            for (int dim = 0; dim < this->data_dim; dim++)
            {
                print_str
                    += std::to_string(static_cast<int>(buffer[dim])) + ", ";
            }
            print_str += '\n';
        }
        SI_LOG_INFO("{}", print_str);
    }

    void dumpVector(float * begin_ptr) const
    {
        std::string print_str;
        for (int d = 0; d < this->data_dim; d++)
        {
            float * ptr = begin_ptr + d;
            print_str += std::to_string(*ptr) + ", ";
        }
        SI_LOG_INFO("{}", print_str);
    }

    void dumpWholeRange(
        size_t max_page_number, size_t min_page_number, float * begin_ptr)
    {
        std::string print_str;
        for (int i = max_page_number - min_page_number; i >= 0; i--)
        {
            size_t after_page_offset = (i * page_size) << 1;
            for (int j = data_per_page - 1; j >= 0; j--)
            {
                for (int d = 0; d < this->data_dim; d++)
                {
                    float * ptr = reinterpret_cast<float *>(
                                      begin_ptr + after_page_offset)
                        + d;
                    print_str += std::to_string(static_cast<int>(*ptr)) + ", ";
                }
                print_str += "\n";
            }
        }
        SI_LOG_INFO("{}", print_str);
    }

    std::vector<char> page_buffer;
    std::string file_name;
    FILE * file{nullptr};
    int page_size;
    bool reuse_data_file;
    int data_per_page;
    size_t max_pages;
    size_t max_file_size;

    std::vector<char> write_buffer;
    int write_buffer_size{-1};
    int max_buffer_data{0};
    int num_buffer_data{0};
    // how many complete pages have been written out
    size_t num_completed_buffer{0};

    // only data is viewed as states of the layer and these are viewed as mutable variables
    mutable std::atomic<int> fd{-1};
    mutable std::shared_ptr<DiskIOManager> io_manager;
    mutable std::mutex prefetch_mutex;
    mutable std::map<PrefetchInfo *, std::shared_ptr<PrefetchInfo>>
        prefetch_info_map;
};


}
