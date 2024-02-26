#include "IndexSourceData.h"

namespace Search
{

template <>
size_t DataSet<float>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(float);
}

template <>
size_t DataSet<double>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(double);
}

template <>
size_t DataSet<long>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(long);
}

template <>
size_t DataSet<int>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(int);
}

template <>
size_t DataSet<short>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(short);
}

template <>
size_t DataSet<signed char>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(signed char);
}

template <>
size_t DataSet<unsigned char>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(unsigned char);
}

template <>
size_t DataSet<unsigned short>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(unsigned short);
}

template <>
size_t DataSet<unsigned int>::singleVectorSizeInByte() const
{
    return dimension() * sizeof(unsigned int);
}

template <>
size_t DataSet<bool>::singleVectorSizeInByte() const
{
    return (dimension() + 7) / 8;
}
}
