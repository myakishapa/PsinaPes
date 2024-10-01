#pragma once

#include <memory>

template<typename T>
void Read(const std::byte*& ptr, T& dest)
{
    std::memcpy(&dest, ptr, sizeof(T));
    ptr += sizeof(T);
}
template<typename T>
void Write(std::byte*& ptr, T obj)
{
    std::memcpy(ptr, &obj, sizeof(T));
    ptr += sizeof(T);
}