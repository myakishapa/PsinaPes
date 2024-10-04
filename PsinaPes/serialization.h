#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <map>
#include <utility>
#include <span>
#include <concepts>
#include <type_traits>
#include <numeric>
#include <ranges>
#include <expected>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <string_view>  
#include <filesystem>
#include <concepts>
#include <chrono>
#include <functional>

#include "binary_io_utility.h"
#include "ConstexprHash.h"

using namespace std::literals;

namespace fs = std::filesystem;
namespace chr = std::chrono;

template<typename Type>
struct HvValueConvert
{
    static std::uint64_t GetType()
    {
        return ~0ui64;
    }

    static bool HvToType(const std::byte* data, std::size_t size, Type& value)
    {
        return false;
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const Type& obj)
    {
        return false;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<Type>& array)
    {
        return false;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const Type> array)
    {
        return false;
    }
};

constexpr std::uint64_t HvArrayType()
{
    return Hash("hv_array");
}

std::uint64_t HashCombine(std::uint64_t f, std::uint64_t s)
{
    //seed ^= hasher(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    return f ^ (s + 0x9e3779b9 + (f << 6) + (f >> 2));
}

struct HvRawBinaryView
{
    const std::byte* data;
    std::size_t size;
};

struct HvTreePath
{
    std::vector<std::string> path;

    HvTreePath() {}
    HvTreePath(std::string str)
    {
        Parse(str);
    }

    void Parse(std::string str)
    {
        for (auto id : str | std::views::split('/'))
            path.emplace_back(std::from_range, id);
    }

    std::string ToString() const
    {
        return std::string(std::from_range, path | std::views::join_with('/'));
    }
    operator std::string() const
    {
        return ToString();
    }

    std::string Front() const
    {
        if (path.size()) return path[0];
        else return "";
    }
    std::string ExtractFront()
    {
        if (!path.size()) return "";

        auto copy = path[0];
        path.erase(path.begin());

        return copy;
    }

    operator bool() const
    {
        return !path.empty();
    }

    friend std::strong_ordering operator<=>(const HvTreePath& lhs, const HvTreePath& rhs) = default;

    HvTreePath operator/(HvTreePath rhs)
    {
        HvTreePath copy = *this;
        copy.path.append_range(rhs.path);
        return copy;
    }
    HvTreePath& operator/=(HvTreePath rhs)
    {
        path.append_range(rhs.path);
        return *this;
    }

    std::size_t Size() const
    {
        return path.size();
    }

    std::string& operator[](std::size_t index)
    {
        return path[index];
    }
    const std::string& operator[](std::size_t index) const
    {
        return path[index];
    }

    bool Match(const HvTreePath& generic) const
    {
        if (Size() < generic.Size()) return false;
        for (std::size_t i = 0; i < generic.Size(); i++)
        {
            auto& index = generic[i];
            if (index == "*") continue;
            if (path[i] != generic[i]) return false;
        }
        return true;
    }
};
HvTreePath operator ""_tp(const char* str, std::size_t size)
{
    return HvTreePath(str);
}

auto begin(HvTreePath& path)
{
    return path.path.begin();
}
auto end(HvTreePath& path)
{
    return path.path.end();
}
auto begin(const HvTreePath& path)
{
    return path.path.cbegin();
}
auto end(const HvTreePath& path)
{
    return path.path.cend();
}

struct HvBool
{
    std::uint8_t value;

    HvBool(std::uint8_t value) : value(value) {}
    HvBool(bool value) : value(value) {}
    HvBool(VkBool32 value) : value(value) {}
    HvBool() : value(0) {}

    operator bool() const
    {
        return value;
    }
    operator VkBool32() const
    {
        return value;
    }
};

class HvTree
{
    friend void VisualizeTree(HvTree&, std::string);

    std::uint64_t size;
    std::uint64_t type;
    std::byte* data;

    std::map<std::string, HvTree> children;

public:

    auto begin()
    {
        return children.begin();
    }
    auto end()
    {
        return children.end();
    }
    auto begin() const
    {
        return children.cbegin();
    }
    auto end() const
    {
        return children.cend();
    }


    HvTree() : size(0), type(~0ui64), data(nullptr) {}


    HvTree(const HvTree& rhs)
    {
        if (rhs.Empty())
        {
            size = 0;
            type = ~0ui64;
            data = nullptr;
        }
        else
        {
            size = rhs.size;
            type = rhs.type;
            data = new std::byte[size];
            std::memcpy(data, rhs.data, size);

            children = rhs.children;
        }
    }
    HvTree(HvTree&& rhs) noexcept
    {
        if (rhs.Empty())
        {
            size = 0;
            type = ~0ui64;
            data = nullptr;
        }
        else
        {
            size = rhs.size;
            type = rhs.type;
            data = rhs.data;
            children = std::move(rhs.children);

            rhs.data = nullptr;
            rhs.type = ~0ui64;
            rhs.size = 0;
        }
    }

    void LoadNode(const std::byte*& structureData, const std::byte* actualData)
    {
        Read(structureData, size);
        Read(structureData, type);

        std::uint64_t offset;
        Read(structureData, offset);
        if (~type)
        {
            data = new std::byte[size];
            std::memcpy(data, actualData + offset, size);
        }
        else
            data = nullptr;

        std::uint64_t childCount;
        Read(structureData, childCount);

        for (std::uint64_t i = 0; i < childCount; i++)
        {
            std::uint64_t nameLength;
            Read(structureData, nameLength);

            std::string name(reinterpret_cast<const char*>(structureData), nameLength);
            structureData += nameLength;

            auto& newChild = children[name];
            newChild.LoadNode(structureData, actualData);
        }
    }
    void LoadBinary(const std::byte* data, std::size_t size)
    {
        auto ptr = data;

        std::uint64_t offset;
        Read(ptr, offset);

        LoadNode(ptr, data + offset);
    }
    bool LoadFromFile(fs::path file)
    {
        if (!fs::is_regular_file(file)) return false;

        std::ifstream in(file, std::ios::binary);
        std::size_t fileSize = fs::file_size(file);
        std::byte* data = new std::byte[fileSize];

        in.read(reinterpret_cast<char*>(data), fileSize);
        LoadBinary(data, fileSize);
        delete[] data;
        return true;
    }

    void DestroyIfExists()&
    {
        if (data)
        {
            auto h = this;
            delete[] data;
            data = nullptr;
            size = 0;
            type = ~0ui64;
        }
    }

    template<typename Type>
    bool Store(const Type& obj)
    {
        DestroyIfExists();

        if (HvValueConvert<Type>::TypeToHv(data, size, obj))
        {
            type = HvValueConvert<Type>::GetType();
            return true;
        }
        else
        {
            type = ~0ui64;
            return false;
        }
    }
    template<typename ArrayElement>
    bool StoreArray(std::span<const ArrayElement> arr)
    {
        DestroyIfExists();
        if (HvValueConvert<ArrayElement>::TypeArrayToHv(data, size, arr))
        {
            type = HashCombine(HvValueConvert<ArrayElement>::GetType(), HvArrayType());
            return true;
        }
        else
        {
            type = ~0ui64;
            return false;
        }
    }


    bool Store(const void* rawData, std::size_t rawDataSize)
    {
        DestroyIfExists();

        static std::hash<std::string> hasher;
        static auto rawBinaryType = hasher("raw_binary");

        data = new std::byte[rawDataSize];
        std::memcpy(data, rawData, rawDataSize);
        size = rawDataSize;
        type = rawBinaryType;

        return true;
    }

    struct TakeOwnershipTag {};
    static constexpr TakeOwnershipTag TakeOwnership;

    bool Store(TakeOwnershipTag, void* rawData, std::size_t rawDataSize)
    {
        DestroyIfExists();

        static std::hash<std::string> hasher;
        static auto rawBinaryType = hasher("raw_binary");

        data = reinterpret_cast<std::byte*>(rawData);
        size = rawDataSize;
        type = rawBinaryType;

        return true;
    }

    enum class HvTreeAcquireError
    {
        WRONG_TYPE,
        NODE_EMPTY,
        CONVERTION_ERROR
    };

    template<typename Type>
    std::expected<Type, HvTreeAcquireError> Acquire() const
    {
        if (!data)
            return std::unexpected(HvTreeAcquireError::NODE_EMPTY);
        if (type != HvValueConvert<Type>::GetType())
            return std::unexpected(HvTreeAcquireError::WRONG_TYPE);

        Type result;
        if (HvValueConvert<Type>::HvToType(data, size, result)) return result;
        else return std::unexpected(HvTreeAcquireError::CONVERTION_ERROR);
    }
    template<>
    std::expected<HvRawBinaryView, HvTreeAcquireError> Acquire<HvRawBinaryView>() const
    {
        if (!data)
            return std::unexpected(HvTreeAcquireError::NODE_EMPTY);

        HvRawBinaryView view;
        view.data = data;
        view.size = size;
        return view;
    }
    template<typename ArrayElement>
    std::expected<std::vector<ArrayElement>, HvTreeAcquireError> AcquireArray() const
    {
        if (!data)
            return std::unexpected(HvTreeAcquireError::NODE_EMPTY);
        if (type != HashCombine(HvValueConvert<ArrayElement>::GetType(), HvArrayType()))
            return std::unexpected(HvTreeAcquireError::WRONG_TYPE);

        std::vector<ArrayElement> result;
        if (HvValueConvert<ArrayElement>::HvToTypeArray(data, size, result)) return result;
        else return std::unexpected(HvTreeAcquireError::CONVERTION_ERROR);
    }

    template<typename Type>
    Type AcquireOr(Type defaultValue = Type()) const
    {
        auto expected = Acquire<Type>();
        if (expected.has_value()) return expected.value();
        else return defaultValue;
    }

    bool HasValue() const
    {
        return data;
    }
    bool Empty() const
    {
        return !data && children.empty();
    }
    std::size_t CountChildren() const
    {
        return children.size();
    }

    std::size_t RecursiveBinaryStructureSize() const
    {
        std::size_t result = 32;
        for (auto& child : children)
        {
            result += 8;
            result += child.first.size();
            result += child.second.RecursiveBinaryStructureSize();
        }
        return result;
    }
    std::size_t RecursiveBinaryDataSize() const
    {
        std::size_t result = size;
        for (auto& child : children)
            result += child.second.RecursiveBinaryDataSize();
        return result;
    }

    void SaveNode(std::byte*& structureData, std::byte* actualData, std::size_t& actualDataOffset) const
    {
        Write(structureData, size);
        Write(structureData, type);

        Write(structureData, actualDataOffset);

        std::memcpy(actualData + actualDataOffset, data, size);

        actualDataOffset += size;

        std::uint64_t childCount = children.size();
        Write(structureData, childCount);

        for (auto& child : children)
        {
            std::uint64_t nameLength = child.first.length();
            Write(structureData, nameLength);

            std::memcpy(structureData, child.first.data(), nameLength);
            structureData += nameLength;

            child.second.SaveNode(structureData, actualData, actualDataOffset);
        }
    }
    void SaveBinary(std::byte*& outData, std::size_t& outSize) const
    {
        auto structureSize = RecursiveBinaryStructureSize();
        auto dataSize = RecursiveBinaryDataSize();

        outSize = structureSize + dataSize + 8;
        outData = new std::byte[outSize];

        auto ptr = outData;
        Write(ptr, structureSize + 8ui64);

        std::size_t offset = 0;
        SaveNode(ptr, outData + structureSize + 8, offset);
    }
    bool SaveToFile(fs::path file)
    {
        std::ofstream out(file, std::ios::binary);
        if (!out) return false;

        std::byte* data;
        std::size_t size;
        SaveBinary(data, size);
        out.write(reinterpret_cast<const char*>(data), size);
        delete[] data;
        return true;
    }

    HvTree& Subtree(const std::string& key)
    {
        return children.try_emplace(key, this).first->second;
    }
    const HvTree& Subtree(const std::string& key) const
    {
        return children.at(key);
    }

    auto Subtree(this auto& self, HvTreePath path) -> decltype(self)
    {
        std::add_pointer_t<decltype(self)> current = &self;
        for (const auto& id : path.path)
            current = &current->Subtree(id);

        return *current;
    }

    template<typename KeyType>
    auto operator[](this auto& self, KeyType&& key) -> decltype(self)
    {
        return self.Subtree(std::forward<KeyType>(key));
    }

    ~HvTree()
    {
        DestroyIfExists();
    }

    HvTree& operator=(const HvTree& rhs)
    {
        if (rhs.Empty()) return *this;
        if (&rhs == this) return *this;

        DestroyIfExists();

        children = rhs.children;
        size = rhs.size;
        type = rhs.type;
        data = new std::byte[size];
        std::memcpy(data, rhs.data, size);
        return *this;
    }
    HvTree& operator=(HvTree&& rhs) noexcept
    {
        if (rhs.Empty()) return *this;

        DestroyIfExists();

        size = rhs.size;
        type = rhs.type;
        data = rhs.data;
        children = std::move(rhs.children);

        rhs.data = nullptr;
        rhs.type = ~0ui64;
        rhs.size = 0;

        return *this;
    }

    HvTree Extract(std::string key)
    {
        return std::move(children.extract(key).mapped());
    }
    void ClearChildren()
    {
        children.clear();
    }

    bool Exists(HvTreePath path) const
    {
        if (!path) return true;
        auto front = path.ExtractFront();
        return children.contains(front) ? children.at(front).Exists(path) : false;
    }

    template<typename T>
    operator T() const
    {
        return Acquire<T>().value();
    }
    template<typename T>
    operator std::vector<T>() const
    {
        return AcquireArray<T>().value();
    }

    template<typename T>
    HvTree(const T& value) : size(0), type(~0ui64), data(nullptr)
    {
        Store<T>(value);
    }
    template<typename T>
    HvTree(const std::vector<T>& value) : size(0), type(~0ui64), data(nullptr)
    {
        StoreArray<T>(value);
    }

    template<typename T>
    HvTree& operator=(const T& value)
    {
        Store<T>(value);
        return *this;
    }
    template<typename T>
    HvTree& operator=(const std::vector<T>& value)
    {
        StoreArray<T>(value);
        return *this;
    }

    HvTree& PushArray()
    {
        return Subtree(std::to_string(CountChildren()));
    }

    static constexpr std::size_t UnlimitedDepth = std::numeric_limits<std::size_t>::max();
    void CopyFrom(const HvTree& rhs, std::size_t maxDepth = UnlimitedDepth)
    {
        if (rhs.Empty())
        {
            size = 0;
            type = ~0ui64;
            data = nullptr;
        }
        else
        {
            size = rhs.size;
            type = rhs.type;
            data = new std::byte[size];
            std::memcpy(data, rhs.data, size);
        }

        if (!maxDepth) return;

        for (auto& [index, tree] : rhs)
        {
            children[index].CopyFrom(tree, maxDepth - 1);
        }
    }

    void CopyIf(const HvTree& rhs, std::function<bool(HvTreePath)> pred, HvTreePath basePath = ""_tp, std::size_t maxDepth = UnlimitedDepth)
    {
        if (rhs.Empty())
        {
            size = 0;
            type = ~0ui64;
            data = nullptr;
        }
        else
        {
            size = rhs.size;
            type = rhs.type;
            data = new std::byte[size];
            std::memcpy(data, rhs.data, size);
        }

        if (!maxDepth) return;

        for (auto& [index, tree] : rhs)
        {
            if (pred(basePath / index)) children[index].CopyIf(tree, pred, basePath / index, maxDepth - 1);
        }
    }
};


void VisualizeTree(HvTree& dst, std::string prefix = "")
{    
    if (dst.data)
    {
        std::cout << std::dec << prefix << "Node data(size: " << dst.size << ", type: " << dst.type << "); ";
        if (dst.size <= 8)
        {
            std::cout << "\n" << prefix << "Hex decode: 0x" << std::noshowbase << std::hex;
            for (std::size_t i = 0; i < dst.size; i++)
                std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(*reinterpret_cast<std::uint8_t*>(dst.data + i));
            std::cout << std::endl;
        }
        else
            std::cout << std::endl;
    }
    else
        std::cout << prefix << "No node data; " << std::endl;

    if (dst.children.empty())
        std::cout << prefix << "No children; " << std::endl;
    else
        std::cout << prefix << std::dec << dst.children.size() << " children; " << std::endl;

    prefix += "--";

    for (auto &child : dst.children)
    {
        std::cout << prefix << child.first << std::endl;
        VisualizeTree(child.second, prefix + "--");
    }
}

template<typename Type>
concept NumberConcept = std::floating_point<Type> || std::integral<Type>;

template<std::integral IntType>
struct HvValueConvert<IntType>
{
    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_cpp_int"sv);
    }
    
    static bool HvToType(const std::byte* data, std::size_t size, IntType& value)
    {
        if(size < sizeof(IntType))
            return false;
        std::memcpy(&value, data, sizeof(IntType));
        return true;
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const IntType& obj)
    {
        data = reinterpret_cast<std::byte*>(new IntType(obj));
        size = sizeof(IntType);
        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<IntType>& array)
    {
        array.resize(size / sizeof(IntType));
        std::memcpy(array.data(), data, size);
        return true;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const IntType> array)
    {
        size = array.size() * sizeof(IntType);
        data = new std::byte[size];
        std::memcpy(data, array.data(), size);
        return true;
    }
};

template<>
struct HvValueConvert<std::string>
{
    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_stl_string"sv);
    }

    static bool HvToType(const std::byte* data, std::size_t size, std::string& value)
    {
        if (size < sizeof(std::size_t))
            return false;

        std::size_t strSize;
        Read(data, strSize);

        if (size - sizeof(std::size_t) < strSize)
            return false;

        value.assign(reinterpret_cast<const char*>(data), strSize);
        return true;
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const std::string& obj)
    {
        size = obj.size() + sizeof(std::size_t);
        data = new std::byte[size];

        auto ptr = data;
        Write(ptr, obj.size());
        std::memcpy(ptr, obj.data(), obj.size());
        
        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<std::string>& array)
    {
        std::uint64_t count;
        Read(data, count);
        array.resize(size);
        for (std::size_t i = 0; i < count; i++)
        {
            std::size_t strSize;
            Read(data, strSize);

            array[i].assign(reinterpret_cast<const char*>(data), strSize);
            data += strSize;
        }
        return true;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const std::string> array)
    {
        size = 8;
        for (auto& str : array)
        {
            size += 8;
            size += str.size();
        }

        data = new std::byte[size];
        auto ptr = data;

        Write(ptr, array.size());

        for (auto& str : array)
        {
            Write(ptr, str.size());

            std::memcpy(ptr, str.data(), str.size());
            ptr += str.size();
        }

        return true;
    }
};

template<glm::length_t L, typename T, glm::qualifier Q>
struct HvValueConvert<glm::vec<L, T, Q>>
{
    using VecType = glm::vec<L, T, Q>;

    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_glm_vec"sv);
    }

    static bool HvToType(const std::byte* data, std::size_t size, VecType& value)
    {
        if (size < sizeof(VecType))
            return false;

        std::memcpy(&value, data, size);
        return true;
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const VecType& obj)
    {
        size = sizeof(VecType);
        data = new std::byte[size];
        std::memcpy(data, &obj, size);

        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<VecType>& array)
    {
        array.resize(size / sizeof(VecType));
        std::memcpy(array.data(), data, size);

        return true;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<VecType> array)
    {
        size = array.size() * sizeof(VecType);
        data = new std::byte[size];

        std::memcpy(data, array.data(), size);

        return true;
    }

};

template<typename T>
concept EnumConcept = std::is_enum_v<T>;


template<EnumConcept EnumType>
struct HvValueConvert<EnumType>
{
    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_cpp_enum"sv);
    }

    static bool HvToType(const std::byte* data, std::size_t size, EnumType& value)
    {
        if (size < sizeof(EnumType))
            return false;
        std::memcpy(&value, data, sizeof(EnumType));
        return true;
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const EnumType& obj)
    {
        data = reinterpret_cast<std::byte*>(new EnumType(obj));
        size = sizeof(EnumType);
        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<EnumType>& array)
    {
        array.resize(size / sizeof(EnumType));
        std::memcpy(array.data(), data, size);
        return true;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const EnumType> array)
    {
        size = array.size() * sizeof(EnumType);
        data = new std::byte[size];
        std::memcpy(data, array.data(), size);
        return true;
    }
};

template<>
struct HvValueConvert<HvBool>
{
    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_hv_bool"sv);
    }

    static bool HvToType(const std::byte* data, std::size_t size, HvBool& value)
    {
        value.value = *reinterpret_cast<const std::uint8_t*>(data);
        return true;
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const HvBool& obj)
    {
        data = reinterpret_cast<std::byte*>(new std::uint8_t(obj.value));
        size = 1;
        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<HvBool>& array)
    {
        array.resize(size);
        std::memcpy(array.data(), data, size);
        return true;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const HvBool> array)
    {
        size = array.size();
        data = new std::byte[size];
        std::memcpy(data, array.data(), size);
        return true;
    }
};


template<class Clock, class Duration>
struct HvValueConvert<chr::time_point<Clock, Duration>>
{
    using TimePoint = chr::time_point<Clock, Duration>;
    using Underlying = decltype(std::declval<TimePoint>().time_since_epoch().count());

    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_cpp_chrono_time_point"sv);
    }

    static bool HvToType(const std::byte* data, std::size_t size, TimePoint& value)
    {
        auto underlying = *reinterpret_cast<const Underlying*>(data);

        value = TimePoint(Duration(underlying));
        return true;
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const TimePoint& obj)
    {
        auto underlying = obj.time_since_epoch().count();

        data = reinterpret_cast<std::byte*>(new decltype(underlying)(underlying));
        size = sizeof(underlying);
        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<TimePoint>& array)
    {
        return false;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const TimePoint> array)
    {
        return false;
    }
};

template<>
struct HvValueConvert<fs::path>
{
    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_stl_path"sv);
    }

    static bool HvToType(const std::byte* data, std::size_t size, fs::path& value)
    {
        value.assign(std::u32string_view(reinterpret_cast<const char32_t*>(data), size / sizeof(char32_t)));
        return true;
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const fs::path& obj)
    {
        auto u32str = obj.generic_u32string();

        size = u32str.size() * sizeof(char32_t);
        data = new std::byte[size];

        std::memcpy(data, u32str.data(), size);

        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<fs::path>& array)
    {
        std::uint64_t count;
        Read(data, count);
        array.resize(count);

        for (std::size_t i = 0; i < count; i++)
        {
            std::size_t strSize;
            Read(data, strSize);

            std::u32string_view strView(reinterpret_cast<const char32_t*>(data), strSize);
            array[i].assign(strView);
            data += strSize * sizeof(char32_t);
        }
        return true;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const fs::path> array)
    {
        size = 8;
        for (auto& str : array)
        {
            size += 8;
            size += str.generic_u32string().size() * sizeof(char32_t);
        }

        data = new std::byte[size];
        auto ptr = data;

        Write(ptr, array.size());

        for (auto& str : array)
        {
            auto u32str = str.generic_u32string();
            Write(ptr, u32str.size());

            std::memcpy(ptr, u32str.data(), u32str.size() * sizeof(char32_t));
            ptr += u32str.size() * sizeof(char32_t);
        }

        return true;
    }
};

template<>
struct HvValueConvert<HvTreePath>
{
    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_hv_tree_path"sv);
    }

    static bool HvToType(const std::byte* data, std::size_t size, HvTreePath& value)
    {
        std::string str(reinterpret_cast<const char*>(data), size);

        value.Parse(str);
        return true;
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const HvTreePath& obj)
    {
        auto str = obj.ToString();

        size = str.size();
        data = new std::byte[size];

        std::memcpy(data, str.data(), str.size());

        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<HvTreePath>& array)
    {
        std::uint64_t count;
        Read(data, count);
        array.resize(size);
        for (std::size_t i = 0; i < count; i++)
        {
            std::size_t strSize;
            Read(data, strSize);

            array[i].Parse(std::string(reinterpret_cast<const char*>(data), strSize));
            data += strSize;
        }
        return true;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const HvTreePath> array)
    {
        size = 8;
        for (auto& path : array)
        {
            auto str = path.ToString();

            size += 8;
            size += str.size();
        }

        data = new std::byte[size];
        auto ptr = data;

        Write(ptr, array.size());

        for (auto& path : array)
        {
            auto str = path.ToString();

            Write(ptr, str.size());

            std::memcpy(ptr, str.data(), str.size());
            ptr += str.size();
        }

        return true;
    }
};