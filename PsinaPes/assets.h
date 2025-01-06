#pragma once

#include <map>
#include <set>
#include <algorithm>
#include <string>
#include <filesystem>
#include <tuple>

#include "binary_io_utility.h"
#include "serialization.h"
#include "VulkanContext.h"  

namespace fs = std::filesystem;

struct AssetDescriptor
{
    fs::path file;
    HvTreePath treePath;

    friend std::strong_ordering operator<=>(const AssetDescriptor& f, const AssetDescriptor& s) = default;
};

struct AssetReferenceCounter
{
    std::size_t counter = 0;
};

struct BinaryDataDescriptor
{
    virtual std::size_t Size() const = 0;
    virtual const std::byte* Data() const = 0;
};

struct BinaryFileData : BinaryDataDescriptor
{
    std::byte* data;
    std::size_t size;
    fs::path source;

    BinaryFileData(fs::path file) : source(file)
    {
        size = fs::file_size(file);
        data = new std::byte[size];

        std::ifstream in(file, std::ios::binary);
        in.read(reinterpret_cast<char*>(data), size);
    }
    ~BinaryFileData()
    {
        delete[] data;
    }

    virtual std::size_t Size() const override
    {
        return size;
    }
    virtual const std::byte* Data() const override
    {
        return data;
    }
};

template<typename AssetType>
void Destroy(void* asset)
{
    delete reinterpret_cast<AssetType*>(asset);
}
using AssetDestroyFunction = void(*)(void*);

class AssetManager : MoveConstructOnly, VulkanResource
{
    template<typename T>
    friend class AssetReference;

public:

    enum class AssetInstancingPolicy
    {
        FORCE_NEW,
        RANDOM
    };

    AssetManager(VulkanContext& context) : VulkanResource(context)
    {
        instanceData["default"];
    }
    

    struct Asset
    {
        void* data;
        AssetReferenceCounter counter;
        AssetDestroyFunction destroyFunction;

        Asset(void* data, AssetDestroyFunction func) : data(data), destroyFunction(func)
        {

        }
        ~Asset()
        {
            destroyFunction(data);
        }
    };

    std::map<fs::path, HvTree> fileCache;

    using AssetListType = std::map<HvTreePath, Asset>;
    std::map<AssetDescriptor, AssetListType> assets;

    HvTree instanceData;

    HvTree& FileCache(fs::path file)
    {
        auto cachedFile = fileCache.find(file);
        if (cachedFile != fileCache.end())
        {
            return cachedFile->second;
        }
        else
        {
            BinaryFileData data(file);
            HvTree newTree;
            newTree.LoadBinary(data.data, data.size);
            auto newFile = fileCache.emplace(file, std::move(newTree));
            return newFile.first->second;
        }
    }

    AssetListType& AssetList(AssetDescriptor desc)
    {
        auto assetList = assets.find(desc);
        if (assetList != assets.end())
        {
            return assetList->second;
        }
        else
        {
            return assets[desc];
        }
    }

    template<typename AssetType>
    Asset& Acquire(AssetDescriptor desc, HvTreePath instance = "default"_tp, AssetInstancingPolicy instancing = AssetInstancingPolicy::FORCE_NEW)
    {
        auto& list = AssetList(desc);
        
        auto exact = list.find(instance);
        if (exact != list.end())
            return exact->second;

        if (!list.empty())
        {
            auto any = list.begin();
            if (instancing == AssetInstancingPolicy::RANDOM)
                return any->second;

            auto &fileData = FileCache(desc.file);
            AssetType* newAsset = new AssetType(context, fileData[desc.treePath], instanceData[instance], reinterpret_cast<AssetType*>(any->second.data), instanceData[any->first]);
            auto storedNewAsset = list.emplace(std::piecewise_construct, std::forward_as_tuple(instance), std::forward_as_tuple(newAsset, Destroy<AssetType>));
            return storedNewAsset.first->second;
        }
        else
        {
            auto& fileData = FileCache(desc.file);
            AssetType* newAsset = new AssetType(context, fileData[desc.treePath], instanceData[instance], nullptr, instanceData["default"]);
            auto storedNewAsset = list.emplace(std::piecewise_construct, std::forward_as_tuple(instance), std::forward_as_tuple(newAsset, Destroy<AssetType>));
            return storedNewAsset.first->second;
        }
    }


    void MountInstanceData(HvTreePath dst, fs::path src)
    {
        BinaryFileData data(src);
        instanceData[dst].LoadBinary(data.data, data.size);
    }

    void MountInstanceData(HvTreePath dst, HvTree src)
    {
        instanceData[dst] = src;
    }


};
struct AssetInstanceDescriptor
{
    AssetDescriptor asset;
    HvTreePath instance = "default"_tp;
    AssetManager::AssetInstancingPolicy instancingPolicy = AssetManager::AssetInstancingPolicy::FORCE_NEW;
};

template<typename AssetType>
class AssetReference
{
    AssetManager& manager;
    AssetManager::Asset* asset;
    AssetDescriptor descriptor;
    HvTreePath instance;
    AssetManager::AssetInstancingPolicy instancingPolicy;

public:

    AssetReference(AssetManager& manager, AssetDescriptor descriptor, HvTreePath instance = "default"_tp, AssetManager::AssetInstancingPolicy instancingPolicy = AssetManager::AssetInstancingPolicy::FORCE_NEW, bool autoAcquire = false) : manager(manager), descriptor(descriptor), asset(nullptr), instance(instance), instancingPolicy(instancingPolicy)
    {
        if (autoAcquire)
            Acquire();
    }
    AssetReference(AssetManager& manager, AssetInstanceDescriptor descriptor, bool autoAcquire = false) : AssetReference(manager, descriptor.asset, descriptor.instance, descriptor.instancingPolicy, autoAcquire)
    {

    }

    void Acquire()
    {
        if (asset) return;
        auto& newAsset = manager.Acquire<AssetType>(descriptor, instance, instancingPolicy);
        newAsset.counter.counter++;
        asset = &newAsset;
    }
    void Release()
    {
        if (!asset) return;
        asset->counter.counter--;
        asset = nullptr;
    }
    bool Test() const
    {
        return asset;
    }

    AssetType& Get()
    {
        if (!Test()) Acquire();
        return *reinterpret_cast<AssetType*>(asset->data);
    }
    operator AssetType& ()
    {
        return Get();
    }
    AssetType& operator*()
    {
        return Get();
    }
    AssetType* operator->()
    {
        if (!Test()) Acquire();
        return reinterpret_cast<AssetType*>(asset->data);
    }

    AssetDescriptor Descriptor() const
    {
        return descriptor;
    }
    AssetInstanceDescriptor FullDescriptor() const
    {
        return AssetInstanceDescriptor{ descriptor, instance, instancingPolicy };
    }

    ~AssetReference()
    {
        Release();
    }
};