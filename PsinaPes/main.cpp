#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <volk.h>

#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RIGHT_HANDED
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <cmath>
#include <span>
#include <numeric>
#include <filesystem>
#include <string>
#include <string_view>
#include <map>
#include <concepts>
#include <tuple>
#include <expected>
#include <mdspan>
#include <random>
#include <numbers>
#include <thread>
#include <print>
#include <functional>

#include "assets.h"
#include "serialization.h"
#include "ConstexprHash.h"
#include "VulkanContext.h"
#include "Window.h"
#include "GLFWContext.h"

#include <spirv.hpp>
#include <spirv_reflect.hpp>

#include <vk_mem_alloc.h>

#include <HvTree2/HvTree.h>
#include <HvTree2/Common/ConstexprHash.h>
#include <HvTree2/Conversion/Conversion.h>
#include <HvTree2/Handle/StringWrapper.h>
#include <HvTree2/Data/DedicatedAllocation.h>

#include <DependencyGraph/Graph.hpp>

using namespace std::literals;
using namespace hv::literals;

namespace fs = std::filesystem;

float speed = 6.f;
float rotationSpeed = 0.001f;
std::chrono::steady_clock::time_point lastTime;

float indirectMultiplier = 1.f;
float indirectMultiplierStep = 0.1f;
float indirectMultiplierStep2 = 1.f;

enum class AxisBinding
{
    X = 0,
    Y = 1,
    Z = 2,
    W = 3
};
enum class AxisInversion : std::uint8_t
{
    INVERT_NONE = 0x0,
    INVERT_X = 0x1,
    INVERT_Y = 0x2,
    INVERT_Z = 0x4,
    INVERT_W = 0x8
};
constexpr AxisInversion operator&(AxisInversion lhs, AxisInversion rhs)
{
    using Underlying = std::underlying_type_t<AxisInversion>;
    return static_cast<AxisInversion>(static_cast<Underlying>(lhs) & static_cast<Underlying>(rhs));
}

static constexpr glm::mat4 AxisSwizzle(AxisBinding xBinding, AxisBinding yBinding, AxisBinding zBinding, AxisBinding wBinding, AxisInversion inversion = AxisInversion::INVERT_NONE)
{
    glm::mat4 axisSwizzle = glm::mat4(0.f);

    axisSwizzle[static_cast<int>(xBinding)][0] = static_cast<bool>(inversion & AxisInversion::INVERT_X) ? -1.f : 1.f;
    axisSwizzle[static_cast<int>(yBinding)][1] = static_cast<bool>(inversion & AxisInversion::INVERT_Y) ? -1.f : 1.f;
    axisSwizzle[static_cast<int>(zBinding)][2] = static_cast<bool>(inversion & AxisInversion::INVERT_Z) ? -1.f : 1.f;
    axisSwizzle[static_cast<int>(wBinding)][3] = static_cast<bool>(inversion & AxisInversion::INVERT_W) ? -1.f : 1.f;

    return axisSwizzle;
}
static constexpr glm::mat3 AxisSwizzle3(AxisBinding xBinding, AxisBinding yBinding, AxisBinding zBinding, AxisInversion inversion = AxisInversion::INVERT_NONE)
{
    glm::mat3 axisSwizzle = glm::mat4(0.f);

    axisSwizzle[static_cast<int>(xBinding)][0] = static_cast<bool>(inversion & AxisInversion::INVERT_X) ? -1.f : 1.f;
    axisSwizzle[static_cast<int>(yBinding)][1] = static_cast<bool>(inversion & AxisInversion::INVERT_Y) ? -1.f : 1.f;
    axisSwizzle[static_cast<int>(zBinding)][2] = static_cast<bool>(inversion & AxisInversion::INVERT_Z) ? -1.f : 1.f;

    return axisSwizzle;
}

static constexpr glm::mat4 AssimpToHvost()
{
    using enum AxisBinding;
    return AxisSwizzle(Z, X, Y, W, AxisInversion::INVERT_X);
}
static constexpr glm::mat3 AssimpToHvost3()
{
    using enum AxisBinding;
    return AxisSwizzle3(Z, X, Y, AxisInversion::INVERT_X);
}

glm::mat4 EulerAnglesRotationMatrix(glm::vec3 angles)
{
    return glm::rotate(glm::mat4(1.f), angles.x, glm::vec3(-1.f, 0.f, 0.f)) *
        glm::rotate(glm::mat4(1.f), angles.z, glm::vec3(0.f, 0.f, -1.f)) *
        glm::rotate(glm::mat4(1.f), angles.y, glm::vec3(0.f, -1.f, 0.f));
}

std::ostream& operator<<(std::ostream& out, const glm::mat4& matrix)
{
    for (std::size_t i = 0; i < 4; i++)
    {
        std::cout << "[";
        for (std::size_t j = 0; j < 4; j++)
        {
            std::cout << std::setw(10) << matrix[j][i] << " ";
        }
        std::cout << "]\n";
    }
    return std::cout << std::endl;
}
std::ostream& operator<<(std::ostream& out, const glm::mat3& matrix)
{
    for (std::size_t i = 0; i < 3; i++)
    {
        std::cout << "[";
        for (std::size_t j = 0; j < 3; j++)
        {
            std::cout << std::setw(10) << matrix[j][i] << " ";
        }
        std::cout << "]\n";
    }
    return std::cout << std::endl;
}

std::ostream& operator<<(std::ostream& out, const glm::vec3& vec)
{
    return std::cout << "vec3(" << std::setprecision(2) << vec.x << ", " << std::setprecision(2) << vec.y << ", " << std::setprecision(2) << vec.z << ")";
}
std::ostream& operator<<(std::ostream& out, const glm::vec4& vec)
{
    return std::cout << "vec4(" << std::setprecision(2) << vec.x << ", " << std::setprecision(2) << vec.y << ", " << std::setprecision(2) << vec.z << ", " << std::setprecision(2) << vec.w << ")";
}

class Transformable
{
protected:
    glm::vec3 position;
    glm::vec3 rotation;
    glm::vec3 scale;

    mutable glm::mat4 modelMatrix;
    bool matrixNeedsUpdate = false;

    void UpdateMatrix() const
    {
        modelMatrix = glm::translate(glm::mat4(1.f), position) *
            EulerAnglesRotationMatrix(rotation) *
            glm::scale(glm::mat4(1.f), scale);


    }

public:

    Transformable() : position(0.f), rotation(0.f), scale(1.f), modelMatrix(1.f) {}

    void SetPosition(glm::vec3 newPosition)
    {
        position = newPosition;
        matrixNeedsUpdate = true;
    }
    void SetRotation(glm::vec3 newRotation)
    {
        rotation = newRotation;
        matrixNeedsUpdate = true;
    }
    void SetScale(glm::vec3 newScale)
    {
        scale = newScale;
        matrixNeedsUpdate = true;
    }

    const glm::vec3& GetPosition() const
    {
        return position;
    }
    const glm::vec3& GetRotation() const
    {
        return rotation;
    }
    const glm::vec3& GetScale() const
    {
        return scale;
    }

    const glm::mat4& GetModelMatrix() const
    {
        if (matrixNeedsUpdate)
            UpdateMatrix();
        return modelMatrix;
    }
};


template<typename T>
VkExtent3D Extent(glm::vec<3, T> vec)
{
    return VkExtent3D(vec.x, vec.y, vec.z);
}
template<typename T>
VkOffset3D Offset(glm::vec<3, T> vec)
{
    return VkOffset3D(vec.x, vec.y, vec.z);
}

glm::uvec2 Extent(VkExtent2D extent)
{
    return glm::uvec2(extent.width, extent.height);
}
glm::uvec3 Extent(VkExtent3D extent)
{
    return glm::uvec3(extent.width, extent.height, extent.depth);
}
glm::uvec2 Offset(VkOffset2D offset)
{
    return glm::uvec2(offset.x, offset.y);
}
glm::uvec3 Offset(VkOffset3D offset)
{
    return glm::uvec3(offset.x, offset.y, offset.z);
}

template<typename T>
VkExtent2D Extent(glm::vec<2, T> vec)
{
    return VkExtent2D(vec.x, vec.y);
}
template<typename T>
VkOffset2D Offset(glm::vec<2, T> vec)
{
    return VkOffset2D(vec.x, vec.y);
}

template<typename T = std::uint32_t>
struct Rect
{
    glm::vec<2, T> offset;
    glm::vec<2, T> extent;

    operator VkRect2D()
    {
        VkRect2D result{};
        result.offset.x = offset.x;
        result.offset.y = offset.y;
        result.extent.width = extent.x;
        result.extent.height = extent.y;
        return result;
    }
};

struct Camera : public Transformable
{
    glm::mat4 view;
    glm::mat4 projection;

    glm::mat4 offsetMatrix;

    float fov;

    bool perspective = true;

    Camera() : projection(glm::mat4(1.f))
    {
        UpdateView();
        UpdateOffsetMatrix();
    }

    void UpdateView()
    {
        auto ndcPosition = glm::vec3(AxisSwizzle() * glm::vec4(position, 1.f));
        auto ndcRotation = AxisSwizzle() * glm::vec4(rotation, 0.f);

        view = glm::translate(glm::mat4(1.f), ndcPosition) *
            glm::rotate(glm::mat4(1.f), ndcRotation.y, glm::vec3(0.f, 1.f, 0.f)) *
            glm::rotate(glm::mat4(1.f), ndcRotation.z, glm::vec3(0.f, 0.f, 1.f)) *
            glm::rotate(glm::mat4(1.f), ndcRotation.x, glm::vec3(1.f, 0.f, 0.f));

        view = glm::inverse(view);


    }

    void UpdateProjection(glm::uvec2 swapChainExtent)
    {
        if (perspective)
            projection = glm::perspectiveLH_ZO(glm::radians(fov), swapChainExtent.x / (float)swapChainExtent.y, 0.0001f, std::numeric_limits<float>::max());
        else
            projection = glm::orthoLH_ZO(-2.f, 2.f, -2.f, 2.f, 0.1f, std::numeric_limits<float>::max());
    }

    void UpdateOffsetMatrix()
    {
        /*offsetMatrix = glm::mat4(1.f);

        offsetMatrix = glm::rotate(offsetMatrix, rotation.y, glm::vec3(0.f, 1.f, 0.f));
        offsetMatrix = glm::rotate(offsetMatrix, rotation.z, glm::vec3(0.f, 0.f, 1.f));
        offsetMatrix = glm::rotate(offsetMatrix, rotation.x, glm::vec3(1.f, 0.f, 0.f));

        offsetMatrix = glm::inverse(offsetMatrix);*/

        offsetMatrix = EulerAnglesRotationMatrix(rotation);
    }

    void AddOffset(glm::vec4 offset, bool applyOffset = true)
    {
        if (applyOffset) position += glm::vec3(offsetMatrix * offset);
        else position += glm::vec3(offset);

        //std::cout << glm::to_string(position) << std::endl;
        //std::cout << glm::to_string(glm::vec3(position)) << std::endl;

        UpdateView();
    }

    void AddRotation(glm::vec3 rotation)
    {
        this->rotation += rotation;

        this->rotation.y = std::clamp(this->rotation.y, -glm::radians(89.f), glm::radians(89.f));

        UpdateView();
        UpdateOffsetMatrix();
    }

    static constexpr glm::mat4 AxisSwizzle()
    {
        return ::AxisSwizzle(AxisBinding::Y, AxisBinding::Z, AxisBinding::X, AxisBinding::W, AxisInversion::INVERT_Y);
    }
    static constexpr glm::mat3 AxisSwizzle3()
    {
        return ::AxisSwizzle3(AxisBinding::Y, AxisBinding::Z, AxisBinding::X, AxisInversion::INVERT_Y);
    }

    void hdebug()
    {
        system("cls");

        std::cout << "Axis-Swizzle:" << std::endl;
        std::cout << AxisSwizzle();

        std::cout << "View:" << std::endl;
        std::cout << view;

        std::cout << "Projection:" << std::endl;
        std::cout << projection;

        std::cout << std::endl;

        glm::vec4 testVec(4.f, 0.f, 0.f, 1.f);
        std::cout << "Vector:                           " << testVec << std::endl;
        std::cout << "After Axis-Swizzle:               " << AxisSwizzle() * testVec << std::endl;

        std::cout << std::endl;

        std::cout << "After View:                       " << view * AxisSwizzle() * testVec << std::endl;
        std::cout << "After Projection:                 " << projection * view * AxisSwizzle() * testVec << std::endl;
        auto resultVec = projection * view * AxisSwizzle() * testVec;
        std::cout << "After Perspective-Divide:         " << resultVec / resultVec.w << std::endl;

        std::cout << std::endl;

        auto resultVecNoView = projection * AxisSwizzle() * testVec;
        std::cout << "After Projection(no view):        " << projection * AxisSwizzle() * testVec << std::endl;
        std::cout << "After Perspective-Divide(no view):" << resultVecNoView / resultVecNoView.w << std::endl;
    }
};

Camera camera; 

/*

std::vector<const char*> getRequiredExtensions()
{
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}
VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}
*/

VkShaderModule createShaderModule(VulkanContext& context, const void* code, std::size_t size)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = size;
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code);

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(context.device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}
VkShaderModule createShaderModule(VulkanContext& context, const std::vector<char>& code)
{
    return createShaderModule(context, code.data(), code.size());
}


struct alignas(VkSemaphore) Semaphore : VulkanResource, MoveConstructOnly
{
    VkSemaphore semaphore;

    Semaphore(VulkanContext& context) : VulkanResource(context)
    {
        VkSemaphoreCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        vkCreateSemaphore(context.device, &createInfo, nullptr, &semaphore);
    }

    Semaphore(Semaphore&& rhs) noexcept : VulkanResource(rhs.context), semaphore(rhs.semaphore)
    {
        rhs.semaphore = VK_NULL_HANDLE;
    }

    ~Semaphore()
    {
        if (semaphore != VK_NULL_HANDLE) vkDestroySemaphore(context.device, semaphore, nullptr);
    }

    operator VkSemaphore() const
    {
        return semaphore;
    }

    void Signal()
    {
        VkSemaphoreSignalInfo signalInfo{};
        signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
        signalInfo.semaphore = semaphore;

        vkSignalSemaphore(context.device, &signalInfo);
    }
};
struct alignas(VkFence) Fence : VulkanResource, MoveConstructOnly
{
    VkFence fence;

    Fence(VulkanContext& context, bool signaled = false) : VulkanResource(context)
    {
        VkFenceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        createInfo.flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0;

        vkCreateFence(context.device, &createInfo, nullptr, &fence);
    }

    Fence(Fence&& rhs) noexcept : VulkanResource(rhs.context), fence(rhs.fence)
    {
        rhs.fence = VK_NULL_HANDLE;
    }

    operator VkFence() const
    {
        return fence;
    }

    void Wait(std::uint64_t timeout = std::numeric_limits<std::uint64_t>::max())
    {
        auto h = vkWaitForFences(context.device, 1, &fence, VK_TRUE, timeout);
    }
    void Reset()
    {
        auto h = vkResetFences(context.device, 1, &fence);
    }
    VkResult Status()
    {
        return vkGetFenceStatus(context.device, fence);
    }

    ~Fence()
    {
        if (fence != VK_NULL_HANDLE) vkDestroyFence(context.device, fence, nullptr);
    }
};


struct CommandPool : public VulkanResource
{
    VkCommandPool pool;

    CommandPool(VulkanContext& context) : VulkanResource(context)
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        
        poolInfo.queueFamilyIndex = context.queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(context.device, &poolInfo, nullptr, &pool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics command pool!");
        }
    }

    CommandPool(const CommandPool&) = delete;
    CommandPool(CommandPool&&) = delete;

    ~CommandPool()
    {
        vkDestroyCommandPool(context.device, pool, nullptr);
    }
};

struct CommandBuffer : VulkanResource
{
    VkCommandPool commandPool;
    VkCommandBuffer buffer;

    CommandBuffer(const CommandPool& commandPool) : CommandBuffer(commandPool.context, commandPool.pool)
    {

    }

    CommandBuffer(VulkanContext& context, VkCommandPool commandPool) : commandPool(commandPool), VulkanResource(context)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(context.device, &allocInfo, &buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void Reset()
    {
        vkResetCommandBuffer(buffer, 0);
    }
    void Begin()
    {
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        
        vkBeginCommandBuffer(buffer, &beginInfo);
    }
    void End()
    {
        vkEndCommandBuffer(buffer);
    }

    operator VkCommandBuffer() const
    {
        return buffer;
    }

    ~CommandBuffer()
    {
        vkFreeCommandBuffers(context.device, commandPool, 1, &buffer);
    }
};
struct TempCommandBuffer : VulkanResource
{
    static inline CommandPool* tempCommandPool = nullptr;

    VkCommandBuffer buffer;
    Fence fence;

    TempCommandBuffer(VulkanContext& context) : buffer(VK_NULL_HANDLE), VulkanResource(context), fence(context)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = tempCommandPool->pool;
        allocInfo.commandBufferCount = 1;
    
        vkAllocateCommandBuffers(context.device, &allocInfo, &buffer);
    
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        
        vkBeginCommandBuffer(buffer, &beginInfo);
    }

    TempCommandBuffer(const TempCommandBuffer&) = delete;
    TempCommandBuffer(TempCommandBuffer&&) = delete;

    void Execute()
    {
        if (buffer == VK_NULL_HANDLE) return;

        vkEndCommandBuffer(buffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &buffer;

        vkQueueSubmit(context.graphicsQueue, 1, &submitInfo, fence);
        fence.Wait();

        vkFreeCommandBuffers(context.device, tempCommandPool->pool, 1, &buffer);

        buffer = VK_NULL_HANDLE;
    }
    void ExecuteAndReset()
    {
        if (buffer == VK_NULL_HANDLE) return;

        auto h = vkEndCommandBuffer(buffer);
        
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &buffer;
        
        auto hhh = vkQueueSubmit(context.graphicsQueue, 1, &submitInfo, fence);
        auto hh = fence.Status();
        fence.Wait(); 
        fence.Reset();

        auto hhhh = vkResetCommandBuffer(buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(buffer, &beginInfo);
    }

    operator VkCommandBuffer() const
    {
        return buffer;
    }

    ~TempCommandBuffer()
    {
        Execute();
    }
};

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoord;
    glm::vec3 tangent;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, normal);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[3].offset = offsetof(Vertex, tangent);

        return attributeDescriptions;
    }
};

struct Texture;

struct DescriptorSetWritable
{
    virtual VkWriteDescriptorSet GetWrite(std::size_t imageIndex) const = 0;
};

struct Texture
{
    glm::uvec2 size;
    std::vector<std::byte> textureData;
    bool useFloat;

    Texture() : size(0), useFloat(false) {}
    Texture(fs::path file, bool loadFloat = false)
    {
        LoadFromFile(file, loadFloat);
    }

    bool LoadFromFile(fs::path file, bool loadFloat = false)
    {
        useFloat = loadFloat;

        if (!fs::is_regular_file(file)) return false;

        int texWidth, texHeight, texChannels;
        void* pixels;
        if(loadFloat) pixels = stbi_loadf(file.generic_string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        else pixels = stbi_load(file.generic_string().c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * (loadFloat ? 16 : 4);
        
        size.x = texWidth;
        size.y = texHeight;

        if (!pixels)
        {
            return false;
        }

        textureData.resize(imageSize);
        std::memcpy(textureData.data(), pixels, imageSize);

        stbi_image_free(pixels);
    }

};

struct Buffer : VulkanResource
{
    VkDeviceSize size;
    VkBuffer buffer;

    VmaAllocation allocation;
    VmaAllocationInfo allocInfo;

    Buffer(VulkanContext& context, VkDeviceSize size, VkBufferUsageFlags usage, VmaAllocationCreateFlags allocationCreateFlags) : size(size), VulkanResource(context)
    {
        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = size;
        bufferInfo.usage = usage;

        VmaAllocationCreateInfo allocCreateInfo = {};
        allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocCreateInfo.flags = allocationCreateFlags;

        vmaCreateBuffer(context.allocator, &bufferInfo, &allocCreateInfo, &buffer, &allocation, &allocInfo);
    }

    void CopyTo(VkCommandBuffer commandBuffer, VkBuffer dstBuffer)
    {
        VkBufferCopy copyRegion{};
        copyRegion.size = size;

        vkCmdCopyBuffer(commandBuffer, buffer, dstBuffer, 1, &copyRegion);
    }

    ~Buffer()
    {
        vmaDestroyBuffer(context.allocator, buffer, allocation);
    }
};
struct UniformBuffer : VulkanResource
{
    std::array<Delayed<Buffer>, MAX_FRAMES_IN_FLIGHT> buffers;
    VkDeviceSize size;

    UniformBuffer(VulkanContext& context, VkDeviceSize size) : size(size), VulkanResource(context)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
            buffers[i].Create(context, size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    }

    Buffer& GetBuffer(std::size_t index)
    {
        return buffers[index];
    }
    const Buffer& GetBuffer(std::size_t index) const
    {
        return buffers[index];
    }

    template<typename T>
    void CopyFrom(T&& object, std::size_t index)
    {
        std::memcpy(GetBuffer(index).allocInfo.pMappedData, &object, size);
    }

    ~UniformBuffer()
    {
    
    }
};

struct StagingBuffer : VulkanResource
{
    Buffer buffer;

    StagingBuffer(VulkanContext& context, VkDeviceSize size) : VulkanResource(context), buffer(context, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT)
    {
        
    }
    StagingBuffer(VulkanContext& context, VkDeviceSize size, const void* data) : StagingBuffer(context, size)
    {
        std::memcpy(buffer.allocInfo.pMappedData, data, size);
    }
};
struct StagedBuffer : VulkanResource
{
    Buffer buffer;

    StagedBuffer(VulkanContext& context, VkDeviceSize size, const void* data, VkBufferUsageFlags flags = 0) : VulkanResource(context), buffer(context, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | flags, 0)
    {
        StagingBuffer stagingBuffer(context, size, data);
        
        TempCommandBuffer commandBuffer(context);
        stagingBuffer.buffer.CopyTo(commandBuffer.buffer, buffer.buffer);
    }

    ~StagedBuffer()
    {

    }
};


struct ShaderStageDescriptor
{
    VkShaderStageFlagBits stage;
    std::string_view mainFunction = "main";

    fs::path source;

    VkPipelineShaderStageCreateInfo MakeCreateInfo()
    {
        VkPipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = stage;
        shaderStageInfo.pName = mainFunction.data();

        return shaderStageInfo;
    }
};
struct ShaderCodeLoader
{
    std::byte* code;
    std::size_t size;

    ShaderCodeLoader(fs::path file)
    {
        std::ifstream in(file, std::ios::binary | std::ios::in);

        size = fs::file_size(file);

        code = new std::byte[size];

        in.read(reinterpret_cast<char*>(code), size);
    }

    ShaderCodeLoader(ShaderStageDescriptor desc) : ShaderCodeLoader(desc.source)
    {

    }

    ~ShaderCodeLoader()
    {
        delete[] code;
    }
};
struct ShaderModule : MoveConstructOnly, VulkanResource
{
    VkShaderModule shaderModule;
    
    ShaderModule(VulkanContext& context, std::byte* code, std::size_t size) : VulkanResource(context)
    {
        shaderModule = createShaderModule(context, code, size);
    }
    ShaderModule(VulkanContext& context, const ShaderCodeLoader& loader) : VulkanResource(context)
    {
        shaderModule = createShaderModule(context, loader.code, loader.size);
    }
    ~ShaderModule()
    {
        vkDestroyShaderModule(context.device, shaderModule, nullptr);
    }
};

struct HDescriptorSetLayout
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    VkDescriptorSetLayout layout;
};
struct Material : VulkanResource, NonCopyable
{
    std::map<std::uint32_t, HDescriptorSetLayout> setLayouts;

    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;

public:

    struct PipelineCreateInfo
    {
        VkPipelineInputAssemblyStateCreateInfo inputAssembly =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE
        };
        VkPipelineViewportStateCreateInfo viewportState =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr
        };
        VkPipelineRasterizationStateCreateInfo rasterization =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,

        };
        VkPipelineMultisampleStateCreateInfo multisampling =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };
        VkPipelineColorBlendAttachmentState colorBlendAttachment =
        {
            .blendEnable = VK_FALSE,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
        };
        VkPipelineColorBlendStateCreateInfo colorBlending =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment,
            .blendConstants = {0.f, 0.f, 0.f, 0.f}
        };
        std::vector<VkDynamicState> dynamicStates =
        {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDepthStencilStateCreateInfo depthStencil =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE
        };
    };

    Material(VulkanContext &context, const HvTree& tree, const HvTree& instance, Material* base, const HvTree& baseInstance) : VulkanResource(context)
    {
        std::size_t offset = 0;
        std::vector<VkPushConstantRange> pushConstants;


        std::size_t descCount = tree["stages/count"_tp].Acquire<std::size_t>().value();
        for (std::size_t i = 0; i < descCount; i++)
        {
            auto& stageData = tree["stages"][std::to_string(i)];
            auto code = stageData["code"].Acquire<HvRawBinaryView>().value();
            spirv_cross::Compiler comp(reinterpret_cast<const std::uint32_t*>(code.data), code.size / 4);

            VkShaderStageFlagBits stageFlags = stageData["stageType"].Acquire<VkShaderStageFlagBits>().value();



            auto sr = comp.get_shader_resources();
            for (auto desc : sr.acceleration_structures)
            {
                auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
                auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

                VkDescriptorSetLayoutBinding layoutBinding;
                layoutBinding.binding = binding;
                layoutBinding.descriptorCount = 1;
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                layoutBinding.pImmutableSamplers = nullptr;
                layoutBinding.stageFlags = stageFlags;

                auto existing = std::find_if(setLayouts[set].bindings.begin(), setLayouts[set].bindings.end(), [=](VkDescriptorSetLayoutBinding bind) -> bool { return bind.binding == binding && bind.descriptorType == bind.descriptorType; });
                if (existing != setLayouts[set].bindings.end())
                    existing->stageFlags |= stageFlags;
                else
                    setLayouts[set].bindings.push_back(layoutBinding);
            }
            for (auto desc : sr.uniform_buffers)
            {
                auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
                auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

                VkDescriptorSetLayoutBinding layoutBinding;
                layoutBinding.binding = binding;
                layoutBinding.descriptorCount = 1;
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                layoutBinding.pImmutableSamplers = nullptr;
                layoutBinding.stageFlags = stageFlags;


                auto existing = std::find_if(setLayouts[set].bindings.begin(), setLayouts[set].bindings.end(), [=](VkDescriptorSetLayoutBinding bind) -> bool { return bind.binding == binding && bind.descriptorType == bind.descriptorType; });
                if (existing != setLayouts[set].bindings.end())
                    existing->stageFlags |= stageFlags;
                else
                    setLayouts[set].bindings.push_back(layoutBinding);
            }
            for (auto desc : sr.sampled_images)
            {
                auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
                auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

                VkDescriptorSetLayoutBinding layoutBinding;
                layoutBinding.binding = binding;
                layoutBinding.descriptorCount = 1;
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                layoutBinding.pImmutableSamplers = nullptr;
                layoutBinding.stageFlags = stageFlags;


                auto existing = std::find_if(setLayouts[set].bindings.begin(), setLayouts[set].bindings.end(), [=](VkDescriptorSetLayoutBinding bind) -> bool { return bind.binding == binding && bind.descriptorType == bind.descriptorType; });
                if (existing != setLayouts[set].bindings.end())
                    existing->stageFlags |= stageFlags;
                else
                    setLayouts[set].bindings.push_back(layoutBinding);
            }
            for (auto desc : sr.separate_images)
            {
                auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
                auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

                VkDescriptorSetLayoutBinding layoutBinding;
                layoutBinding.binding = binding;
                layoutBinding.descriptorCount = 1;
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
                layoutBinding.pImmutableSamplers = nullptr;
                layoutBinding.stageFlags = stageFlags;


                auto existing = std::find_if(setLayouts[set].bindings.begin(), setLayouts[set].bindings.end(), [=](VkDescriptorSetLayoutBinding bind) -> bool { return bind.binding == binding && bind.descriptorType == bind.descriptorType; });
                if (existing != setLayouts[set].bindings.end())
                    existing->stageFlags |= stageFlags;
                else
                    setLayouts[set].bindings.push_back(layoutBinding);
            }
            for (auto desc : sr.storage_images)
            {
                auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
                auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

                VkDescriptorSetLayoutBinding layoutBinding;
                layoutBinding.binding = binding;
                layoutBinding.descriptorCount = 1;
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                layoutBinding.pImmutableSamplers = nullptr;
                layoutBinding.stageFlags = stageFlags;


                auto existing = std::find_if(setLayouts[set].bindings.begin(), setLayouts[set].bindings.end(), [=](VkDescriptorSetLayoutBinding bind) -> bool { return bind.binding == binding && bind.descriptorType == bind.descriptorType; });
                if (existing != setLayouts[set].bindings.end())
                    existing->stageFlags |= stageFlags;
                else
                    setLayouts[set].bindings.push_back(layoutBinding);
            }
            for (auto desc : sr.separate_samplers)
            {
                auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
                auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

                VkDescriptorSetLayoutBinding layoutBinding;
                layoutBinding.binding = binding;
                layoutBinding.descriptorCount = 1;
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
                layoutBinding.pImmutableSamplers = nullptr;
                layoutBinding.stageFlags = stageFlags;


                auto existing = std::find_if(setLayouts[set].bindings.begin(), setLayouts[set].bindings.end(), [=](VkDescriptorSetLayoutBinding bind) -> bool { return bind.binding == binding && bind.descriptorType == bind.descriptorType; });
                if (existing != setLayouts[set].bindings.end())
                    existing->stageFlags |= stageFlags;
                else
                    setLayouts[set].bindings.push_back(layoutBinding);
            }
            for (auto desc : sr.storage_buffers)
            {
                auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
                auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

                VkDescriptorSetLayoutBinding layoutBinding;
                layoutBinding.binding = binding;
                layoutBinding.descriptorCount = 1;
                layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                layoutBinding.pImmutableSamplers = nullptr;
                layoutBinding.stageFlags = stageFlags;


                auto existing = std::find_if(setLayouts[set].bindings.begin(), setLayouts[set].bindings.end(), [=](VkDescriptorSetLayoutBinding bind) -> bool { return bind.binding == binding && bind.descriptorType == bind.descriptorType; });
                if (existing != setLayouts[set].bindings.end())
                    existing->stageFlags |= stageFlags;
                else
                    setLayouts[set].bindings.push_back(layoutBinding);
            }

            for (auto pushConstant : sr.push_constant_buffers)
            {
                auto type = comp.get_type(pushConstant.type_id);
                std::size_t size = comp.get_declared_struct_size(type);

                auto& newRange = pushConstants.emplace_back();
                newRange.offset = offset;
                newRange.size = size;
                newRange.stageFlags = stageFlags;

                offset += size;
            }
        }

        for (auto& layout : setLayouts)
        {
            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = static_cast<uint32_t>(layout.second.bindings.size());
            layoutInfo.pBindings = layout.second.bindings.data();

            if (vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &layout.second.layout) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create descriptor set layout!");
            }
        }

        std::vector<VkDescriptorSetLayout> vkSetLayouts;
        for (auto& layout : setLayouts)
        {
            vkSetLayouts.push_back(layout.second.layout);
        }

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = vkSetLayouts.size();
        pipelineLayoutInfo.pSetLayouts = vkSetLayouts.data();

        pipelineLayoutInfo.pPushConstantRanges = pushConstants.data();
        pipelineLayoutInfo.pushConstantRangeCount = pushConstants.size();

        if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        auto attributes = tree["attributes"].AcquireArray<VkVertexInputAttributeDescription>().value();
        auto bindings = tree["bindings"].AcquireArray<VkVertexInputBindingDescription>().value();

        auto formats = instance["formats"].AcquireArray<VkFormat>().value();
        auto depthFormat = instance["depthFormat"].Acquire<VkFormat>().value();
        auto renderTargetCount = formats.size();

        vertexInputInfo.vertexBindingDescriptionCount = bindings.size();
        vertexInputInfo.vertexAttributeDescriptionCount = attributes.size();
        vertexInputInfo.pVertexBindingDescriptions = bindings.data();
        vertexInputInfo.pVertexAttributeDescriptions = attributes.data();

        auto createInfo = tree["pipelineCreateInfo"].Acquire<PipelineCreateInfo>().value();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = createInfo.inputAssembly;

        VkPipelineViewportStateCreateInfo viewportState = createInfo.viewportState;

        VkPipelineRasterizationStateCreateInfo rasterizer = createInfo.rasterization;

        VkPipelineMultisampleStateCreateInfo multisampling = createInfo.multisampling;

        VkPipelineColorBlendAttachmentState colorBlendAttachment = createInfo.colorBlendAttachment;

        VkPipelineColorBlendStateCreateInfo colorBlending = createInfo.colorBlending;

        VkPipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.blendEnable = VK_FALSE;
        blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        std::vector blendAttachments(std::from_range, std::views::repeat(blendAttachment, renderTargetCount));

        colorBlending.pAttachments = blendAttachments.data();
        colorBlending.attachmentCount = renderTargetCount;

        std::vector<VkDynamicState> dynamicStates = createInfo.dynamicStates;

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineDepthStencilStateCreateInfo depthStencil = createInfo.depthStencil;

        std::vector<std::string> temp(descCount);
        std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos(descCount);
        for (std::size_t i = 0; i < shaderStageCreateInfos.size(); i++)
        {
            VkPipelineShaderStageCreateInfo createInfo;
            createInfo.stage = tree["stages"][std::to_string(i)]["stageType"].Acquire<VkShaderStageFlagBits>().value();
            temp[i] = tree["stages"][std::to_string(i)]["mainFunction"].Acquire<std::string>().value();
            auto& f = temp[i];
            createInfo.pName = f.c_str();
            createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            createInfo.pNext = nullptr;
            createInfo.pSpecializationInfo = nullptr;
            createInfo.flags = 0;

            shaderStageCreateInfos[i] = createInfo;

            auto code = tree["stages"][std::to_string(i)]["code"].Acquire<HvRawBinaryView>().value();

            shaderStageCreateInfos[i].module = createShaderModule(context, code.data, code.size);
        }

        //std::uint32_t colorAttachmentCount = instance["colorAttachmentCount"].Acquire<std::uint32_t>().value();

        VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo{};
        pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        pipelineRenderingCreateInfo.colorAttachmentCount = renderTargetCount;
        pipelineRenderingCreateInfo.pColorAttachmentFormats = formats.data();
        pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat;
        pipelineRenderingCreateInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = shaderStageCreateInfos.size();
        pipelineInfo.pStages = shaderStageCreateInfos.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout;
        //pipelineInfo.renderPass = renderPass;
        //pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = base ? base->pipeline : VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &depthStencil;

        pipelineInfo.pNext = &pipelineRenderingCreateInfo;

        if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    void Bind(VkCommandBuffer commandBuffer)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    }

    ~Material()
    {
        vkDestroyPipeline(context.device, pipeline, nullptr);
        vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    }
};

template<>
struct HvValueConvert<Material::PipelineCreateInfo>
{
    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_hv_pipeline_create_info"sv);
    }

    static bool HvToType(const std::byte* data, std::size_t size, Material::PipelineCreateInfo& value)
    {
        auto ptr = data;
        Read(ptr, value.inputAssembly);
        Read(ptr, value.viewportState);
        Read(ptr, value.rasterization);
        Read(ptr, value.multisampling);
        Read(ptr, value.colorBlendAttachment);
        Read(ptr, value.colorBlending);
        Read(ptr, value.depthStencil);

        std::size_t stateCount;
        Read(ptr, stateCount);

        value.dynamicStates.resize(stateCount);
        for (std::size_t i = 0; i < stateCount; i++)
            Read(ptr, value.dynamicStates[i]);

        value.colorBlending.pAttachments = &value.colorBlendAttachment;

        return true;
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const Material::PipelineCreateInfo& obj)
    {
        size = sizeof(obj.colorBlendAttachment) + sizeof(obj.colorBlending) + sizeof(obj.depthStencil) + sizeof(obj.inputAssembly) + sizeof(obj.multisampling) + sizeof(obj.rasterization) + sizeof(obj.viewportState) + 8 + obj.dynamicStates.size() * 4;
        data = new std::byte[size];
        auto ptr = data;
        Write(ptr, obj.inputAssembly);
        Write(ptr, obj.viewportState);
        Write(ptr, obj.rasterization);
        Write(ptr, obj.multisampling);
        Write(ptr, obj.colorBlendAttachment);
        Write(ptr, obj.colorBlending);
        Write(ptr, obj.depthStencil);
        Write(ptr, obj.dynamicStates.size());
        for (auto state : obj.dynamicStates)
            Write(ptr, state);
        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<std::string>& array)
    {
        return false;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const std::string> array)
    {
        return false;
    }
};


struct HMesh : MoveConstructOnly, VulkanResource
{
    std::vector<Vertex> vertices;
    std::vector<std::uint32_t> indices;

    Delayed<StagedBuffer> vertexBuffer;
    Delayed<StagedBuffer> indexBuffer;

    HMesh(VulkanContext& context, const HvTree& tree, const HvTree& instance, HMesh* base, const HvTree& baseInstance) : VulkanResource(context)
    {
        auto vertexCount = tree["vertexCount"].Acquire<std::uint64_t>().value();
        auto indexCount = tree["indexCount"].Acquire<std::uint64_t>().value();

        vertices.resize(vertexCount);
        indices.resize(indexCount);

        auto vertexData = tree["vertices"].Acquire<HvRawBinaryView>().value();
        std::memcpy(vertices.data(), vertexData.data, vertexData.size);

        auto indexData = tree["indices"].Acquire<HvRawBinaryView>().value();
        std::memcpy(indices.data(), indexData.data, indexData.size);

        CreateVertexBuffer();
        CreateIndexBuffer();
    }

    void CreateVertexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        vertexBuffer.Create(context, bufferSize, vertices.data(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    }
    void CreateIndexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        indexBuffer.Create(context, bufferSize, indices.data(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    }
};

struct PipelineBarrierData
{
    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;
    VkAccessFlags srcAccessMask;
    VkAccessFlags dstAccessMask;
};
PipelineBarrierData DeducePipelineBarrierData(VkImageLayout srcLayout, VkImageLayout dstLayout)
{
        PipelineBarrierData result{};

        result.sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        result.destinationStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
     
        if (dstLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) result.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        if (dstLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) result.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        if (dstLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) result.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        if (dstLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) result.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        if (dstLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) result.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        if (srcLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) result.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        if (srcLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) result.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        if (srcLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) result.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        if (srcLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) result.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        if (srcLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) result.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;

        if (dstLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) result.destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (dstLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) result.destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (dstLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) result.destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        if (dstLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) result.destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        if (dstLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) result.destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

        if (srcLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) result.sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (srcLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) result.sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        if (srcLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) result.sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        if (srcLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) result.sourceStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        if (srcLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) result.sourceStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

        return result;
}

struct ImageSubresource
{
    VkImage image;
    VkFormat format;

    VkImageLayout layout;

    glm::uvec3 offset;
    glm::uvec3 extent;

    std::uint32_t baseMipLevel;
    std::uint32_t mipLevels;

    std::uint32_t baseArrayLayer;
    std::uint32_t arrayLayerCount;

    VkImageAspectFlags aspects;


    operator VkImageSubresource() const
    {
        VkImageSubresource result{};

        result.arrayLayer = baseArrayLayer;
        result.aspectMask = aspects;
        result.mipLevel = baseMipLevel;

        return result;
    }
    operator VkImageSubresourceLayers() const
    {
        VkImageSubresourceLayers result{};

        result.baseArrayLayer = baseArrayLayer;
        result.layerCount = arrayLayerCount;
        result.aspectMask = aspects;
        result.mipLevel = baseMipLevel;

        return result;
    }
    operator VkImageSubresourceRange() const
    {
        VkImageSubresourceRange result{};

        result.baseArrayLayer = baseArrayLayer;
        result.layerCount = arrayLayerCount;
        result.aspectMask = aspects;
        result.baseMipLevel = baseMipLevel;
        result.levelCount = mipLevels;

        return result;
    }

    void CopyFrom(VkCommandBuffer commandBuffer, const Buffer& src)
    {
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource = *this;
        region.imageOffset = Offset(offset);
        region.imageExtent = Extent(extent);

        vkCmdCopyBufferToImage(commandBuffer, src.buffer, image, layout, 1, &region);
    }

    void Blit(VkCommandBuffer commandBuffer, const ImageSubresource &dstSubresource, VkFilter filter = VK_FILTER_LINEAR)
    {
        VkImageBlit blit{};

        blit.srcSubresource = *this;
        blit.srcOffsets[0] = Offset(offset);
        blit.srcOffsets[1] = Offset(offset + extent);
        blit.dstSubresource = dstSubresource;
        blit.dstOffsets[0] = Offset(dstSubresource.offset);
        blit.dstOffsets[1] = Offset(dstSubresource.offset + dstSubresource.extent);

        vkCmdBlitImage(commandBuffer, image, layout, dstSubresource.image, dstSubresource.layout, 1, &blit, filter);
    }

    void TransitionLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout, PipelineBarrierData barrierData)
    {
        VkImageMemoryBarrier barrier{};

        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = layout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = *this;

        barrier.srcAccessMask = barrierData.srcAccessMask;
        barrier.dstAccessMask = barrierData.dstAccessMask;

        vkCmdPipelineBarrier(
            commandBuffer,
            barrierData.sourceStage, barrierData.destinationStage,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        layout = newLayout;
    }
    void TransitionLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout)
    {
        auto bd = DeducePipelineBarrierData(layout, newLayout);
        TransitionLayout(commandBuffer, newLayout, bd);
    }
};

struct Image : VulkanResource, MoveConstructOnly
{
    VkImage image;

    VmaAllocation allocation;

    glm::uvec3 size;
    VkImageLayout layout;
    VkFormat format;
    VkImageType imageType;
    std::uint32_t arrayLayers;
    std::uint32_t mipLevels;

    void Destroy()
    {
        if (image != VK_NULL_HANDLE) vmaDestroyImage(context.allocator, image, allocation);
        image = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;
    }

public:

    struct CreateInfo
    {
        glm::uvec3 size;
        VkImageUsageFlags imageUsage = 0;
        VkImageCreateFlags createFlags = 0;
        VkImageType imageType = VK_IMAGE_TYPE_2D;
        VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;
        std::uint32_t arrayLayers = 1;
        std::uint32_t mipLevels = MipLevelsDeduceFromSize;

        static constexpr std::uint32_t MipLevelsDeduceFromSize = 0;
    };

    Image(VulkanContext& context, CreateInfo createInfo) : 
        VulkanResource(context),
        size(createInfo.size),
        format(createInfo.format),
        imageType(createInfo.imageType),
        arrayLayers(createInfo.arrayLayers),
        mipLevels(createInfo.mipLevels == CreateInfo::MipLevelsDeduceFromSize ? DeduceMipLevels(createInfo.size) : createInfo.mipLevels),
        layout(VK_IMAGE_LAYOUT_UNDEFINED)
    {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = imageType;
        imageInfo.extent.width = size.x;
        imageInfo.extent.height = size.y;
        imageInfo.extent.depth = size.z;
        imageInfo.mipLevels = mipLevels;
        imageInfo.arrayLayers = arrayLayers;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = layout;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | createInfo.imageUsage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.flags = createInfo.createFlags;
        
        VmaAllocationCreateInfo allocCreateInfo{};
        allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;

        vmaCreateImage(context.allocator, &imageInfo, &allocCreateInfo, &image, &allocation, nullptr);
    }
    
    Image(Image&& rhs) noexcept : VulkanResource(rhs.context)
    {
        image = rhs.image;
        allocation = rhs.allocation;

        size = rhs.size;
        layout = rhs.layout;
        format = rhs.format;
        imageType = rhs.imageType;
        arrayLayers = rhs.arrayLayers;
        mipLevels = rhs.mipLevels;

        rhs.image = VK_NULL_HANDLE;
        rhs.allocation = VK_NULL_HANDLE;
    }

    static VkImageAspectFlags IdentifyAspects(VkFormat format)
    {
        static std::set depthFormats = { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D16_UNORM };
        static std::set stencilFormats = { VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D16_UNORM_S8_UINT };
        
        if (depthFormats.contains(format))
            if (stencilFormats.contains(format))
                return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            else
                return VK_IMAGE_ASPECT_DEPTH_BIT;
        else
            return VK_IMAGE_ASPECT_COLOR_BIT;
    }

    static std::uint32_t DeduceMipLevels(glm::uvec3 size)
    {
        return std::floor(std::log2(std::max(size.x, std::max(size.y, size.z)))) + 1;
    }

    glm::uvec3 MipLevelSize(std::uint32_t mipLevel = 0) const
    {
        return glm::max(size / glm::uvec3(1u << mipLevel), glm::uvec3(1, 1, 1));
    }

    ImageSubresource WholeLevel(std::uint32_t mipLevel = 0) const
    {
        ImageSubresource result{};

        result.image = image;
        result.offset = { 0, 0, 0 };
        result.extent = MipLevelSize(mipLevel);
        result.baseArrayLayer = 0;
        result.arrayLayerCount = arrayLayers;
        result.baseMipLevel = mipLevel;
        result.mipLevels = 1;
        result.aspects = GetAspects();
        result.layout = layout;
        result.format = format;

        return result;
    }
    ImageSubresource WholeLayer(std::uint32_t arrayLayer = 0) const
    {
        ImageSubresource result{};

        result.image = image;
        result.offset = { 0, 0, 0 };
        result.extent = size;
        result.baseArrayLayer = arrayLayer;
        result.arrayLayerCount = 1;
        result.baseMipLevel = 0;
        result.mipLevels = mipLevels;
        result.aspects = GetAspects();
        result.layout = layout;
        result.format = format;

        return result;
    }
    ImageSubresource WholeImage() const
    {
        ImageSubresource result{};

        result.image = image;
        result.offset = { 0, 0, 0 };
        result.extent = size;
        result.baseArrayLayer = 0;
        result.arrayLayerCount = arrayLayers;
        result.baseMipLevel = 0;
        result.mipLevels = mipLevels;
        result.aspects = GetAspects();
        result.layout = layout;
        result.format = format;

        return result;
    }

    bool IsWholeImage(ImageSubresource subresource) const
    {
        return subresource.offset == glm::uvec3{ 0, 0, 0 } && subresource.extent == size && subresource.baseArrayLayer == 0 && subresource.arrayLayerCount == arrayLayers && subresource.baseMipLevel == 0 && subresource.mipLevels == mipLevels && subresource.format == format && subresource.image == image;
    }

    void TransitionLayout(VkCommandBuffer commandBuffer, VkImageLayout oldLayout, VkImageLayout newLayout, PipelineBarrierData barrierData)
    {
        auto subres = WholeImage();
        subres.layout = oldLayout;
        subres.TransitionLayout(commandBuffer, newLayout, barrierData);

        layout = newLayout;
    }
    void TransitionLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout, PipelineBarrierData barrierData)
    {
        TransitionLayout(commandBuffer, layout, newLayout, barrierData);
    }
    void TransitionLayout(VkCommandBuffer commandBuffer, VkImageLayout newLayout)
    {
        auto bd = DeducePipelineBarrierData(layout, newLayout);
        TransitionLayout(commandBuffer, newLayout, bd);
    }

    VkImageAspectFlags GetAspects() const
    {
        return IdentifyAspects(format);
    }

    void GenerateMipMapChain(VkCommandBuffer buffer, std::uint32_t minLod, std::uint32_t maxLod)
    {
        auto dstLayout = layout;

        std::vector<ImageSubresource> levels;
        levels.reserve(maxLod - minLod);
        for (auto srcLevel = minLod; srcLevel <= maxLod; srcLevel++) levels.emplace_back(WholeLevel(srcLevel));

        PipelineBarrierData data{};
        data.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        data.sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        data.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        data.destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;

        //layers[0].TransitionLayout(buffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        for (auto srcLevel = 0ull; srcLevel < levels.size() - 1; srcLevel++)
        {
            levels[srcLevel].TransitionLayout(buffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, data);
            levels[srcLevel + 1].TransitionLayout(buffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, data);
            
            levels[srcLevel].Blit(buffer, levels[srcLevel + 1]);
        }

        PipelineBarrierData finalData{};
        finalData.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        finalData.sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        finalData.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        finalData.destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        for (auto& level : levels) level.TransitionLayout(buffer, dstLayout, finalData);

        layout = dstLayout;
    }
                  
    operator ImageSubresource() const
    {
        return WholeImage();
    }

    ~Image()
    {
        Destroy();
    }
};

template<typename ImageType>
concept ImageViewableConcept = std::is_base_of_v<VulkanResource, std::remove_reference_t<ImageType>> && requires(ImageType image)
{
    { image } -> std::convertible_to<ImageSubresource>;
};

struct ImageView : VulkanResource, MoveConstructOnly
{
    VkImageView view;

    template<ImageViewableConcept Image>
    ImageView(Image&& image, VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D) : ImageView(image.context, static_cast<ImageSubresource>(image), viewType)
    {

    }

    ImageView(VulkanContext& context, const ImageSubresource &subresource, VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D) : VulkanResource(context)
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = subresource.image;
        viewInfo.viewType = viewType;
        viewInfo.format = subresource.format;
        viewInfo.subresourceRange = subresource;
        
        if (vkCreateImageView(context.device, &viewInfo, nullptr, &view) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture image view!");
        }
    }

    ImageView(ImageView&& rhs) noexcept : VulkanResource(rhs.context), view(rhs.view)
    {
        rhs.view = VK_NULL_HANDLE;
    }

    operator VkImageView() const
    {
        return view;
    }

    ~ImageView()
    {
        if(view) vkDestroyImageView(context.device, view, nullptr);
    }
};
struct Sampler : VulkanResource, MoveConstructOnly
{
    VkSampler sampler;

    Sampler(VulkanContext& context) : VulkanResource(context)
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(context.physicalDevice, &properties);

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerInfo.unnormalizedCoordinates = VK_FALSE;
        samplerInfo.compareEnable = VK_FALSE;
        samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.minLod = 0.f;
        samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
        
        if (vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create texture sampler!");
        }

    }
    Sampler(Sampler&& rhs) noexcept : VulkanResource(rhs.context), sampler(rhs.sampler)
    {
        rhs.sampler = VK_NULL_HANDLE;
    }
    
    ~Sampler()
    {
        if(sampler) vkDestroySampler(context.device, sampler, nullptr);
    }
};

struct HTexture : MoveConstructOnly, VulkanResource
{
    std::vector<std::byte> pixels;
    glm::uvec2 size;

    Delayed<Image> image;
    Delayed<ImageView> imageView;
    Delayed<Sampler> sampler;

    HTexture(VulkanContext& context, const HvTree& tree, const HvTree& instance, HTexture* base, const HvTree& baseInstance) : VulkanResource(context)
    {
        size = tree["size"].Acquire<glm::uvec2>().value();
        
        auto data = tree["data"].Acquire<HvRawBinaryView>().value();
        
        bool floatData = tree["float"].Acquire<HvBool>().value();
        
        VkDeviceSize imageSize = size.x * size.y * (floatData ? 16 : 4);

        pixels.resize(imageSize);
        std::memcpy(pixels.data(), data.data, imageSize);

        
        auto format = instance["format"].Acquire<VkFormat>().value();
        auto mipLevels = instance["genMipLevels"].Acquire<int>().value();

        Image::CreateInfo ci{};
        ci.size = glm::uvec3(size, 1);
        ci.format = format;
        ci.imageUsage = VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.mipLevels = mipLevels;
        image.Create(context, ci);

        Image::CreateInfo stagingImageCI{};
        stagingImageCI.size = glm::uvec3(size, 1);
        stagingImageCI.format = floatData ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
        stagingImageCI.mipLevels = mipLevels;
        Image stagingImage(context, stagingImageCI);

        StagingBuffer sb(context, data.size, data.data);
        
        TempCommandBuffer cmd(context);

        image->TransitionLayout(cmd.buffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        stagingImage.TransitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

        stagingImage.WholeImage().CopyFrom(cmd.buffer, sb.buffer);
        
        stagingImage.TransitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        cmd.ExecuteAndReset();

        stagingImage.WholeImage().Blit(cmd, image->WholeImage());
        cmd.ExecuteAndReset();

        image->TransitionLayout(cmd.buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        cmd.ExecuteAndReset();

        if (image->mipLevels != 1)
        {
            image->GenerateMipMapChain(cmd, 0, image->mipLevels - 1);
            cmd.ExecuteAndReset();
        }

        imageView.Create(context, image->WholeImage());
        sampler.Create(context);
    }
};

struct TextureWriteWrapper : DescriptorSetWritable
{
    VkDescriptorImageInfo imageInfo;

    TextureWriteWrapper(VkImageView view, VkSampler sampler, VkImageLayout layout)
    {
        imageInfo.imageView = view;
        imageInfo.imageLayout = layout;
        imageInfo.sampler = sampler;
    }

    virtual VkWriteDescriptorSet GetWrite(std::size_t imageIndex) const override
    {      
        VkWriteDescriptorSet write{};

        write.pImageInfo = &imageInfo;
        
        return write;
    }
};
struct UniformBufferWriteWrapper : DescriptorSetWritable
{
    VkDescriptorBufferInfo bufferInfos[MAX_FRAMES_IN_FLIGHT];

    UniformBufferWriteWrapper(UniformBuffer &buffer)
    {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            bufferInfos[i].buffer = buffer.GetBuffer(i).buffer;
            bufferInfos[i].offset = 0;
            bufferInfos[i].range = VK_WHOLE_SIZE;
        }
    }

    virtual VkWriteDescriptorSet GetWrite(std::size_t imageIndex) const override
    {
        VkWriteDescriptorSet write{};

        write.pBufferInfo = &bufferInfos[imageIndex];

        return write;
    }

};
struct ImageWriteWrapper : DescriptorSetWritable
{
    VkDescriptorImageInfo imageInfo;

    ImageWriteWrapper(VkImageView view, VkImageLayout layout)
    {
        imageInfo.imageView = view;
        imageInfo.imageLayout = layout;
        imageInfo.sampler = VK_NULL_HANDLE;
    }

    virtual VkWriteDescriptorSet GetWrite(std::size_t imageIndex) const override
    {
        VkWriteDescriptorSet write{};

        write.pImageInfo = &imageInfo;

        return write;
    }

};

TextureWriteWrapper Describe(ImageView& imageView, Sampler& sampler)
{
    return TextureWriteWrapper(imageView.view, sampler.sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}
TextureWriteWrapper Describe(HTexture &texture)
{
    return TextureWriteWrapper(texture.imageView->view, texture.sampler->sampler, texture.image->layout);
}
UniformBufferWriteWrapper Describe(UniformBuffer& buffer)
{
    return UniformBufferWriteWrapper(buffer);
}
ImageWriteWrapper Describe(ImageView& imageView)
{
    return ImageWriteWrapper(imageView.view, VK_IMAGE_LAYOUT_GENERAL);
}

template<typename Type>
concept Describable =  requires(Type desc)
{
    { Describe(desc) } -> std::derived_from<DescriptorSetWritable>;
};

class HDescriptorPool : VulkanResource, MoveConstructOnly
{
    friend class HDescriptorSets;

    HDescriptorSetLayout& setLayout;

    VkDescriptorPool descriptorPool;

public:

    HDescriptorPool(VulkanContext& context, HDescriptorSetLayout& setLayout) : setLayout(setLayout), VulkanResource(context)
    {
        std::vector<VkDescriptorPoolSize> poolSizes;
        poolSizes.reserve(setLayout.bindings.size());

        for (auto& binding : setLayout.bindings)
        {
            VkDescriptorPoolSize size{};
            size.descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
            size.type = binding.descriptorType;

            poolSizes.push_back(size);
        }

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }
    HDescriptorPool(HDescriptorPool&& rhs) noexcept : VulkanResource(rhs.context), descriptorPool(rhs.descriptorPool), setLayout(rhs.setLayout)
    {
        rhs.descriptorPool = VK_NULL_HANDLE;
    }

    ~HDescriptorPool()
    {
        if (descriptorPool) vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    }
};
struct HDescriptorSets : VulkanResource, MoveConstructOnly
{
    HDescriptorPool& pool;

    std::vector<VkDescriptorSet> descriptorSets;

public:

    HDescriptorSets(HDescriptorPool& pool) : pool(pool), VulkanResource(pool.context)
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, pool.setLayout.layout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = pool.descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (auto h = vkAllocateDescriptorSets(context.device, &allocInfo, descriptorSets.data()); h != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
    }
    HDescriptorSets(HDescriptorSets&& rhs) noexcept : descriptorSets(std::move(rhs.descriptorSets)), pool(rhs.pool), VulkanResource(rhs.context)
    {

    }

    template<std::derived_from<DescriptorSetWritable> ...DescriptorTypes>
    void Update(DescriptorTypes... descriptors)
    {
        for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            std::vector<VkWriteDescriptorSet> writes{ (descriptors.GetWrite(i))... };

            for (std::size_t j = 0; j < writes.size(); j++)
            {
                writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[j].pNext = nullptr;

                writes[j].dstArrayElement = 0;
                writes[j].descriptorCount = pool.setLayout.bindings[j].descriptorCount;
                writes[j].descriptorType = pool.setLayout.bindings[j].descriptorType;
                //writes[j].dstBinding = pool.setLayout.bindings[j].binding;
                writes[j].dstBinding = j;
                
                writes[j].dstSet = descriptorSets[i];
            }

            vkUpdateDescriptorSets(context.device, writes.size(), writes.data(), 0, nullptr);
        }
    }

    template<Describable ...DescriptorTypes>
    void Update(DescriptorTypes&... descriptors)
    {
        Update(Describe(descriptors)...);
    }

    void Bind(VkCommandBuffer commandBuffer, VkPipelineLayout pipelineLayout, std::uint32_t imageIndex, VkPipelineBindPoint bindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, std::uint32_t set = 0)
    {
        vkCmdBindDescriptorSets(commandBuffer, bindPoint, pipelineLayout, set, 1, &descriptorSets[imageIndex], 0, nullptr);
    }
};

struct Drawable
{
    virtual VkPipeline GetPipeline() = 0;
    virtual VkBuffer GetVertexBuffer() = 0;
    virtual VkBuffer GetIndexBuffer() = 0;
    virtual std::span<VkDescriptorSet> GetDescriptorSets() = 0;
    virtual std::size_t IndexCount() = 0;
    virtual VkPipelineLayout GetLayout() = 0;
};

struct MeshInstance : public Drawable
{
    AssetReference<HMesh> mesh;
    AssetReference<Material> material;
    HDescriptorSets& sets;

    MeshInstance(AssetManager &assetManager, AssetDescriptor meshDescriptor, AssetInstanceDescriptor materialDescriptor, HDescriptorSets& sets) : mesh(assetManager, meshDescriptor), material(assetManager, materialDescriptor), sets(sets) {}

    virtual VkPipeline GetPipeline() override
    {
        return material->pipeline;
    }
    virtual VkBuffer GetVertexBuffer() override
    {
        return mesh->vertexBuffer.Get().buffer.buffer;
    }
    virtual VkBuffer GetIndexBuffer() override
    {
        return mesh->indexBuffer.Get().buffer.buffer;
    }
    virtual std::span<VkDescriptorSet> GetDescriptorSets() override
    {
        return sets.descriptorSets;
    }
    virtual std::size_t IndexCount() override
    {
        return mesh->indices.size();
    }
    virtual VkPipelineLayout GetLayout() override
    {
        return material->pipelineLayout;
    }
};

struct ComputePipeline : VulkanResource, NonCopyable
{
    std::map<std::uint32_t, HDescriptorSetLayout> setLayouts;

    VkPipelineLayout pipelineLayout;
    VkPipeline pipeline;

    ComputePipeline(VulkanContext &context, HvTree& tree, const HvTree& instance, ComputePipeline* base, const HvTree& baseInstance) : VulkanResource(context)
    {
        std::size_t offset = 0;
        std::vector<VkPushConstantRange> pushConstants;

        auto code = tree["code"].Acquire<HvRawBinaryView>().value();


        spirv_cross::Compiler comp(reinterpret_cast<const std::uint32_t*>(code.data), code.size / 4);

        VkShaderStageFlagBits stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        auto sr = comp.get_shader_resources();
        for (auto desc : sr.acceleration_structures)
        {
            auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
            auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

            VkDescriptorSetLayoutBinding layoutBinding;
            layoutBinding.binding = binding;
            layoutBinding.descriptorCount = 1;
            layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            layoutBinding.pImmutableSamplers = nullptr;
            layoutBinding.stageFlags = stageFlags;

            setLayouts[set].bindings.push_back(layoutBinding);
        }
        for (auto desc : sr.uniform_buffers)
        {
            auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
            auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

            VkDescriptorSetLayoutBinding layoutBinding;
            layoutBinding.binding = binding;
            layoutBinding.descriptorCount = 1;
            layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            layoutBinding.pImmutableSamplers = nullptr;
            layoutBinding.stageFlags = stageFlags;

            setLayouts[set].bindings.push_back(layoutBinding);
        }
        for (auto desc : sr.sampled_images)
        {
            auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
            auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

            VkDescriptorSetLayoutBinding layoutBinding;
            layoutBinding.binding = binding;
            layoutBinding.descriptorCount = 1;
            layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            layoutBinding.pImmutableSamplers = nullptr;
            layoutBinding.stageFlags = stageFlags;

            setLayouts[set].bindings.push_back(layoutBinding);

        }
        for (auto desc : sr.separate_images)
        {
            auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
            auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

            VkDescriptorSetLayoutBinding layoutBinding;
            layoutBinding.binding = binding;
            layoutBinding.descriptorCount = 1;
            layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            layoutBinding.pImmutableSamplers = nullptr;
            layoutBinding.stageFlags = stageFlags;

            setLayouts[set].bindings.push_back(layoutBinding);
        }
        for (auto desc : sr.storage_images)
        {
            auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
            auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

            VkDescriptorSetLayoutBinding layoutBinding;
            layoutBinding.binding = binding;
            layoutBinding.descriptorCount = 1;
            layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            layoutBinding.pImmutableSamplers = nullptr;
            layoutBinding.stageFlags = stageFlags;

            setLayouts[set].bindings.push_back(layoutBinding);
        }
        for (auto desc : sr.separate_samplers)
        {
            auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
            auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

            VkDescriptorSetLayoutBinding layoutBinding;
            layoutBinding.binding = binding;
            layoutBinding.descriptorCount = 1;
            layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            layoutBinding.pImmutableSamplers = nullptr;
            layoutBinding.stageFlags = stageFlags;

            setLayouts[set].bindings.push_back(layoutBinding);
        }
        for (auto desc : sr.storage_buffers)
        {
            auto binding = comp.get_decoration(desc.id, spv::DecorationBinding);
            auto set = comp.get_decoration(desc.id, spv::DecorationDescriptorSet);

            VkDescriptorSetLayoutBinding layoutBinding;
            layoutBinding.binding = binding;
            layoutBinding.descriptorCount = 1;
            layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            layoutBinding.pImmutableSamplers = nullptr;
            layoutBinding.stageFlags = stageFlags;

            setLayouts[set].bindings.push_back(layoutBinding);
        }

        for (auto pushConstant : sr.push_constant_buffers)
        {
            auto type = comp.get_type(pushConstant.type_id);
            std::size_t size = comp.get_declared_struct_size(type);

            auto& newRange = pushConstants.emplace_back();
            newRange.offset = offset;
            newRange.size = size;
            newRange.stageFlags = stageFlags;

            offset += size;
        }

        for (auto& layout : setLayouts)
        {
            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = static_cast<uint32_t>(layout.second.bindings.size());
            layoutInfo.pBindings = layout.second.bindings.data();

            if (vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &layout.second.layout) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create descriptor set layout!");
            }
        }

        std::vector<VkDescriptorSetLayout> vkSetLayouts;
        for (auto& layout : setLayouts)
        {
            vkSetLayouts.push_back(layout.second.layout);
        }

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = vkSetLayouts.size();
        pipelineLayoutInfo.pSetLayouts = vkSetLayouts.data();

        pipelineLayoutInfo.pPushConstantRanges = pushConstants.data();
        pipelineLayoutInfo.pushConstantRangeCount = pushConstants.size();


        if (vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkComputePipelineCreateInfo pipelineInfo{};

        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = pipelineLayout;

        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.module = createShaderModule(context, code.data, code.size);
        pipelineInfo.stage.stage = stageFlags;

        auto entry = tree["mainFunction"].Acquire<std::string>().value();
        pipelineInfo.stage.pName = entry.c_str();


        if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }
    void Bind(VkCommandBuffer cmdBuffer)
    {
        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    }

    ~ComputePipeline()
    {
        vkDestroyPipeline(context.device, pipeline, nullptr);
        vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    }
};


struct Mesh : public Transformable
{
    friend class Scene;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    //std::vector<AssetReference<Texture>> textures;

public:

    Mesh()
    {

    }

    void LoadFromAssimpMesh(aiMesh* mesh, const aiScene* scene, fs::path directory)
    {
        for (std::size_t i = 0; i < mesh->mNumVertices; i++)
        {
            Vertex vertex;

            std::memcpy(&vertex.pos, &mesh->mVertices[i], sizeof(glm::vec3));

            vertex.pos = AssimpToHvost3() * vertex.pos;

            std::memcpy(&vertex.normal, &mesh->mNormals[i], sizeof(glm::vec3));

            vertex.normal = AssimpToHvost3() * vertex.normal;

            if (mesh->mTextureCoords[0])
            {
                glm::vec2 vec;
                vec.x = mesh->mTextureCoords[0][i].x;
                vec.y = mesh->mTextureCoords[0][i].y;
                vertex.texCoord = vec;
            }
            else
                vertex.texCoord = glm::vec2(0.0f, 0.0f);

            std::memcpy(&vertex.tangent, &mesh->mTangents[i], sizeof(glm::vec3));

            vertex.tangent = AssimpToHvost3() * vertex.tangent;

            vertices.push_back(vertex);
        }

        for (std::size_t i = 0; i < mesh->mNumFaces; i++)
        {
            aiFace face = mesh->mFaces[i];
            for (std::size_t j = 0; j < face.mNumIndices; j++)
                indices.push_back(face.mIndices[j]);
        }

        if (mesh->mMaterialIndex >= 0)
        {
            auto material = scene->mMaterials[mesh->mMaterialIndex];

            LoadMaterialTextures(material, aiTextureType_BASE_COLOR, directory);
            LoadMaterialTextures(material, aiTextureType_DIFFUSE, directory);
            LoadMaterialTextures(material, aiTextureType_EMISSIVE, directory);
        }

        std::cout << "Mesh loaded: " << std::endl;
        std::cout << "  Vertex count:" << mesh->mNumVertices << std::endl;
        std::cout << "  Index count:" << indices.size() << std::endl;
        //std::cout << "  Texture count:" << textures.size() << std::endl;
    }
    void LoadMaterialTextures(aiMaterial* mat, aiTextureType type, fs::path directory)
    {
        for (unsigned int i = 0; i < mat->GetTextureCount(type); i++)
        {
            aiString str;
            mat->GetTexture(type, i, &str);

            auto file = directory / str.C_Str();

            AssetDescriptor desc{ file, "texture"_tp };
            //textures.emplace_back(assetManager, desc, false);

            /*Texture& texture = textures.emplace_back();
            texture.LoadFromFile(directory / str.C_Str());*/
        }
    }
};
struct Scene
{
    std::vector<Mesh> meshes;

    bool LoadFromFile(fs::path file)
    {
        if (!fs::is_regular_file(file)) return false;

        Assimp::Importer importer;
        
        const aiScene* scene = importer.ReadFile(file.generic_string().c_str(), aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_FlipWindingOrder | aiProcess_JoinIdenticalVertices | aiProcess_ForceGenNormals | aiProcess_CalcTangentSpace);

        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            std::cout << importer.GetErrorString() << std::endl;
            return false;
        }

        LoadNode(scene->mRootNode, scene, file.parent_path());

        return true;
    }
    void LoadNode(aiNode* node, const aiScene* scene, fs::path directory)
    {
        for (unsigned int i = 0; i < node->mNumMeshes; i++)
        {
            aiMesh* aiMesh = scene->mMeshes[node->mMeshes[i]];
            auto& mesh = meshes.emplace_back();
            mesh.LoadFromAssimpMesh(aiMesh, scene, directory);
        }

        for (unsigned int i = 0; i < node->mNumChildren; i++)
            LoadNode(node->mChildren[i], scene, directory);
    }
};

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);

VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    for (const auto& availableFormat : availableFormats)
    {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT)
        {
            return availableFormat;
        }
    }

    return availableFormats[0];
}
VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    for (const auto& availablePresentMode : availablePresentModes)
    {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return availablePresentMode;
        }
    }

    return VK_PRESENT_MODE_FIFO_KHR;
}

VkFormat findSupportedFormat(VulkanContext& context, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates)
    {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(context.physicalDevice, format, &props);

        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
        {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
        {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}
VkFormat findDepthFormat(VulkanContext& context)
{
    return findSupportedFormat(context,
        { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
    );
}

void RenderDrawable(VkCommandBuffer commandBuffer, Drawable& drawable, uint32_t instanceCount, std::uint32_t currentFrame)
{
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, drawable.GetPipeline());
    auto sets = drawable.GetDescriptorSets();
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, drawable.GetLayout(), 0, 1, &(sets[currentFrame]), 0, nullptr);

    VkDeviceSize offsets[] = { 0 };

    auto vertexBuffer = drawable.GetVertexBuffer();
    auto indexBuffer = drawable.GetIndexBuffer();

    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(drawable.IndexCount()), instanceCount, 0, 0, 0);
}
void RenderDrawable(VkCommandBuffer commandBuffer, Drawable& drawable, std::uint32_t currentFrame = 0)
{
    RenderDrawable(commandBuffer, drawable, 1, currentFrame);
}

void StoreMesh(HvTree& dst, Mesh& mesh)
{
    dst["vertexCount"].Store(mesh.vertices.size());
    dst["indexCount"].Store(mesh.indices.size());
    dst["vertices"].Store(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex));
    dst["indices"].Store(mesh.indices.data(), mesh.indices.size() * sizeof(std::uint32_t));
}
void StoreTexture(HvTree& dst, Texture& texture)
{
    dst["size"].Store(texture.size);
    dst["data"].Store(texture.textureData.data(), texture.textureData.size());
    dst["float"].Store(HvBool(texture.useFloat));
}

void StoreMaterial(HvTree& dst, std::span<ShaderStageDescriptor> descriptors, Material::PipelineCreateInfo createInfo, std::span<VkVertexInputAttributeDescription> attributes, std::span<VkVertexInputBindingDescription> bindings)
{
    dst["pipelineCreateInfo"].Store(createInfo);
    dst["attributes"].StoreArray<VkVertexInputAttributeDescription>(attributes);
    dst["bindings"].StoreArray<VkVertexInputBindingDescription>(bindings);
    dst["stages/count"_tp].Store(descriptors.size());
    for (std::size_t i = 0; i < descriptors.size(); i++)
    {
        auto& descDst = dst["stages"][std::to_string(i)];
        descDst["stageType"].Store(descriptors[i].stage);
        descDst["mainFunction"].Store(std::string(descriptors[i].mainFunction));

        ShaderCodeLoader loader(descriptors[i]);

        descDst["code"].Store(loader.code, loader.size);
    }
}
void StoreComputePipeline(HvTree& dst, ShaderStageDescriptor stage)
{
    dst["mainFunction"].Store(std::string(stage.mainFunction));

    ShaderCodeLoader loader(stage);

    dst["code"].Store(loader.code, loader.size);
}

void SaveHvToFile(const HvTree& tree, fs::path file)
{
    std::byte* data;
    std::size_t size;
    tree.SaveBinary(data, size);

    std::ofstream out(file, std::ios::binary);
    out.write(reinterpret_cast<char*>(data), size);

    delete[] data;
}
void ReadHvFromFile(HvTree& dst, fs::path file)
{
    auto size = fs::file_size(file);
    auto data = new std::byte[size];
    std::ifstream in(file, std::ios::binary);
    in.read(reinterpret_cast<char*>(data), size);
    dst.LoadBinary(data, size);
}
 
std::size_t IndexByCoordinates(glm::uvec2 coords, glm::uvec2 segments)
{
    return (coords.x % segments.x) + (coords.y % segments.y) * segments.x;
}
void GenerateSphere(Mesh &mesh, glm::uvec2 segments = glm::uvec2(64, 64))
{
    segments.x++;
    mesh.vertices.resize((segments.x) * segments.y);
    mesh.indices.reserve((segments.x) * (segments.y) * 2);

    for (std::size_t row = 1; row < segments.y; row++)
    {
        for (std::size_t col = 0; col < segments.x - 1; col++)
        {
            mesh.indices.push_back(IndexByCoordinates(glm::uvec2(col, row), segments));
            mesh.indices.push_back(IndexByCoordinates(glm::uvec2(col, row - 1), segments));
            mesh.indices.push_back(IndexByCoordinates(glm::uvec2(col + 1, row - 1), segments));
        }
        for (std::size_t col = 1; col < segments.x; col++)
        {
            mesh.indices.push_back(IndexByCoordinates(glm::uvec2(col, row), segments));
            mesh.indices.push_back(IndexByCoordinates(glm::uvec2(col - 1, row), segments));
            mesh.indices.push_back(IndexByCoordinates(glm::uvec2(col, row - 1), segments));
        }
    }

    using namespace std::numbers;
    for (auto x = 0ull; x < segments.x; x++)
        for (auto y = 0ull; y < segments.y; y++)
        {
            double fx = double(x) / (segments.x - 1) + 0.00000001;
            double fy = double(y) / (segments.y - 1) + 0.00000001;

            double ax = fx * pi * 2;
            double ay = (fy - 0.5) * pi;

            Vertex &vertex = mesh.vertices[IndexByCoordinates(glm::uvec2(x, y), segments)];

            glm::vec3 pos{};
            pos.x = std::cos(ax) * std::cos(ay);
            pos.z = std::sin(ay);
            pos.y = std::sin(ax) * std::cos(ay);

            vertex.pos = pos;

            vertex.texCoord = glm::vec2(fx, fy);

            vertex.normal = glm::normalize(pos);

            
        }

    for (auto x = 0ull; x < segments.x; x++)
        for (auto y = 0ull; y < segments.y; y++)
        {
            double fx = double(x) / (segments.x - 1);
            double fy = double(y) / (segments.y - 1);

            double ax = fx * pi * 2;
            double ay = (fy - 0.5) * pi;

            Vertex& vertex = mesh.vertices[IndexByCoordinates(glm::uvec2(x, y), segments)];
            Vertex& right = mesh.vertices[IndexByCoordinates(glm::uvec2(x == segments.x - 1 ? 1 : x + 1, y), segments)];
            
            vertex.tangent = glm::normalize(right.pos - vertex.pos);
        }
}
 
void GeneratePlane(Mesh &mesh, glm::vec2 size)
{
    mesh.vertices.resize(4);

    mesh.vertices[0].pos = glm::vec3(-size.x, -size.y, 0.f);
    mesh.vertices[0].normal = glm::vec3(0.f, 0.f, 1.f);
    mesh.vertices[0].tangent = glm::vec3(0.f, 1.f, 0.f);
    mesh.vertices[0].texCoord = glm::vec2(0.f, 1.f);

    mesh.vertices[1].pos = glm::vec3(size.x, -size.y, 0.f);
    mesh.vertices[1].normal = glm::vec3(0.f, 0.f, 1.f);
    mesh.vertices[1].tangent = glm::vec3(0.f, 1.f, 0.f);
    mesh.vertices[1].texCoord = glm::vec2(0.f, 0.f);

    mesh.vertices[2].pos = glm::vec3(size.x, size.y, 0.f);
    mesh.vertices[2].normal = glm::vec3(0.f, 0.f, 1.f);
    mesh.vertices[2].tangent = glm::vec3(0.f, 1.f, 0.f);
    mesh.vertices[2].texCoord = glm::vec2(1.f, 0.f);

    mesh.vertices[3].pos = glm::vec3(-size.x, size.y, 0.f);
    mesh.vertices[3].normal = glm::vec3(0.f, 0.f, 1.f);
    mesh.vertices[3].tangent = glm::vec3(0.f, 1.f, 0.f);
    mesh.vertices[3].texCoord = glm::vec2(1.f, 1.f);

    mesh.indices = { 0, 1, 2, 0, 2, 3};
}

struct Particle
{
    glm::vec4 position;
    glm::vec4 rotation;
    glm::vec4 deltaPos;
    glm::vec4 deltaRotation;
};
   
struct ComputePushConstants
{
    std::uint64_t inAddress;
    std::uint64_t outAddress;
    float deltaTime;
};

void hinput(const GLFW::Window& window)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - lastTime).count();
    lastTime = currentTime;

    using enum GLFW::Action;
    if (window.GetKeyState(GLFW_KEY_W) == PRESS) camera.AddOffset(glm::vec4(1.f, 0.f, 0.f, 0.f) * speed * deltaTime);
    if (window.GetKeyState(GLFW_KEY_S) == PRESS) camera.AddOffset(glm::vec4(-1.f, 0.f, 0.f, 0.f) * speed * deltaTime);
    if (window.GetKeyState(GLFW_KEY_A) == PRESS) camera.AddOffset(glm::vec4(0.f, -1.f, 0.f, 0.f) * speed * deltaTime);
    if (window.GetKeyState(GLFW_KEY_D) == PRESS) camera.AddOffset(glm::vec4(0.f, 1.f, 0.f, 0.f) * speed * deltaTime);
    if (window.GetKeyState(GLFW_KEY_E) == PRESS) camera.AddOffset(glm::vec4(0.f, 0.f, 1.f, 0.f) * speed * deltaTime, false);
    if (window.GetKeyState(GLFW_KEY_Q) == PRESS) camera.AddOffset(glm::vec4(0.f, 0.f, -1.f, 0.f) * speed * deltaTime, false);
    if (window.GetKeyState(GLFW_KEY_F) == PRESS) camera.hdebug();

    static glm::dvec2 lastMousePos = window.GetCursorPosition();

    glm::dvec2 mousePos = window.GetCursorPosition();
    glm::dvec2 delta = mousePos - lastMousePos;
    lastMousePos = mousePos;

    camera.AddRotation(glm::vec3(0.f, -delta.y, -delta.x) * rotationSpeed);
}
VkExtent2D chooseSwapExtent(const GLFW::Window& window, const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    else
    {
        auto size = window.GetFramebufferSize();

        VkExtent2D actualExtent = Extent(size);

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

struct Swapchain : VulkanResource, MoveConstructOnly
{
    VkSwapchainKHR swapchain;
    std::uint32_t imageCount;
    std::vector<VkImage> rawImages;
    VkFormat format;
    VkExtent2D extent;
    std::vector<ImageSubresource> imageRefs;
    std::vector<ImageView> imageViews;

    bool valid;

    //std::function<void()> recreateCallback;


    SwapChainSupportDetails QuerySwapChainSupport()
    {
        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(context.physicalDevice, context.surface, &details.capabilities);

        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(context.physicalDevice, context.surface, &formatCount, nullptr);

        if (formatCount != 0)
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(context.physicalDevice, context.surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(context.physicalDevice, context.surface, &presentModeCount, nullptr);

        if (presentModeCount != 0)
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(context.physicalDevice, context.surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    Swapchain(VulkanContext& context, const GLFW::Window& window) : valid(false), VulkanResource(context)
    {
        Create(window);
    }

    void Create(const GLFW::Window& window)
    {
        if (valid) Destroy();

        SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport();

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        extent = chooseSwapExtent(window, swapChainSupport.capabilities);

        imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = context.surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT;

        QueueFamilyIndices indices = context.queueFamilyIndices;
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily)
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
                                                                                                                
        auto result = vkCreateSwapchainKHR(context.device, &createInfo, nullptr, &swapchain);
        if (result != VK_SUCCESS)
        {
            //throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(context.device, swapchain, &imageCount, nullptr);
        rawImages.resize(imageCount);
        vkGetSwapchainImagesKHR(context.device, swapchain, &imageCount, rawImages.data());

        format = surfaceFormat.format;
        valid = true;

        TempCommandBuffer tmpBuffer(context);
        imageRefs.reserve(imageCount);
        for (int i = 0; i < imageCount; i++)
        {
            ImageSubresource res{};
            
            res.image = rawImages[i];
            res.format = format;
            res.offset = { 0, 0, 0 };
            res.extent = glm::uvec3(Extent(extent), 1);
            res.baseArrayLayer = 0;
            res.baseMipLevel = 0;
            res.arrayLayerCount = 1;
            res.mipLevels = 1;
            res.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
            res.layout = VK_IMAGE_LAYOUT_UNDEFINED;

            imageRefs.emplace_back(res);
            imageRefs[i].TransitionLayout(tmpBuffer, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        }

        imageViews.reserve(imageCount);
        for (int i = 0; i < imageCount; i++)
            imageViews.emplace_back(context, imageRefs[i]);

        context.swapChainImageFormat = format;
    }
    void Destroy()
    {
        if (valid)
        {
            valid = false;
            vkDestroySwapchainKHR(context.device, swapchain, nullptr);
            imageRefs.clear();
            imageViews.clear();
            rawImages.clear();
        }
    }

    void Recreate(const GLFW::Window& window)
    {
        auto size = window.GetFramebufferSize();
        while (size.x == 0 || size.y == 0)
        {
            size = window.GetFramebufferSize();
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(context.device);

        Create(window);
        //recreateCallback();
    }

    std::uint32_t AcquireNextImage(VkSemaphore semaphore, const GLFW::Window& window)
    {
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(context.device, swapchain, UINT64_MAX, semaphore, VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
            Recreate(window);
        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("failed to acquire swap chain image!");

        return imageIndex;
    }

    ~Swapchain()
    {
        Destroy();
    }
};

struct SwapchainImageWriteWrapper : DescriptorSetWritable
{
    std::vector<VkDescriptorImageInfo> imageInfos;

    SwapchainImageWriteWrapper(Swapchain& swapchain)
    {
        imageInfos.resize(swapchain.imageCount);

        for (std::size_t i = 0; i < imageInfos.size(); i++)
        {
            imageInfos[i].imageView = swapchain.imageViews[i];
            imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageInfos[i].sampler = VK_NULL_HANDLE;
        }
    }

    virtual VkWriteDescriptorSet GetWrite(std::size_t imageIndex) const override
    {
        VkWriteDescriptorSet write{};

        write.pImageInfo = &imageInfos[imageIndex];
        
        return write;
    }
};
SwapchainImageWriteWrapper Describe(Swapchain& swapchain)
{
    return SwapchainImageWriteWrapper(swapchain);
}


template<>
struct HvValueConvert<VkVertexInputBindingDescription>
{
    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_vk_vertex_input_binding_description"sv);
    }

    static bool HvToType(const std::byte* data, std::size_t size, VkVertexInputBindingDescription& value)
    {
        Read(data, value);
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const VkVertexInputBindingDescription& obj)
    {
        size = sizeof(VkVertexInputBindingDescription);
        data = new std::byte[size];
        std::memcpy(data, &obj, size);
        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<VkVertexInputBindingDescription>& array)
    {
        array.resize(size / sizeof(VkVertexInputBindingDescription));
        std::memcpy(array.data(), data, size);
        return true;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const VkVertexInputBindingDescription> array)
    {
        size = array.size() * sizeof(VkVertexInputBindingDescription);
        data = new std::byte[size];
        std::memcpy(data, array.data(), size);
        return true;
    }
};
template<>
struct HvValueConvert<VkVertexInputAttributeDescription>
{
    static constexpr std::uint64_t GetType()
    {
        return Hash("builtin_vk_vertex_input_attribute_description"sv);
    }

    static bool HvToType(const std::byte* data, std::size_t size, VkVertexInputAttributeDescription& value)
    {
        Read(data, value);
    }
    static bool TypeToHv(std::byte*& data, std::size_t& size, const VkVertexInputAttributeDescription& obj)
    {
        size = sizeof(VkVertexInputAttributeDescription);
        data = new std::byte[size];
        std::memcpy(data, &obj, size);
        return true;
    }

    static bool HvToTypeArray(const std::byte* data, std::size_t size, std::vector<VkVertexInputAttributeDescription>& array)
    {
        array.resize(size / sizeof(VkVertexInputAttributeDescription));
        std::memcpy(array.data(), data, size);
        return true;
    }
    static bool TypeArrayToHv(std::byte*& data, std::size_t& size, std::span<const VkVertexInputAttributeDescription> array)
    {
        size = array.size() * sizeof(VkVertexInputAttributeDescription);
        data = new std::byte[size];
        std::memcpy(data, array.data(), size);
        return true;
    }
};

struct UniformBufferPBR
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
    alignas(16) glm::mat4 axisSwizzle;

    alignas(16) glm::mat4 normal;

    alignas(16) glm::vec3 cameraPos;
};

struct alignas(16) PointLight
{
    alignas(16) glm::vec4 position;
    alignas(16) glm::vec4 color;
};

struct PushConstantsPBR
{
    std::uint64_t lightsBuffer;
    int count;
    float indirectMultiplier;
};

struct EquiToCubeUBO
{
    glm::mat4 proj;
    glm::mat4 axisSwizzle;
    glm::mat4 view[6];
};

struct ClearValue
{
    VkClearValue value;

    ClearValue(glm::vec4 color)
    {
        std::memcpy(&value.color.float32, &color, sizeof(glm::vec4));
    }
    ClearValue(glm::ivec4 color)
    {
        std::memcpy(&value.color.int32, &color, sizeof(glm::ivec4));
    }
    ClearValue(glm::uvec4 color)
    {
        std::memcpy(&value.color.uint32, &color, sizeof(glm::uvec4));
    }

    ClearValue(float depth = 0.f, std::uint32_t stencil = 0)
    {
        value.depthStencil.depth = depth;
        value.depthStencil.stencil = stencil;
    }
};
struct RenderPassAttachment
{
    VkRenderingAttachmentInfo info;

    RenderPassAttachment(ImageView &image, ClearValue clearValue = ClearValue(glm::vec4(0.f)), VkImageLayout layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VkAttachmentLoadOp loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, VkAttachmentStoreOp storeOp = VK_ATTACHMENT_STORE_OP_STORE) : info{}
    {
        info.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        info.imageView = image;
        info.clearValue = clearValue.value;
        info.imageLayout = layout;
        info.loadOp = loadOp;
        info.storeOp = storeOp;
    }
};
struct RenderPass
{
    VkRenderingInfo renderingInfo;
    std::vector<RenderPassAttachment> colorAttachments;
    std::optional<RenderPassAttachment> depthStencilAttachment;

    struct Info
    {
        int layerCount = 1;
        bool useStencil = false;
        bool useDepth = true;
    };
    RenderPass(Rect<> renderArea, std::vector<RenderPassAttachment> colorAttachments, RenderPassAttachment depthStencilAttachment, Info info = Info{}) : renderingInfo{}, colorAttachments(colorAttachments), depthStencilAttachment(depthStencilAttachment)
    {
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
        renderingInfo.renderArea = renderArea;
        renderingInfo.layerCount = info.layerCount;

        renderingInfo.pColorAttachments = reinterpret_cast<VkRenderingAttachmentInfo*>(this->colorAttachments.data());
        renderingInfo.colorAttachmentCount = colorAttachments.size();
        renderingInfo.pDepthAttachment = info.useDepth ? &this->depthStencilAttachment.value().info : VK_NULL_HANDLE;
        renderingInfo.pStencilAttachment = info.useStencil ? &this->depthStencilAttachment.value().info : VK_NULL_HANDLE;
    }
    RenderPass(Rect<> renderArea, std::vector<RenderPassAttachment> colorAttachments, Info info = Info{}) : renderingInfo{}, colorAttachments(colorAttachments), depthStencilAttachment(std::nullopt)
    {
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR;
        renderingInfo.renderArea = renderArea;
        renderingInfo.layerCount = info.layerCount;

        renderingInfo.pColorAttachments = reinterpret_cast<VkRenderingAttachmentInfo*>(this->colorAttachments.data());
        renderingInfo.colorAttachmentCount = colorAttachments.size();
        renderingInfo.pDepthAttachment = VK_NULL_HANDLE;
        renderingInfo.pStencilAttachment = VK_NULL_HANDLE;
    }

    void Begin(VkCommandBuffer commandBuffer)
    {
        vkCmdBeginRendering(commandBuffer, &renderingInfo);
    }
    void End(VkCommandBuffer commandBuffer)
    {
        vkCmdEndRendering(commandBuffer);
    }
};

void DispatchBatched(TempCommandBuffer& commandBuffer, glm::uvec3 dimensions, glm::uvec3 batchSize)
{
    auto batches = dimensions / batchSize;

    glm::uvec3 batch(0);
    for(batch.x = 0; batch.x < batches.x; batch.x++)
        for(batch.y = 0; batch.y < batches.y; batch.y++)
            for (batch.z = 0; batch.z < batches.z; batch.z++)
            {
                vkCmdDispatchBase(commandBuffer, batch.x * batchSize.x, batch.y * batchSize.y, batch.z * batchSize.z, batchSize.x, batchSize.y, batchSize.z);
                commandBuffer.ExecuteAndReset();
            }
}

void EquirectangularToCubemapCompute(AssetManager& assetManager, VulkanContext& context, Image& equirect, Image& cubemap)
{
    AssetReference<ComputePipeline> pipeline(assetManager, AssetInstanceDescriptor{ AssetDescriptor{"huinya.ab", "comp/equiToCube"_tp}, ""_tp});

    HDescriptorPool pool(context, pipeline->setLayouts[0]);
    HDescriptorSets set(pool);

    ImageView equirectView(equirect);
    ImageView cubemapView(cubemap, VK_IMAGE_VIEW_TYPE_CUBE);

    Sampler sampler(context);

    set.Update(Describe(equirectView, sampler), Describe(cubemapView));

    TempCommandBuffer cmdBuffer(context);

    cubemap.TransitionLayout(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL);

    cmdBuffer.ExecuteAndReset();

    pipeline->Bind(cmdBuffer);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipelineLayout, 0, 1, set.descriptorSets.data(), 0, nullptr);
    
    vkCmdDispatch(cmdBuffer, cubemap.size.x / 32, cubemap.size.y / 32, 6);
}

/*
void ConvolutionIBL(Image& environment, Image& dst)
{
    AssetReference<ComputePipeline> pipeline(assetManager, AssetInstanceDescriptor{ AssetDescriptor{"huinya.ab", "comp/iblConvolution"_tp}, ""_tp });

    HDescriptorPool pool(pipeline->setLayouts[0]);
    HDescriptorSets set(pool);

    ImageView environmentView(environment.WholeImage(), VK_IMAGE_VIEW_TYPE_CUBE);

    Sampler sampler;
    
    TempCommandBuffer cmdBuffer;
    dst.TransitionLayout(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL);
    cmdBuffer.ExecuteAndReset();
    
    for (int level = 0; level < dst.mipLevels; level++)
    {
        ImageView dstView(dst.WholeLevel(level), VK_IMAGE_VIEW_TYPE_CUBE);
        set.Update(Describe(environmentView, sampler), Describe(dstView));

        pipeline->Bind(cmdBuffer);
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipelineLayout, 0, 1, set.descriptorSets.data(), 0, nullptr);
        auto levelSize = glm::max(dst.MipLevelSize(level), glm::uvec3(32, 32, 1));

        float roughness = (float)level / float(dst.mipLevels - 1);
        vkCmdPushConstants(cmdBuffer, pipeline->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &roughness);

        vkCmdDispatch(cmdBuffer, levelSize.x / 32, levelSize.y / 32, 6);
        cmdBuffer.ExecuteAndReset();
    }
}
void ComputeBRDFLUT(Image& LUT)
{
    AssetReference<ComputePipeline> pipeline(assetManager, AssetInstanceDescriptor{ AssetDescriptor{"huinya.ab", "comp/brdfLUT"_tp}, ""_tp });

    HDescriptorPool pool(pipeline->setLayouts[0]);
    HDescriptorSets set(pool);

    ImageView dstView(LUT.WholeImage());

    TempCommandBuffer cmdBuffer;
    LUT.TransitionLayout(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL);
    cmdBuffer.ExecuteAndReset();

    set.Update(Describe(dstView));

    pipeline->Bind(cmdBuffer);
    vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipelineLayout, 0, 1, set.descriptorSets.data(), 0, nullptr);

    vkCmdDispatch(cmdBuffer, LUT.size.x / 32, LUT.size.y / 32, 1);    
}
*/

void RadiantIntensity(AssetManager& assetManager, VulkanContext& context, Image& environment, Image& dst)
{
    AssetReference<ComputePipeline> pipeline(assetManager, AssetInstanceDescriptor{ AssetDescriptor{"huinya.ab", "comp/radiantIntensity"_tp}, ""_tp });

    HDescriptorPool pool(context, pipeline->setLayouts[0]);
    HDescriptorSets set(pool);

    ImageView environmentView(environment, VK_IMAGE_VIEW_TYPE_CUBE);

    Sampler sampler(context);

    TempCommandBuffer cmdBuffer(context);
    dst.TransitionLayout(cmdBuffer, VK_IMAGE_LAYOUT_GENERAL);
    cmdBuffer.ExecuteAndReset();

    for (int level = 0; level < dst.mipLevels; level++)
    {
        ImageView dstView(context, dst.WholeLevel(level), VK_IMAGE_VIEW_TYPE_CUBE);
        set.Update(Describe(environmentView, sampler), Describe(dstView));

        pipeline->Bind(cmdBuffer);
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->pipelineLayout, 0, 1, set.descriptorSets.data(), 0, nullptr);
        auto levelSize = glm::max(dst.MipLevelSize(level), glm::uvec3(32, 32, 1));

        const float maxTheta = 1.26;

        float h = (float)level / float(dst.mipLevels - 1);
        float theta = h * maxTheta + 1e-4;
        vkCmdPushConstants(cmdBuffer, pipeline->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &theta);

        vkCmdDispatch(cmdBuffer, levelSize.x / 32, levelSize.y / 32, 6);
        cmdBuffer.ExecuteAndReset();
    }
}

Image LoadRaw(VulkanContext& context, fs::path file, glm::uvec3 size, std::uint64_t components = 1)
{
    std::ifstream in(file, std::ios::binary);

    std::vector<double> rawBuffer(fs::file_size(file) / sizeof(double));
    in.read(reinterpret_cast<char*>(rawBuffer.data()), rawBuffer.size() * sizeof(double));

    std::vector<float> floatBuffer(size.x * size.y * size.z * components);
    for (std::size_t i = 0; i < rawBuffer.size(); i++)
    {
        floatBuffer[i] = static_cast<float>(rawBuffer[i]);
    }

    StagingBuffer stagingBuffer(context, floatBuffer.size() * sizeof(float), floatBuffer.data());

    std::array formats = { VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32G32_SFLOAT };

    Image::CreateInfo ci{};
    ci.size = size;
    ci.format = formats[components - 1];
    ci.mipLevels = 1;
    ci.imageType = size.z == 1 ? VK_IMAGE_TYPE_2D : VK_IMAGE_TYPE_3D;

    ci.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    Image image(context, ci);

    TempCommandBuffer commandBuffer(context);

    image.TransitionLayout(commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    commandBuffer.ExecuteAndReset();

    image.WholeImage().CopyFrom(commandBuffer, stagingBuffer.buffer);

    commandBuffer.Execute();

    return image;
}

struct GraphPushConstants
{
    std::uint32_t resolution;
    float textureScale;
};

static void key_callback(GLFW::Window::KeyCallbackArgs args)
{
    using enum GLFW::Action;
    if (args.key == GLFW_KEY_R && args.action == PRESS)
    {
        camera.perspective = !camera.perspective;
        //camera.UpdateProjection(glm::uvec2(swapChainExtent.width, swapChainExtent.height));
    }

    if (args.key == GLFW_KEY_T && args.action == PRESS)
    {
        indirectMultiplier += indirectMultiplierStep;
        std::println("Indirect multiplier: {}", indirectMultiplier);
    }
    if (args.key == GLFW_KEY_G && args.action == PRESS)
    {
        indirectMultiplier -= indirectMultiplierStep;
        std::println("Indirect multiplier: {}", indirectMultiplier);
    }

    if (args.key == GLFW_KEY_Y && args.action == PRESS)
    {
        indirectMultiplier += indirectMultiplierStep2;
        std::println("Indirect multiplier: {}", indirectMultiplier);
    }
    if (args.key == GLFW_KEY_H && args.action == PRESS)
    {
        indirectMultiplier -= indirectMultiplierStep2;
        std::println("Indirect multiplier: {}", indirectMultiplier);
    }
}

struct DescriptorSetLayout : VulkanResource, MoveConstructOnly
{
    VkDescriptorSetLayout layout;

    DescriptorSetLayout(std::span<VkDescriptorSetLayoutBinding> bindings, VulkanContext& context) : VulkanResource(context), layout(VK_NULL_HANDLE)
    {
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.pBindings = bindings.data();
        layoutInfo.bindingCount = bindings.size();

        vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &layout);
    }

    auto Handle() const
    {
        return layout;
    }

    ~DescriptorSetLayout()
    {
        vkDestroyDescriptorSetLayout(context.device, layout, nullptr);
    }
};

template<>
struct hv::conversion::HvToType<VkDescriptorSetLayoutBinding>
{
    static inline constexpr bool UseBinary = false;

    template<typename DescriptorType, typename ...Args>
    static VkDescriptorSetLayoutBinding Convert(DescriptorType desc, Args&&... args)
    {
        VkDescriptorSetLayoutBinding binding{};

        binding.binding = desc["binding"_sk];
        binding.descriptorCount = desc["descriptorCount"_sk];
        binding.descriptorType = desc["descriptorType"_sk];
        binding.stageFlags = desc["stageFlags"_sk];
        binding.pImmutableSamplers = nullptr;

        return binding;
    }
};
template<>
struct hv::conversion::TypeToHv<VkDescriptorSetLayoutBinding>
{
    static inline constexpr bool UseBinary = false;

    template<typename DescriptorType, typename ...Args>
    static void Convert(DescriptorType desc, VkDescriptorSetLayoutBinding binding, Args&&... args)
    {
        desc["binding"_sk] = binding.binding;
        desc["descriptorCount"_sk] = binding.descriptorCount;
        desc["descriptorType"_sk] = binding.descriptorType;
        desc["stageFlags"_sk] = binding.stageFlags;

        return;
    }
};

template<>
struct dg::CreateTraits<DescriptorSetLayout>
{
    template<typename CreateArgs, typename DependencyProvider>
    static dg::Container Create(CreateArgs tree, DependencyProvider&& prov)
    {
        auto& vulkanContext = prov.Acquire<VulkanContext>("vulkanContext");
         
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(hv::AcquireArraySize(tree));
        for (auto tree : hv::Array(tree)) bindings.push_back(tree.Acquire<VkDescriptorSetLayoutBinding>());

        return dg::Container::Create<DescriptorSetLayout>(std::span(bindings), vulkanContext);
    }
};

struct PipelineLayout : VulkanResource, MoveConstructOnly
{
    VkPipelineLayout layout;

    PipelineLayout(std::span<VkDescriptorSetLayout> setLayouts, std::span<VkPushConstantRange> pushConstantRanges, VulkanContext& context) : VulkanResource(context), layout(VK_NULL_HANDLE)
    {
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;

        pipelineLayoutInfo.setLayoutCount = setLayouts.size();
        pipelineLayoutInfo.pSetLayouts = setLayouts.data();

        pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges.data();
        pipelineLayoutInfo.pushConstantRangeCount = pushConstantRanges.size();

        vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &layout);
    }

    auto Handle() const
    {
        return layout;
    }

    ~PipelineLayout()
    {
        vkDestroyPipelineLayout(context.device, layout, nullptr);
    }
};

template<>
struct hv::conversion::HvToType<VkPushConstantRange>
{
    static inline constexpr bool UseBinary = false;

    template<typename DescriptorType, typename ...Args>
    static VkPushConstantRange Convert(DescriptorType desc, Args&&... args)
    {
        VkPushConstantRange range{};

        range.offset = desc["offset"_sk];
        range.size = desc["size"_sk];
        range.stageFlags = desc["stageFlags"_sk];

        return range;
    }
};
template<>
struct hv::conversion::TypeToHv<VkPushConstantRange>
{
    static inline constexpr bool UseBinary = false;

    template<typename DescriptorType, typename ...Args>
    static void Convert(DescriptorType desc, VkPushConstantRange range, Args&&... args)
    {
        desc["offset"_sk] = range.offset;
        desc["size"_sk] = range.size;
        desc["stageFlags"_sk] = range.stageFlags;
    }
};


template<>
struct dg::CreateTraits<PipelineLayout>
{
    template<typename CreateArgs, typename DependencyProvider>
    static dg::Container Create(CreateArgs tree, DependencyProvider&& prov)
    {
        auto& vulkanContext = prov.Acquire<VulkanContext>("vulkanContext");

        auto setLayoutGraphHandles = tree["setLayouts"_sk];
        std::vector<VkDescriptorSetLayout> setLayouts;
        setLayouts.reserve(hv::AcquireArraySize(setLayoutGraphHandles));
        for (auto tree : hv::Array(setLayoutGraphHandles)) setLayouts.push_back( prov.Acquire<DescriptorSetLayout>(tree.Acquire<std::string>()).Handle() );

        auto pushConstantRangesTree = tree["pushConstantRanges"_sk];
        std::vector<VkPushConstantRange> pushConstantRanges;
        pushConstantRanges.reserve(hv::AcquireArraySize(pushConstantRangesTree));
        for (auto tree : hv::Array(pushConstantRangesTree)) pushConstantRanges.push_back(tree.Acquire<VkPushConstantRange>());
        
        return dg::Container::Create<PipelineLayout>(std::span(setLayouts), std::span(pushConstantRanges), vulkanContext);
    }
};

struct GraphicsPipeline : VulkanResource, NonCopyable
{
    VkPipeline pipeline;

public:

    struct CreateInfo
    {
        VkPipelineInputAssemblyStateCreateInfo inputAssembly =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE
        };
        VkPipelineViewportStateCreateInfo viewportState =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .pViewports = nullptr,
            .scissorCount = 1,
            .pScissors = nullptr
        };
        VkPipelineRasterizationStateCreateInfo rasterization =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,

        };
        VkPipelineMultisampleStateCreateInfo multisampling =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable = VK_FALSE,
        };

        std::vector<VkPipelineColorBlendAttachmentState> colorBlendAttachments;
        
        VkPipelineColorBlendStateCreateInfo colorBlending =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = VK_FALSE,
            .logicOp = VK_LOGIC_OP_COPY,
            .attachmentCount = 0,
            .pAttachments = nullptr,
            .blendConstants = {0.f, 0.f, 0.f, 0.f}
        };

        std::vector<VkDynamicState> dynamicStates =
        {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDepthStencilStateCreateInfo depthStencil =
        {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS,
            .depthBoundsTestEnable = VK_FALSE,
            .stencilTestEnable = VK_FALSE
        };

        std::vector<VkVertexInputAttributeDescription> vertexAttributes;
        std::vector<VkVertexInputBindingDescription> vertexBindings;

        std::vector<VkFormat> renderTargerFormats;
        VkFormat depthStencilFormat;

        CreateInfo()
        {

        }

        CreateInfo(const CreateInfo& rhs) : inputAssembly(rhs.inputAssembly),
            viewportState(rhs.viewportState),
            rasterization(rhs.rasterization),
            multisampling(rhs.multisampling),
            colorBlendAttachments(rhs.colorBlendAttachments),
            colorBlending(rhs.colorBlending),
            dynamicStates(rhs.dynamicStates),
            depthStencil(rhs.depthStencil),
            vertexAttributes(rhs.vertexAttributes),
            vertexBindings(rhs.vertexBindings),
            renderTargerFormats(rhs.renderTargerFormats),
            depthStencilFormat(rhs.depthStencilFormat)
        {
            FixReferences();
        }

        void FixReferences()
        {
            colorBlending.pAttachments = colorBlendAttachments.data();
            colorBlending.attachmentCount = colorBlendAttachments.size();
        }

        CreateInfo& operator=(const CreateInfo&) = delete;
    };

    GraphicsPipeline(VulkanContext& context, PipelineLayout& pipelineLayout, const CreateInfo& createInfo) : VulkanResource(context)
    {
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        vertexInputInfo.vertexBindingDescriptionCount = createInfo.vertexBindings.size();
        vertexInputInfo.pVertexBindingDescriptions = createInfo.vertexBindings.data();

        vertexInputInfo.vertexAttributeDescriptionCount = createInfo.vertexAttributes.size();
        vertexInputInfo.pVertexAttributeDescriptions = createInfo.vertexAttributes.data();

        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(createInfo.dynamicStates.size());
        dynamicState.pDynamicStates = createInfo.dynamicStates.data();


        std::vector<std::string> temp(descCount);
        std::vector<VkPipelineShaderStageCreateInfo> shaderStageCreateInfos(descCount);
        for (std::size_t i = 0; i < shaderStageCreateInfos.size(); i++)
        {
            VkPipelineShaderStageCreateInfo createInfo;
            createInfo.stage = tree["stages"][std::to_string(i)]["stageType"].Acquire<VkShaderStageFlagBits>().value();
            temp[i] = tree["stages"][std::to_string(i)]["mainFunction"].Acquire<std::string>().value();
            auto& f = temp[i];
            createInfo.pName = f.c_str();
            createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            createInfo.pNext = nullptr;
            createInfo.pSpecializationInfo = nullptr;
            createInfo.flags = 0;

            shaderStageCreateInfos[i] = createInfo;

            auto code = tree["stages"][std::to_string(i)]["code"].Acquire<HvRawBinaryView>().value();

            shaderStageCreateInfos[i].module = createShaderModule(context, code.data, code.size);
        }


        VkPipelineRenderingCreateInfo pipelineRenderingCreateInfo{};
        pipelineRenderingCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        pipelineRenderingCreateInfo.colorAttachmentCount = createInfo.renderTargerFormats.size();
        pipelineRenderingCreateInfo.pColorAttachmentFormats = createInfo.renderTargerFormats.data();
        pipelineRenderingCreateInfo.depthAttachmentFormat = createInfo.depthStencilFormat;
        pipelineRenderingCreateInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = shaderStageCreateInfos.size();
        pipelineInfo.pStages = shaderStageCreateInfos.data();
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &createInfo.inputAssembly;
        pipelineInfo.pViewportState = &createInfo.viewportState;
        pipelineInfo.pRasterizationState = &createInfo.rasterization;
        pipelineInfo.pMultisampleState = &createInfo.multisampling;
        pipelineInfo.pColorBlendState = &createInfo.colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = pipelineLayout.Handle();
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
        pipelineInfo.pDepthStencilState = &createInfo.depthStencil;

        pipelineInfo.pNext = &pipelineRenderingCreateInfo;

        if (vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }
    }

    void Bind(VkCommandBuffer commandBuffer)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    }

    ~GraphicsPipeline()
    {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
};


int main()
{
    //std::cout << std::setprecision(16) << IntegrateQuad([](double x) { return x * (8.0 - x); }, 0, 8, 2'000'000.0) << "    " << IntegrateZhopa() << "\n\n\n\n";
    using PesHandle = hv::handle::hierarchical::FixedCapacity<std::uint64_t, 8>;
    using PesStorage = hv::data::DedicatedAllocationStorage<PesHandle>;
    //std::cout << std::setprecision(16) << IntegrateQuad([](double x) { return L::Checked(x, 0.325, 0.34); }, -hpi, hpi, 2'000'000.0) << "\n\n\n\n";
    //std::cout << std::setprecision(16) << IntegrateQuad([](double x) { return L::Checked(x, 0.325, 0.34); }, 0, 8) << "\n\n\n\n";

    dg::Graph<std::string, hv::Descriptor<PesHandle, PesStorage>> dependencyGraph;
    PesStorage treeData;
    hv::Descriptor root(treeData);
    
    volkInitialize();
    
    GLFW::Context glfwContext;

    auto windowRef = dependencyGraph.EmplaceNode<GLFW::Window>("window");

    GLFW::Window &window = *windowRef;

    glfwSetInputMode(window.Handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetInputMode(window.Handle(), GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

    window.BindKeyCallback(key_callback);
    
    root["createArgs"_sk / "vulkanContext"_sk] = "window"sv;
    dependencyGraph.SetDependencies("vulkanContext", root["createArgs"_sk / "vulkanContext"_sk]);

    auto contextRef = dependencyGraph.Acquire<VulkanContext>("vulkanContext");

    VulkanContext &context = *contextRef;

    camera.fov = 90.f;
    camera.UpdateProjection(glm::uvec2(1920, 1080));
    lastTime = std::chrono::high_resolution_clock::now();

    CommandPool tempCommandPool(context);
    TempCommandBuffer::tempCommandPool = &tempCommandPool;

    AssetManager assetManager(context);

    std::vector<Semaphore> imageAvailableSemaphores;
    std::vector<Semaphore> renderFinishedSemaphores;
    std::vector<Fence> inFlightFences;
    uint32_t currentFrame = 0;

    imageAvailableSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);
    for (std::size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    {
        imageAvailableSemaphores.emplace_back(context);
        renderFinishedSemaphores.emplace_back(context);
        inFlightFences.emplace_back(context, true);
    }

         
    std::array bindings = { Vertex::getBindingDescription() };
    std::array attributes = { Vertex::getAttributeDescriptions() };

    constexpr std::size_t renderTargetCount = 8;
    constexpr VkFormat renderTargetFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    std::vector formats(std::from_range, std::views::repeat(renderTargetFormat, renderTargetCount));

    assetManager.instanceData["mat/formats"_tp].StoreArray<VkFormat>(std::span(formats));
    assetManager.instanceData["mat/depthFormat"_tp].Store(findDepthFormat(context));
    
    assetManager.instanceData["textures/srgb/format"_tp].Store(VK_FORMAT_R8G8B8A8_SRGB);
    assetManager.instanceData["textures/linear/format"_tp].Store(VK_FORMAT_R8G8B8A8_UNORM);
    assetManager.instanceData["textures/float/format"_tp].Store(VK_FORMAT_R16G16B16A16_SFLOAT);

    assetManager.instanceData["textures/srgb/genMipLevels"_tp].Store(0);
    assetManager.instanceData["textures/linear/genMipLevels"_tp].Store(0);
    assetManager.instanceData["textures/float/genMipLevels"_tp].Store(0);

    HvTree tree;
    
    Scene scene;
    scene.LoadFromFile("models/lvl1playertonk.obj");
    StoreMesh(tree["apa"], scene.meshes[0]);
     
    Texture tex;
    tex.LoadFromFile("models/lvl1playertonk.png");
    StoreTexture(tree["hui"], tex);

    Texture etcEqui;
    etcEqui.LoadFromFile("textures/etc/h.png", true);
    StoreTexture(tree["etc/src"_tp], etcEqui);

    Texture etcEquiHdr;
    etcEquiHdr.LoadFromFile("textures/etc/hh.hdr", true);
    StoreTexture(tree["etc/srcHdr"_tp], etcEquiHdr);

    Scene sphere; 
    sphere.LoadFromFile("models/sphere.fbx");
    //StoreMesh(tree["sphere"], sphere.meshes[0]);

    Mesh sphereMesh;
    GenerateSphere(sphereMesh);
    StoreMesh(tree["sphere"], sphereMesh);

    Mesh planeMesh;
    GeneratePlane(planeMesh, glm::vec2(1.f, 1.f));
    StoreMesh(tree["plane"], planeMesh);

    Scene scene3;
    scene3.LoadFromFile("models/retardedRoom.obj");
    StoreMesh(tree["room"], scene3.meshes[0]);

    Texture texr;
    texr.LoadFromFile("models/lvl1playertonk.png");
    StoreTexture(tree["roomTexture"], texr);

    Material::PipelineCreateInfo materialCreateInfo;
    std::vector<ShaderStageDescriptor> stagesHui = { {.stage = VK_SHADER_STAGE_VERTEX_BIT, .source = "./shaders/vertTest.spv" }, {.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .source = "./shaders/fragTest.spv" } };

    StoreMaterial(tree["mat"], stagesHui, materialCreateInfo, attributes, bindings);

    materialCreateInfo.rasterization.polygonMode = VK_POLYGON_MODE_LINE;

    StoreMaterial(tree["mat2"], stagesHui, materialCreateInfo, attributes, bindings);
    
    materialCreateInfo.rasterization.polygonMode = VK_POLYGON_MODE_POINT;

    StoreMaterial(tree["mat3"], stagesHui, materialCreateInfo, attributes, bindings);


    Material::PipelineCreateInfo materialCreateInfoZhopa;
    std::vector<ShaderStageDescriptor> stagesHuiGovno = { {.stage = VK_SHADER_STAGE_VERTEX_BIT, .source = "./shaders/vertInstanced.spv" }, {.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .source = "./shaders/fragInstanced.spv" } };
    StoreMaterial(tree["ins"], stagesHuiGovno, materialCreateInfoZhopa, attributes, bindings);



    Material::PipelineCreateInfo materialCreateInfoZalupa;
    std::vector<ShaderStageDescriptor> stagesHuiZalupa = { {.stage = VK_SHADER_STAGE_VERTEX_BIT, .source = "./shaders/graphics/part.spv.vert" }, {.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .source = "./shaders/graphics/part.spv.frag" } };
    StoreMaterial(tree["part"], stagesHuiZalupa, materialCreateInfoZalupa, attributes, bindings);

    StoreComputePipeline(tree["comp/partHui"_tp], { .stage = VK_SHADER_STAGE_COMPUTE_BIT, .source = "./shaders/compute/particlesHui.spv.comp" });
    StoreComputePipeline(tree["comp/equiToCube"_tp], { .stage = VK_SHADER_STAGE_COMPUTE_BIT, .source = "./shaders/compute/equiToCube.spv.comp" });
    StoreComputePipeline(tree["comp/diffuseIrradiance"_tp], { .stage = VK_SHADER_STAGE_COMPUTE_BIT, .source = "./shaders/compute/diffuseIrradiance.spv.comp" });
    StoreComputePipeline(tree["comp/iblConvolution"_tp], { .stage = VK_SHADER_STAGE_COMPUTE_BIT, .source = "./shaders/compute/iblConvolution.spv.comp" });
    StoreComputePipeline(tree["comp/brdfLUT"_tp], { .stage = VK_SHADER_STAGE_COMPUTE_BIT, .source = "./shaders/compute/computeBrdfLUT.spv.comp" });
    StoreComputePipeline(tree["comp/radiantIntensity"_tp], { .stage = VK_SHADER_STAGE_COMPUTE_BIT, .source = "./shaders/compute/radiantIntensity.spv.comp" });
    StoreComputePipeline(tree["comp/lobeSolidAngle"_tp], { .stage = VK_SHADER_STAGE_COMPUTE_BIT, .source = "./shaders/myakishIBL/lobeSolidAngle.spv.comp" });
    StoreComputePipeline(tree["comp/averagedBRDF"_tp], { .stage = VK_SHADER_STAGE_COMPUTE_BIT, .source = "./shaders/myakishIBL/averagedBRDF.spv.comp" });
    StoreComputePipeline(tree["comp/fullscreenPass"_tp], { .stage = VK_SHADER_STAGE_COMPUTE_BIT, .source = "./shaders/compute/fullscreenPass.spv.comp" });

    Material::PipelineCreateInfo materialCreateInfoPiska;
    //materialCreateInfoPiska.rasterization.polygonMode = VK_POLYGON_MODE_LINE;
    //materialCreateInfoPiska.rasterization.cullMode = VK_CULL_MODE_NONE;
    //std::vector<ShaderStageDescriptor> stagesHuiPiska = { {.stage = VK_SHADER_STAGE_VERTEX_BIT, .source = "./shaders/pbr/pbr.spv.vert" }, {.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .source = "./shaders/pbr/pbr.spv.frag" } };
    std::vector<ShaderStageDescriptor> stagesHuiPiska = { {.stage = VK_SHADER_STAGE_VERTEX_BIT, .source = "./shaders/wet/wet.spv.vert" }, {.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .source = "./shaders/wet/wet.spv.frag" } };
    StoreMaterial(tree["pbr/mat"_tp], stagesHuiPiska, materialCreateInfoPiska, attributes, bindings);

    std::array<VkVertexInputBindingDescription, 0> nullBindings;
    std::array<VkVertexInputAttributeDescription, 0> nullAttributes;

    Material::PipelineCreateInfo materialCreateInfoGraph;
    materialCreateInfoGraph.rasterization.cullMode = VK_CULL_MODE_NONE;
    std::vector<ShaderStageDescriptor> stagesGraph = { {.stage = VK_SHADER_STAGE_VERTEX_BIT, .source = "./shaders/textureGraph/graph.spv.vert" }, {.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .source = "./shaders/textureGraph/graph.spv.frag" } };
    StoreMaterial(tree["graph/mat"_tp], stagesGraph, materialCreateInfoGraph, nullAttributes, nullBindings);

    Material::PipelineCreateInfo materialCreateEtc;
    materialCreateEtc.depthStencil.depthTestEnable = VK_FALSE;
    std::vector<ShaderStageDescriptor> stagesEtc = { {.stage = VK_SHADER_STAGE_VERTEX_BIT, .source = "./shaders/equiToCube/etc.spv.vert" }, {.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .source = "./shaders/equiToCube/etc.spv.frag" }, {.stage = VK_SHADER_STAGE_GEOMETRY_BIT, .source = "./shaders/equiToCube/etc.spv.geom" } };
    StoreMaterial(tree["etc/mat"_tp], stagesEtc, materialCreateEtc, attributes, bindings);

    Material::PipelineCreateInfo materialCreateCbr;
    materialCreateCbr.rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    std::vector<ShaderStageDescriptor> stagesCbr = { {.stage = VK_SHADER_STAGE_VERTEX_BIT, .source = "./shaders/cubemapRender/cbm.spv.vert" }, {.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .source = "./shaders/cubemapRender/cbm.spv.frag" } };
    StoreMaterial(tree["cbr/mat"_tp], stagesCbr, materialCreateCbr, attributes, bindings);

    Material::PipelineCreateInfo materialCreateLsr;
    std::vector<ShaderStageDescriptor> stagesLsr = { {.stage = VK_SHADER_STAGE_VERTEX_BIT, .source = "./shaders/lightSourceRender/lsr.spv.vert" }, {.stage = VK_SHADER_STAGE_FRAGMENT_BIT, .source = "./shaders/lightSourceRender/lsr.spv.frag" }};
    StoreMaterial(tree["lsr/mat"_tp], stagesLsr, materialCreateLsr, attributes, bindings);

    Texture baseColor;
    baseColor.LoadFromFile("./textures/pbr/rustediron2_basecolor.png");
    StoreTexture(tree["pbr/baseColor"_tp], baseColor);

    Texture metallic;
    metallic.LoadFromFile("./textures/pbr/rustediron2_metallic.png");
    StoreTexture(tree["pbr/metallic"_tp], metallic);

    Texture roughness;
    roughness.LoadFromFile("./textures/pbr/rustediron2_roughness.png");
    StoreTexture(tree["pbr/roughness"_tp], roughness);

    Texture normal;
    normal.LoadFromFile("./textures/pbr/rustediron2_normal.png");
    StoreTexture(tree["pbr/normal"_tp], normal);

    Texture ambientOcclusion;
    ambientOcclusion.LoadFromFile("./textures/pbr/rustediron2_ao.png");
    StoreTexture(tree["pbr/ambientOcclusion"_tp], ambientOcclusion);


    SaveHvToFile(tree, "huinya.ab");

    



    AssetReference<Material> pbrMaterial (assetManager, AssetDescriptor("huinya.ab", "pbr/mat"_tp), "mat"_tp);

    HDescriptorPool pool(context, pbrMaterial->setLayouts[0]);
    HDescriptorSets sets(pool);

    UniformBuffer uniformBuffer(context, sizeof(UniformBufferPBR));
       
    auto planeMeshDescriptor = AssetDescriptor{ "huinya.ab", "plane"_tp };
    auto sphereMeshDescriptor = AssetDescriptor{ "huinya.ab", "sphere"_tp };

    auto baseColorDesc = AssetDescriptor{ "huinya.ab", "pbr/baseColor"_tp };
    auto metallicDesc = AssetDescriptor{ "huinya.ab", "pbr/metallic"_tp };
    auto roughnessDesc = AssetDescriptor{ "huinya.ab", "pbr/roughness"_tp };
    auto normalDesc = AssetDescriptor{ "huinya.ab", "pbr/normal"_tp };
    auto ambientOcclusionDesc = AssetDescriptor{ "huinya.ab", "pbr/ambientOcclusion"_tp };

    AssetReference<HTexture> baseColorTextureReference(assetManager, baseColorDesc, "textures/linear"_tp);
    AssetReference<HTexture> metallicTextureReference(assetManager, metallicDesc, "textures/linear"_tp);
    AssetReference<HTexture> roughnessTextureReference(assetManager, roughnessDesc, "textures/linear"_tp);
    AssetReference<HTexture> normalTextureReference(assetManager, normalDesc, "textures/linear"_tp);
    AssetReference<HTexture> ambientOcclusionTextureReference(assetManager, ambientOcclusionDesc, "textures/linear"_tp);


    Transformable t;

    CommandPool commandPool(context);

    std::vector<CommandBuffer> commandBuffers;
    commandBuffers.reserve(MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        commandBuffers.emplace_back(commandPool);

    camera.SetPosition(glm::vec3(0, 0, 0));

    MeshInstance pbrMesh(assetManager, planeMeshDescriptor, pbrMaterial.FullDescriptor(), sets);

    /*int lightCount = 8;
    std::vector<PointLight> lights(lightCount);
    for (int i = 0; i < lightCount; i++)
    {
        lights[i].position = glm::vec4(std::cos(i) * 4, std::sin(i) * 4, 0, 1);
        lights[i].color = glm::vec4(23.47, 21.31, 20.79, 0.0);
    }*/

    int lightCount = 1;
    std::vector<PointLight> lights(lightCount);
    lights[0].position = glm::vec4(1.f, 0.f, 1.f, 1.f);
    lights[0].color = glm::vec4(23.47, 21.31, 20.79, 0.0);

    StagedBuffer lightsBuffer(context, lights.size() * sizeof(PointLight), lights.data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
       
       
        
    auto etcSrcTexDsc = AssetDescriptor{ "huinya.ab", "etc/src"_tp };

    AssetReference<HTexture> etcSrcTex(assetManager, etcSrcTexDsc, "textures/float"_tp);

    Image::CreateInfo ci{};
    ci.arrayLayers = 6;
    ci.createFlags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    ci.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ci.mipLevels = Image::CreateInfo::MipLevelsDeduceFromSize;
    ci.size = glm::uvec3(2048, 2048, 1);
    ci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    Image environmentMap(context, ci);

    ci.size = glm::uvec3(256, 256, 1);
    Image specularEnvironmentMap(context, ci);

    ci.mipLevels = 1;
    Image irradianceMap(context, ci);

    Image::CreateInfo brdfLUTci{};
    brdfLUTci.imageUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    brdfLUTci.mipLevels = 1;
    brdfLUTci.size = glm::uvec3(256, 256, 1);
    brdfLUTci.format = VK_FORMAT_R32G32_SFLOAT;

    brdfLUTci.format = VK_FORMAT_R32_SFLOAT;

    brdfLUTci.format = VK_FORMAT_R32G32B32A32_SFLOAT;

   // RenderEquirectangularToCubemap(etcSrcTex->image, cubemap);
    EquirectangularToCubemapCompute(assetManager, context, etcSrcTex->image, environmentMap);
                      
    
    TempCommandBuffer cmd(context);
     
    environmentMap.GenerateMipMapChain(cmd, 0, environmentMap.mipLevels - 1);
    cmd.ExecuteAndReset();

    environmentMap.TransitionLayout(cmd.buffer, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    irradianceMap.TransitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL);

    cmd.ExecuteAndReset();
    
    AssetReference<ComputePipeline> diffuseIrradianceCompute(assetManager, AssetInstanceDescriptor{ AssetDescriptor{"huinya.ab", "comp/diffuseIrradiance"_tp}, "mat"_tp });
    
    HDescriptorPool diPool(context, diffuseIrradianceCompute->setLayouts[0]);
    HDescriptorSets diSets(diPool);

    ImageView environmentMapView(environmentMap, VK_IMAGE_VIEW_TYPE_CUBE);
    Sampler cubemapSampler(context);

    ImageView irradianceMapView(irradianceMap, VK_IMAGE_VIEW_TYPE_CUBE);
    diSets.Update(Describe(environmentMapView, cubemapSampler), Describe(irradianceMapView));

    diffuseIrradianceCompute->Bind(cmd);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, diffuseIrradianceCompute->pipelineLayout, 0, 1, diSets.descriptorSets.data(), 0, nullptr);
    vkCmdDispatch(cmd, irradianceMap.size.x / 32, irradianceMap.size.y / 32, 6);

    cmd.ExecuteAndReset();

    irradianceMap.TransitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    cmd.ExecuteAndReset();

    RadiantIntensity(assetManager, context, environmentMap, specularEnvironmentMap);
    specularEnvironmentMap.TransitionLayout(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    cmd.ExecuteAndReset();
    
    AssetReference<Material> cbrMat(assetManager, AssetInstanceDescriptor{ AssetDescriptor{"huinya.ab", "cbr/mat"_tp}, "mat"_tp });

    HDescriptorPool cbrPool(context, cbrMat->setLayouts[0]);
    HDescriptorSets cbrSets(cbrPool);
        
    UniformBuffer cbrUniformBuffer(context, sizeof(UniformBufferPBR));
        
    //cbrSets.Update(Describe(cbrUniformBuffer), Describe(cubemapView, cubemapSampler));

    ImageView specularEnvView(specularEnvironmentMap, VK_IMAGE_VIEW_TYPE_CUBE);

    cbrSets.Update(Describe(cbrUniformBuffer), Describe(specularEnvView, cubemapSampler));
    Transformable cbrTransformable;
    cbrTransformable.SetScale(glm::vec3(1000, 1000, 1000));
    MeshInstance cubemapMesh(assetManager, sphereMeshDescriptor, cbrMat.FullDescriptor(), cbrSets);
        
    Image trueIntegral = LoadRaw(context, "actual20000threshold1e-3.raw", glm::uvec3(64, 64, 1), 2);

    ImageView integratedBRDF(trueIntegral);

    Image specularLobeAngleCPU = LoadRaw(context, "sla_middle.raw", glm::uvec3(64, 64, 1), 2);
    //Image specularLobeAngleCPU2 = LoadRaw("sla_middle.raw", glm::uvec3(256, 256, 256), 2);
    ImageView lobeAngles(specularLobeAngleCPU);

    TempCommandBuffer myakishIBLtransition(context);

    trueIntegral.TransitionLayout(myakishIBLtransition, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    specularLobeAngleCPU.TransitionLayout(myakishIBLtransition, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            
    myakishIBLtransition.Execute();

    sets.Update(Describe(uniformBuffer), Describe(baseColorTextureReference), Describe(normalTextureReference), Describe(metallicTextureReference), Describe(roughnessTextureReference), Describe(ambientOcclusionTextureReference), Describe(irradianceMapView, cubemapSampler), Describe(specularEnvView, cubemapSampler), Describe(lobeAngles, cubemapSampler), Describe(integratedBRDF, cubemapSampler));
                
    AssetReference<Material> lsrMaterial(assetManager, AssetDescriptor("huinya.ab", "lsr/mat"_tp), "mat"_tp);

    HDescriptorPool lsrPool(context, lsrMaterial->setLayouts[0]);
    HDescriptorSets lsrSets(lsrPool);

    UniformBuffer lsrUniformBuffer(context, sizeof(UniformBufferPBR));
    lsrSets.Update(Describe(lsrUniformBuffer));

    MeshInstance lsrMesh(assetManager, sphereMeshDescriptor, lsrMaterial.FullDescriptor(), lsrSets);

    Transformable lsrTransformable;

    lsrTransformable.SetScale(glm::vec3(0.05f));

    Swapchain swapchain(context, window);
    bool framebufferResized = false;
    window.BindFramebufferSizeCallback([&](glm::ivec2) { framebufferResized = true; });


    glm::uvec3 colorImageSize = glm::uvec3(1920, 1080, 1);

    Image::CreateInfo colorImageCI{};
    colorImageCI.size = colorImageSize;
    colorImageCI.format = renderTargetFormat;
    colorImageCI.mipLevels = 1;
    colorImageCI.arrayLayers = renderTargetCount;
    colorImageCI.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    
    Image colorImage(context, colorImageCI);

    std::vector<ImageView> colorImageViews;
    colorImageViews.reserve(renderTargetCount);
    for (auto i = 0; i < renderTargetCount; i++)
    {
        colorImageViews.emplace_back(context, colorImage.WholeLayer(i));
    }

    //ImageView colorImageView(colorImage);


    VkFormat depthFormat = findDepthFormat(context); 
    Image::CreateInfo depthImageCI{};
    depthImageCI.size = colorImageSize;
    depthImageCI.format = depthFormat;
    depthImageCI.mipLevels = 1;
    depthImageCI.imageUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    Image depthImage(context, depthImageCI);
    ImageView depthImageView(depthImage);

    TempCommandBuffer depthImageTransition(context);

    depthImage.TransitionLayout(depthImageTransition, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    depthImageTransition.Execute();

    RenderPass renderPass(Rect<std::uint32_t>{{0, 0}, colorImageSize}, std::vector(std::from_range, colorImageViews | std::views::transform([](auto& imageView) -> RenderPassAttachment { return RenderPassAttachment(imageView); })), RenderPassAttachment(depthImageView, ClearValue(1.f), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL));
 
    

    AssetReference<ComputePipeline> fullscreenPassPipeline(assetManager, AssetInstanceDescriptor{ AssetDescriptor{"huinya.ab", "comp/fullscreenPass"_tp}, ""_tp });

    HDescriptorPool fullscreenPassPool(context, fullscreenPassPipeline->setLayouts[0]);
    HDescriptorSets fullscreenPassSets(fullscreenPassPool);

    fullscreenPassSets.Update(Describe(colorImageViews[0]), Describe(swapchain));

    

    AssetReference<Material> graphMaterial(assetManager, AssetDescriptor("huinya.ab", "graph/mat"_tp), "mat"_tp);

    HDescriptorPool graphPool(context, graphMaterial->setLayouts[0]);
    HDescriptorSets graphSets(graphPool);

    UniformBuffer graphUniformBuffer(context, sizeof(UniformBufferPBR));

    graphSets.Update(Describe(graphUniformBuffer), Describe(lobeAngles, cubemapSampler));

    Transformable graphTransformable;

    std::uint32_t graphResolution = 32;

    while (!window.ShouldClose())
    {                   
        glfwPollEvents();
        
        inFlightFences[currentFrame].Wait();
        
        uint32_t imageIndex = swapchain.AcquireNextImage(imageAvailableSemaphores[currentFrame], window);

        hinput(window);

        inFlightFences[currentFrame].Reset();

        commandBuffers[currentFrame].Reset();
        {
            commandBuffers[currentFrame].Begin();

            /*VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = renderPass;
            renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
            renderPassInfo.renderArea.offset = { 0, 0 };
            renderPassInfo.renderArea.extent = swapChainExtent;

            std::array<VkClearValue, 2> clearValues{};
            clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
            clearValues[1].depthStencil = { 1.0f, 0 };

            renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
            renderPassInfo.pClearValues = clearValues.data();

            vkCmdBeginRenderPass(commandBuffers[currentFrame].buffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);*/

            PipelineBarrierData colorImageInitialTransitionBarrier{};
            colorImageInitialTransitionBarrier.srcAccessMask = VK_ACCESS_NONE;
            colorImageInitialTransitionBarrier.sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            colorImageInitialTransitionBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            colorImageInitialTransitionBarrier.destinationStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

            colorImage.TransitionLayout(commandBuffers[currentFrame], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
     
            {
                renderPass.Begin(commandBuffers[currentFrame].buffer);

                VkViewport viewport{};
                viewport.x = 0.0f;
                viewport.y = 0.0f;
                viewport.width = (float)swapchain.extent.width;
                viewport.height = (float)swapchain.extent.height;
                viewport.minDepth = 0.0f;
                viewport.maxDepth = 1.0f;
                vkCmdSetViewport(commandBuffers[currentFrame].buffer, 0, 1, &viewport);

                VkRect2D scissor{};
                scissor.offset = { 0, 0 };
                scissor.extent = swapchain.extent;
                vkCmdSetScissor(commandBuffers[currentFrame].buffer, 0, 1, &scissor);

                //main UBO
                {
                    UniformBufferPBR ubo{};
                    ubo.model = t.GetModelMatrix();
                    ubo.view = camera.view;
                    ubo.proj = camera.projection;

                    ubo.axisSwizzle = Camera::AxisSwizzle();

                    ubo.normal = glm::transpose(glm::inverse(ubo.model));

                    ubo.cameraPos = camera.GetPosition();

                    uniformBuffer.CopyFrom(std::move(ubo), imageIndex);
                }

                //main push constants
                {
                    VkBufferDeviceAddressInfo addressInfo{};
                    addressInfo.buffer = lightsBuffer.buffer.buffer;
                    addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;

                    auto deviceAddress = vkGetBufferDeviceAddress(context.device, &addressInfo);

                    PushConstantsPBR constants{};
                    constants.lightsBuffer = deviceAddress;
                    constants.count = lightCount;
                    constants.indirectMultiplier = ::indirectMultiplier;

                    vkCmdPushConstants(commandBuffers[currentFrame].buffer, pbrMaterial->pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, &constants);
                }

                //Cubemap UBO
                {
                    UniformBufferPBR ubo{};
                    ubo.model = cbrTransformable.GetModelMatrix();
                    ubo.view = camera.view;
                    ubo.proj = camera.projection;

                    ubo.axisSwizzle = Camera::AxisSwizzle();

                    ubo.normal = glm::transpose(glm::inverse(ubo.model));

                    ubo.cameraPos = camera.GetPosition();

                    cbrUniformBuffer.CopyFrom(std::move(ubo), imageIndex);
                }

                //Light source UBO
                {
                    UniformBufferPBR ubo{};
                    ubo.model = lsrTransformable.GetModelMatrix();
                    ubo.view = camera.view;
                    ubo.proj = camera.projection;

                    ubo.axisSwizzle = Camera::AxisSwizzle();

                    ubo.normal = glm::transpose(glm::inverse(ubo.model));

                    ubo.cameraPos = camera.GetPosition();

                    lsrUniformBuffer.CopyFrom(std::move(ubo), imageIndex);
                }

                //graph UBO
                {
                    UniformBufferPBR ubo{};
                    ubo.model = graphTransformable.GetModelMatrix();
                    ubo.view = camera.view;
                    ubo.proj = camera.projection;

                    ubo.axisSwizzle = Camera::AxisSwizzle();

                    ubo.normal = glm::transpose(glm::inverse(ubo.model));

                    ubo.cameraPos = camera.GetPosition();

                    graphUniformBuffer.CopyFrom(std::move(ubo), imageIndex);
                }

                RenderDrawable(commandBuffers[currentFrame].buffer, pbrMesh, currentFrame);

                RenderDrawable(commandBuffers[currentFrame].buffer, cubemapMesh, currentFrame);

                //light source push constants
                {
                    VkBufferDeviceAddressInfo addressInfo{};
                    addressInfo.buffer = lightsBuffer.buffer.buffer;
                    addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;

                    auto deviceAddress = vkGetBufferDeviceAddress(context.device, &addressInfo);

                    PushConstantsPBR constants{};
                    constants.lightsBuffer = deviceAddress;
                    constants.count = lightCount;

                    vkCmdPushConstants(commandBuffers[currentFrame].buffer, lsrMaterial->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 12, &constants);
                }

                RenderDrawable(commandBuffers[currentFrame], lsrMesh, lightCount, currentFrame);

                //graph push constants
                /*{
                    GraphPushConstants constants{};
                    constants.resolution = graphResolution;
                    constants.textureScale = 1.f;

                    vkCmdPushConstants(commandBuffers[currentFrame].buffer, graphMaterial->pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 8, &constants);
                }*/

                //dispatch graph draw command
                /*{
                    graphMaterial->Bind(commandBuffers[currentFrame]);

                    graphSets.Bind(commandBuffers[currentFrame], graphMaterial->pipelineLayout, imageIndex, VK_PIPELINE_BIND_POINT_GRAPHICS);

                    VkDeviceSize offsets[] = { 0 };

                    vkCmdDraw(commandBuffers[currentFrame].buffer, (graphResolution - 1) * 2 * graphResolution, 1, 0, 0);
                }*/


                renderPass.End(commandBuffers[currentFrame].buffer);
            }

            PipelineBarrierData colorImageTransitionBarrier{};
            colorImageTransitionBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            colorImageTransitionBarrier.sourceStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            colorImageTransitionBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            colorImageTransitionBarrier.destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            colorImage.TransitionLayout(commandBuffers[currentFrame], VK_IMAGE_LAYOUT_GENERAL, colorImageTransitionBarrier);

            PipelineBarrierData swapchainImageInitialTransitionBarrier{};
            swapchainImageInitialTransitionBarrier.srcAccessMask = VK_ACCESS_NONE;
            swapchainImageInitialTransitionBarrier.sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            swapchainImageInitialTransitionBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            swapchainImageInitialTransitionBarrier.destinationStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

            swapchain.imageRefs[imageIndex].TransitionLayout(commandBuffers[currentFrame], VK_IMAGE_LAYOUT_GENERAL, swapchainImageInitialTransitionBarrier);

            {
                fullscreenPassPipeline->Bind(commandBuffers[currentFrame]);
                fullscreenPassSets.Bind(commandBuffers[currentFrame], fullscreenPassPipeline->pipelineLayout, imageIndex, VK_PIPELINE_BIND_POINT_COMPUTE);

                vkCmdDispatch(commandBuffers[currentFrame], swapchain.extent.width / 32, swapchain.extent.height / 32, 1);
            }

            PipelineBarrierData swapchainImageTransitionBarrier{};
            swapchainImageTransitionBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            swapchainImageTransitionBarrier.sourceStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            swapchainImageTransitionBarrier.dstAccessMask = VK_ACCESS_NONE;
            swapchainImageTransitionBarrier.destinationStage = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;

            swapchain.imageRefs[imageIndex].TransitionLayout(commandBuffers[currentFrame], VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, swapchainImageTransitionBarrier);


            commandBuffers[currentFrame].End();
        }

        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };

        VkSubmitInfo graphicsSubmitInfo{};
        {
            graphicsSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
            graphicsSubmitInfo.waitSemaphoreCount = 1;
            graphicsSubmitInfo.pWaitSemaphores = waitSemaphores;
            graphicsSubmitInfo.pWaitDstStageMask = waitStages;

            graphicsSubmitInfo.commandBufferCount = 1;
            graphicsSubmitInfo.pCommandBuffers = &commandBuffers[currentFrame].buffer;

            graphicsSubmitInfo.signalSemaphoreCount = 1;
            graphicsSubmitInfo.pSignalSemaphores = signalSemaphores;
        }

        std::array submitInfos = { graphicsSubmitInfo };

        if (vkQueueSubmit(context.graphicsQueue, submitInfos.size(), submitInfos.data(), inFlightFences[currentFrame]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = { swapchain.swapchain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        
        presentInfo.pImageIndices = &imageIndex;
        
        VkResult result = vkQueuePresentKHR(context.presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
        {
            framebufferResized = false;
            swapchain.Recreate(window);
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("failed to present swap chain image!");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    vkDeviceWaitIdle(context.device);

    return 0;
}
/*catch (std::exception& ex)
{
    std::cerr << ex.what();
}*/