#pragma once

#include <concepts>

struct MoveConstructOnly
{
    MoveConstructOnly() = default;
    MoveConstructOnly(const MoveConstructOnly&) = delete;

    MoveConstructOnly& operator=(const MoveConstructOnly&) = delete;
    MoveConstructOnly& operator=(MoveConstructOnly&&) = delete;
};

template<typename T>
class Delayed
{
    alignas(T) std::byte buffer[sizeof(T)];
    bool valid = false;

public:

    Delayed() {}
    template<typename ...Args> requires std::constructible_from<T, Args...>
    Delayed(Args... args)
    {
        Create(args...);
    }

    void Destroy()
    {
        if (valid) reinterpret_cast<T*>(&buffer)->~T();
        valid = false;
    }

    template<typename ...Args> requires std::constructible_from<T, Args...>
    void Create(Args&&... args)
    {
        Destroy();
        new(buffer) T(std::forward<Args>(args)...);
        valid = true;
    }

    bool IsValid() const
    {
        return valid;
    }

    operator T& ()
    {
        return *reinterpret_cast<T*>(&buffer);
    }
    operator const T& () const
    {
        return *reinterpret_cast<const T*>(&buffer);
    }

    T& Get()
    {
        return *reinterpret_cast<T*>(&buffer);
    }
    const T& Get() const
    {
        return *reinterpret_cast<const T*>(&buffer);
    }

    T* operator->()
    {
        return reinterpret_cast<T*>(&buffer);
    }
    const T* operator->() const
    {
        return reinterpret_cast<const T*>(&buffer);
    }

    T& operator*()
    {
        return Get();
    }
    const T& operator*() const
    {
        return Get();
    }

    ~Delayed()
    {
        Destroy();
    }
};


template<typename Type>
concept Enum = std::is_enum_v<Type>;

template<Enum Type>
constexpr Type operator|(Type lhs, Type rhs)
{
    return static_cast<Type>(std::to_underlying(lhs) | std::to_underlying(rhs));
}
template<Enum Type>
constexpr Type operator&(Type lhs, Type rhs)
{
    return static_cast<Type>(std::to_underlying(lhs) & std::to_underlying(rhs));
}

