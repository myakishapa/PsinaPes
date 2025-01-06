#pragma once
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RIGHT_HANDED
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>

#include "Utility.h"

namespace GLFW
{
    class Scancode
    {
        int scancode;

    public:

        Scancode(int scancode) : scancode(scancode) {}

        std::string_view Name() const
        {
            auto cString = glfwGetKeyName(GLFW_KEY_UNKNOWN, scancode);
            return std::string_view(cString, std::strlen(cString));
        }

        friend auto operator<=>(Scancode, Scancode) = default;

        int Handle() const
        {
            return scancode;
        }
    };

    class Key
    {
        int key;

    public:

        Key(int key) : key(key) {}

        std::string_view Name() const
        {
            auto cString = glfwGetKeyName(key, -1);
            return std::string_view(cString, std::strlen(cString));
        }

        friend auto operator<=>(Key, Key) = default;

        Scancode GetScancode() const
        {
            return Scancode(glfwGetKeyScancode(key));
        }

        int Handle() const
        {
            return key;
        }

        friend auto operator<=>(Scancode code, Key key)
        {
            return key.GetScancode() = code;
        }
        friend auto operator<=>(Key key, Scancode code)
        {
            return key.GetScancode() = code;
        }
    };

    enum struct Modifier : int
    {
        SHIFT = GLFW_MOD_SHIFT,
        CONTROL = GLFW_MOD_CONTROL,
        ALT = GLFW_MOD_ALT,
        SUPER = GLFW_MOD_SUPER,
        CAPS_LOCK = GLFW_MOD_CAPS_LOCK,
        NUM_LOCK = GLFW_MOD_NUM_LOCK
    };

    enum struct Action
    {
        PRESS = GLFW_PRESS,
        RELEASE = GLFW_RELEASE,
        REPEAT = GLFW_REPEAT
    };



    class Window
    {
    public:

        enum class Flags
        {
            EMPTY = 0x0,
            RESIZABLE = 0x1,
            VISIBLE = 0x2,
            DECORATED = 0x4,
            FOCUSED = 0x8,
            AUTO_ICONIFY = 0x10,
            FLOATING = 0x20,
            MAXIMIZED = 0x40,
            CENTER_CURSOR = 0x80,
            //TRANSPARENT = 0x100,
            FOCUS_ON_SHOW = 0x200,
            SCALE_TO_MONITOR = 0x400,
            //SCALE_FRAMEBUFFER = 0x800,
            //MOUSE_PASSTHROUGH = 0x1000,
        };

        using enum Flags;
        static constexpr Flags DefaultFlags = FOCUSED | VISIBLE | DECORATED | RESIZABLE | AUTO_ICONIFY | CENTER_CURSOR | FOCUS_ON_SHOW/* | SCALE_FRAMEBUFFER*/;

        static constexpr int DontCare = GLFW_DONT_CARE;

        struct CreateInfo
        {
            Flags flags = DefaultFlags;

            glm::ivec2 size = glm::ivec2(1920, 1080);

            int refreshRate = DontCare;

            std::string_view title = "Vulkan";
        };

        using FramebufferSizeCallback = void(glm::ivec2);

        struct KeyCallbackArgs
        {
            Key key;
            Scancode code;
            Action action;
            Modifier mods;
        };
        using KeyCallback = void(KeyCallbackArgs);

    private:

        GLFWwindow* window;

        std::function<FramebufferSizeCallback> framebufferCallback;
        static void GenericFramebufferSizeCallback(GLFWwindow* window, int width, int height)
        {
            auto raiiWindow = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));
            glm::ivec2 size(width, height);

            raiiWindow->framebufferCallback(size);
        }

        std::function<KeyCallback> keyCallback;
        static void GenericKeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
        {
            auto raiiWindow = reinterpret_cast<Window*>(glfwGetWindowUserPointer(window));

            KeyCallbackArgs args{ key, scancode, static_cast<Action>(action), static_cast<Modifier>(mods) };

            raiiWindow->keyCallback(args);
        }

    public:

        Window(CreateInfo ci = {}) : window(nullptr)
        {
            glfwDefaultWindowHints();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

            glfwWindowHint(GLFW_RESIZABLE, ((ci.flags & RESIZABLE) != EMPTY) ? GLFW_TRUE : GLFW_FALSE);
            glfwWindowHint(GLFW_VISIBLE, ((ci.flags & VISIBLE) != EMPTY) ? GLFW_TRUE : GLFW_FALSE);
            glfwWindowHint(GLFW_DECORATED, ((ci.flags & DECORATED) != EMPTY) ? GLFW_TRUE : GLFW_FALSE);
            glfwWindowHint(GLFW_FOCUSED, ((ci.flags & FOCUSED) != EMPTY) ? GLFW_TRUE : GLFW_FALSE);
            glfwWindowHint(GLFW_AUTO_ICONIFY, ((ci.flags & AUTO_ICONIFY) != EMPTY) ? GLFW_TRUE : GLFW_FALSE);
            glfwWindowHint(GLFW_FLOATING, ((ci.flags & FLOATING) != EMPTY) ? GLFW_TRUE : GLFW_FALSE);
            glfwWindowHint(GLFW_MAXIMIZED, ((ci.flags & MAXIMIZED) != EMPTY) ? GLFW_TRUE : GLFW_FALSE);
            glfwWindowHint(GLFW_CENTER_CURSOR, ((ci.flags & CENTER_CURSOR) != EMPTY) ? GLFW_TRUE : GLFW_FALSE);
            glfwWindowHint(GLFW_FOCUS_ON_SHOW, ((ci.flags & FOCUS_ON_SHOW) != EMPTY) ? GLFW_TRUE : GLFW_FALSE);
            glfwWindowHint(GLFW_SCALE_TO_MONITOR, ((ci.flags & SCALE_TO_MONITOR) != EMPTY) ? GLFW_TRUE : GLFW_FALSE);
            //glfwWindowHint(GLFW_SCALE_FRAMEBUFFER, (ci.flags & SCALE_FRAMEBUFFER) == EMPTY ? GLFW_TRUE : GLFW_FALSE);
            //glfwWindowHint(GLFW_MOUSE_PASSTHROUGH, (ci.flags & MOUSE_PASSTHROUGH) == EMPTY ? GLFW_TRUE : GLFW_FALSE);

            glfwWindowHint(GLFW_REFRESH_RATE, ci.refreshRate);

            window = glfwCreateWindow(ci.size.x, ci.size.y, ci.title.data(), nullptr, nullptr);

            glfwSetWindowUserPointer(window, this);

            glfwSetFramebufferSizeCallback(window, GenericFramebufferSizeCallback);
        }
        Window(const Window&) = delete;
        Window(Window&& rhs) noexcept : window(rhs.window),
            framebufferCallback(std::move(rhs.framebufferCallback)),
            keyCallback(std::move(rhs.keyCallback))
        {
            rhs.window = nullptr;
        }

        Window& operator=(const Window&) = delete;
        Window& operator=(Window&& rhs) noexcept
        {
            window = rhs.window;
            rhs.window = nullptr;

            framebufferCallback = std::move(rhs.framebufferCallback);
            keyCallback = std::move(rhs.keyCallback);
        }

        ~Window()
        {
            std::println("~Window()");
            if (window) glfwDestroyWindow(window);
        }

        glm::ivec2 GetWindowSize() const
        {
            glm::ivec2 size{};
            glfwGetWindowSize(window, &size.x, &size.y);
            return size;
        }
        glm::ivec2 GetFramebufferSize() const
        {
            glm::ivec2 size{};
            glfwGetFramebufferSize(window, &size.x, &size.y);
            return size;
        }

        bool ShouldClose() const
        {
            return glfwWindowShouldClose(window);
        }

        GLFWwindow* Handle() const
        {
            return window;
        }

        glm::dvec2 GetCursorPosition() const
        {
            glm::dvec2 position{};
            glfwGetCursorPos(window, &position.x, &position.y);
            return position;
        }

        template<typename F>
        void BindFramebufferSizeCallback(F&& function)
        {
            framebufferCallback = std::forward<F>(function);
        }
        template<typename F>
        void BindKeyCallback(F&& function)
        {
            keyCallback = std::forward<F>(function);
        }

        Action GetKeyState(Key key) const
        {
            return static_cast<Action>(glfwGetKey(window, key.Handle()));
        }
    };
}
