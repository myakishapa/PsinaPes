#pragma once
#include <GLFW/glfw3.h>

#include "Utility.h"

namespace GLFW
{
    class Context : NonCopyable
    {
    public:

        Context()
        {
            glfwInit();
        }

        ~Context()
        {
            glfwTerminate();
        }
    };
}